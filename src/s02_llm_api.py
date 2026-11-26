"""Этап 2. LLM-разметка золота / серебра через provod.ai (OpenAI-совместимый /chat/completions).

Инструкция — та же, что у разметчика и локальной Qwen (s02_llm_annotate.INSTRUCTION). Модели передаются только текст.
Стоимость берётся из usage.cost (RUB) каждого ответа; общий расход = сумма по всем CSV в out_dir;
при достижении stop_at_rub новые запросы не отправляются. Повторный запуск дописывает недостающие uid.

Запуск:
    .venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key gold   [--model M] [--limit N]
    .venv/bin/python src/s02_llm_api.py configs/s02_llm_api.yaml --key silver [--model M] [--limit N]
    .venv/bin/python src/s02_llm_api.py configs/s02_llm_arcs.yaml --key arcs   (LLM-дуги, Ворота 2 п. 2.2)
    .venv/bin/python src/s02_llm_api.py configs/s05_ending.yaml   --key sent   (предложения концовок, Ворота 5 п. 2.2)
Ключ sent: строки с непустым context размечаются той же инструкцией с блоком «Предшествующий контекст (не оценивай)»;
строки без context — исходной инструкцией.
Выход: data/processed/s02/llm_api/{key}__{model}.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s02_llm_annotate import INSTRUCTION, parse  # noqa: E402

# та же инструкция + блок контекста перед фрагментом (Ворота 5, п. 2.2); оценивается только фрагмент
SENT_INSTRUCTION = INSTRUCTION.replace(
    "Фрагмент:\n",
    "Предшествующий контекст (только для понимания — его не оценивай):\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Фрагмент (оцени только его):\n")
assert SENT_INSTRUCTION != INSTRUCTION

FIELDS = ["uid", "tone", "emotion", "skip", "model", "model_returned", "prompt_tokens", "completion_tokens",
          "reasoning_tokens", "cost_rub", "finish_reason", "raw"]


def read_env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k in ("PROVOD_API_KEY", "PROVOD_BASE_URL"):
        env[k] = os.environ.get(k) or env.get(k)
    assert env["PROVOD_API_KEY"], "нет PROVOD_API_KEY в .env"
    return env


def spent_rub(out: Path) -> float:
    s = 0.0
    for f in glob.glob(str(out / "*.csv")):
        d = pd.read_csv(f, usecols=["cost_rub"])
        s += float(pd.to_numeric(d["cost_rub"], errors="coerce").fillna(0).sum())
    return s


def call(env, api, model, text, emotions, context=None):
    em = ", ".join(emotions)
    prompt = (SENT_INSTRUCTION.format(emotions=em, text=text, context=context) if context
              else INSTRUCTION.format(emotions=em, text=text))
    body = {"model": model, "temperature": api["temperature"], "max_tokens": api["max_tokens"],
            "reasoning": api["reasoning"],
            "messages": [{"role": "user", "content": prompt}]}
    data = json.dumps(body).encode()
    for attempt in range(api["retries"] + 1):
        req = urllib.request.Request(env["PROVOD_BASE_URL"].rstrip("/") + "/chat/completions", data=data,
                                     headers={"Authorization": f"Bearer {env['PROVOD_API_KEY']}",
                                              "Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=api["timeout_s"]).read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < api["retries"]:
                time.sleep(2 ** attempt + random.random())
                continue
            return {"error": f"HTTP {e.code}: {msg}"}
        except Exception as e:  # сеть / таймаут
            if attempt < api["retries"]:
                time.sleep(2 ** attempt + random.random())
                continue
            return {"error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--key", choices=["gold", "silver", "arcs", "sent"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None, help="для моделей, у которых рассуждения не отключаются")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_tokens:
        cfg["api"]["max_tokens"] = args.max_tokens
    ecfg = load_config(ROOT / cfg["emotion_config"])
    env = read_env()
    model = args.model or cfg["models"]["main"]
    emotions = ecfg["gold"]["emotions"]
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    kf = ROOT / ecfg["out_dir"] / "gold_key.parquet" if args.key == "gold" else out / f"{args.key}_key.parquet"
    cols = ["uid", "text"] + (["context"] if args.key == "sent" else [])
    key = pd.read_parquet(kf, columns=cols)                       # только uid, текст (и контекст для sent)
    if "context" not in key:
        key["context"] = None
    of = out / f"{args.key}__{model.replace('/', '_')}.csv"
    done = set(pd.read_csv(of)["uid"]) if of.exists() else set()
    todo = key[~key["uid"].isin(done)].sample(frac=1.0, random_state=cfg["seed"])   # порядок отправки случаен (прогноз по первым N)
    if args.limit:
        todo = todo.head(args.limit)
    spent0 = spent_rub(out)
    stop = cfg["stop_at_rub"]
    print(f"{model} | {args.key}: к разметке {len(todo)} (готово {len(done)}) | потрачено всего {spent0:.2f} ₽, стоп {stop} ₽")
    if spent0 >= stop:
        print("бюджет исчерпан")
        return
    lock = threading.Lock()
    state = {"spent": spent0, "n": 0, "err": 0, "stopped": False}
    t0 = time.time()

    def job(uid, text, context):
        if state["stopped"]:
            return uid, None
        return uid, call(env, cfg["api"], model, text, emotions, context if isinstance(context, str) and context else None)

    new = not of.exists()
    with of.open("a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(cfg["api"]["workers"]) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        futs = [ex.submit(job, r.uid, r.text, r.context) for r in todo.itertuples()]
        for fu in as_completed(futs):
            uid, d = fu.result()
            if d is None:
                continue
            if "error" in d and "choices" not in d:
                state["err"] += 1
                print("ошибка", uid, str(d["error"])[:200], flush=True)
                if state["err"] > 50:
                    state["stopped"] = True
                continue
            u = d.get("usage") or {}
            ch = (d.get("choices") or [{}])[0]
            raw = ((ch.get("message") or {}).get("content") or "").strip()
            tone, emo, skip = parse(raw, emotions)
            cost = float(u.get("cost") or 0)
            assert u.get("currency", "RUB") == "RUB", u
            with lock:
                state["spent"] += cost
                state["n"] += 1
                if state["spent"] >= stop:
                    state["stopped"] = True
            w.writerow(dict(uid=uid, tone="" if tone is None else tone, emotion=emo or "", skip=int(bool(skip)),
                            model=model, model_returned=d.get("model", ""), prompt_tokens=u.get("prompt_tokens"),
                            completion_tokens=u.get("completion_tokens"),
                            reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                            cost_rub=cost, finish_reason=ch.get("finish_reason"), raw=raw))
            if state["n"] % 200 == 0:
                f.flush()
                print(f"[{time.time() - t0:6.0f}s] {state['n']}/{len(todo)} | всего {state['spent']:.2f} ₽ | ошибок {state['err']}",
                      flush=True)
    print(f"готово: {state['n']} ответов, ошибок {state['err']}, потрачено всего {state['spent']:.2f} ₽"
          + (" — ОСТАНОВЛЕНО ПО БЮДЖЕТУ/ОШИБКАМ" if state["stopped"] else ""), "|", of)


if __name__ == "__main__":
    main()
