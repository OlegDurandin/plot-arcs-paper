"""Этап 2. Вторая разметка золотой выборки локальной LLM (MLX) — только для оценки согласия.

Та же инструкция, что у разметчика (шкала −2…+2, 7 эмоций, «пропустить»), жадная генерация (температура 0).
Из ключа читаются ТОЛЬКО uid и текст (форма и корпус модели недоступны). Повторы (repeat_of) не размечаются:
при температуре 0 ответ детерминирован.

Запуск:
    .venv/bin/python src/s02_llm_annotate.py configs/s02_emotion.yaml [--limit 5]
Выход: data/processed/s02/gold_llm.csv (uid, tone, emotion, skip, raw) — дописывается, повторный запуск продолжает.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

INSTRUCTION = """Ты — разметчик художественных текстов. Прочитай фрагмент и оцени его так же, как человек-разметчик.

1. Тональность фрагмента в целом по шкале: −2 (очень негативная), −1, 0 (нейтральная), +1, +2 (очень позитивная).
2. Доминирующая эмоция — ровно одна из списка: {emotions}.
3. Если фрагмент невозможно оценить (непонятен, не является связным текстом) — поставь "skip": true.

Ответь только JSON без пояснений, в виде:
{{"tone": <целое от -2 до 2>, "emotion": "<эмоция из списка>", "skip": <true|false>}}

Фрагмент:
\"\"\"
{text}
\"\"\""""


def parse(raw: str, emotions: list[str]):
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None, None, None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None, None
    tone = d.get("tone")
    tone = int(tone) if isinstance(tone, (int, float)) and -2 <= tone <= 2 else None
    emo = d.get("emotion")
    emo = emo if emo in emotions else None
    return tone, emo, bool(d.get("skip", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    out = ROOT / cfg["out_dir"] / "gold_llm.csv"
    key = pd.read_parquet(ROOT / cfg["out_dir"] / "gold_key.parquet", columns=["uid", "text", "repeat_of"])
    key = key[key["repeat_of"].isna()][["uid", "text"]]          # только uid и текст
    done = set(pd.read_csv(out)["uid"]) if out.exists() else set()
    todo = key[~key["uid"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    emotions = cfg["gold"]["emotions"]
    model, tok = load(args.model)
    sampler = make_sampler(temp=0.0)
    new = not out.exists()
    t0 = time.time()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["uid", "tone", "emotion", "skip", "model", "raw"])
        for i, r in enumerate(todo.itertuples()):
            msgs = [{"role": "user", "content": INSTRUCTION.format(emotions=", ".join(emotions), text=r.text)}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            raw = generate(model, tok, prompt=prompt, max_tokens=60, sampler=sampler, verbose=False)
            tone, emo, skip = parse(raw, emotions)
            w.writerow([r.uid, "" if tone is None else tone, emo or "", int(bool(skip)), args.model, raw.strip()])
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(todo)}", flush=True)
    print("готово:", out)


if __name__ == "__main__":
    main()
