"""Этап 2. Классификаторы тональности и эмоций на сегментах N = 10 и N = 20.

Документ режется на куски по предложениям: кусок не пересекает границы сегментов N = 10 и N = 20 и не длиннее
max_tokens (510 токенов + служебные). Оценка сегмента = среднее вероятностей по его кускам, взвешенное по числу слов
(«скользящее окно по 512 токенов с усреднением», PLAN.md). Обе модели — в одном проходе, на MPS.

Запуск:
    .venv/bin/python src/s02_classify.py configs/s02_emotion.yaml            # выборочный прогон этапа 1
    .venv/bin/python src/s02_classify.py configs/s02_emotion.yaml --full     # полный
Выход: <out_dir>/clf[_sample]/N{10,20}/shard-*.parquet, затем clf_N{10,20}.parquet:
    doc_id, seg, n_words, sent_neutral, sent_positive, sent_negative, emo_no_emotion, emo_joy, emo_sadness,
    emo_surprise, emo_fear, emo_anger, n_chunks
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

TOKEN_BUDGET = 64 * 512        # токенов в пакете (с дополнением)


def make_chunks(sent: pd.DataFrame, seg_starts: dict[str, set[int]], tok_len: np.ndarray, max_tok: int):
    """Куски: (doc_id, sent_start, sent_end, n_words, text). Новая граница — на старте любого сегмента N10/N20
    или при превышении max_tok."""
    out = []
    docs = sent["doc_id"].to_numpy()
    sidx = sent["sent_idx"].to_numpy()
    texts = sent["text"].to_numpy()
    nw = sent["n_words"].to_numpy()
    i, n = 0, len(sent)
    while i < n:
        doc = docs[i]
        starts = seg_starts.get(doc, set())
        j = i
        cur_tok, cur_w, buf = 0, 0, []
        a = sidx[i]
        while j < n and docs[j] == doc:
            if buf and (sidx[j] in starts or cur_tok + tok_len[j] > max_tok):
                out.append((doc, a, sidx[j], cur_w, " ".join(buf)))
                buf, cur_tok, cur_w, a = [], 0, 0, sidx[j]
            buf.append(texts[j])
            cur_tok += tok_len[j]
            cur_w += nw[j]
            j += 1
        if buf:
            out.append((doc, a, sidx[j - 1] + 1, cur_w, " ".join(buf)))
        i = j
    return pd.DataFrame(out, columns=["doc_id", "sent_start", "sent_end", "n_words", "text"])


@torch.no_grad()
def run_models(texts: list[str], tok, models: dict, device: str, max_len: int) -> dict[str, np.ndarray]:
    """Пакеты по длине: длина дополняется до кратной 64, размер пакета — степень двойки в пределах TOKEN_BUDGET
    (меньше разных форм тензоров — MPS не перекомпилирует ядра; +15 % скорости, результаты совпадают до 1e-5)."""
    enc = tok(texts, truncation=True, max_length=max_len, add_special_tokens=True)["input_ids"]
    lens = np.array([len(e) for e in enc])
    order = np.argsort(lens)
    res = {k: np.zeros((len(texts), m.config.num_labels), np.float32) for k, m in models.items()}
    i = 0
    while i < len(order):
        L = min(max_len, int(np.ceil(lens[order[i]] / 64) * 64))
        bs = max(1, 2 ** int(np.log2(max(1, TOKEN_BUDGET // max(L, 64)))))
        j = i
        while j < len(order) and j - i < bs and lens[order[j]] <= L:
            j += 1
        idx = order[i:j]
        batch = tok.pad({"input_ids": [enc[k] for k in idx]}, padding="max_length", max_length=L,
                        return_tensors="pt").to(device)
        for k, m in models.items():
            logits = m(**batch).logits.float()
            p = torch.sigmoid(logits) if m.config.problem_type == "multi_label_classification" else torch.softmax(logits, -1)
            res[k][idx] = p.cpu().numpy()
        i = j
    return res


def aggregate(ch: pd.DataFrame, seg: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Кусок целиком лежит в одном сегменте: сегмент по sent_start куска."""
    rows = []
    segg = dict(tuple(seg.groupby("doc_id")))
    for doc, g in ch.groupby("doc_id", sort=False):
        if doc not in segg:
            continue
        sg = segg[doc].sort_values("sent_start")
        k = np.searchsorted(sg["sent_start"].to_numpy(), g["sent_start"].to_numpy(), side="right") - 1
        w = g["n_words"].to_numpy().astype(float)
        df = pd.DataFrame(g[cols].to_numpy() * w[:, None], columns=cols)
        df["seg"] = sg["seg"].to_numpy()[k]
        df["n_words"] = w
        df["n_chunks"] = 1
        agg = df.groupby("seg").sum()
        agg[cols] = agg[cols].div(agg["n_words"].clip(lower=1), axis=0)
        agg.insert(0, "doc_id", doc)
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s01 = ROOT / cfg["s01_run"]
    if not args.full:
        cands = sorted(glob.glob(str(ROOT / "data/processed/s01_*_sample")), key=lambda p: Path(p).stat().st_mtime)
        s01 = Path(cands[-1])
    out = ROOT / cfg["out_dir"] / ("clf" if args.full else "clf_sample")
    mc = cfg["models"]
    device = mc["device"] if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mc["sentiment"]["hf"])
    tok2 = AutoTokenizer.from_pretrained(mc["emotion"]["hf"])
    assert tok.get_vocab() == tok2.get_vocab(), "у моделей разные токенизаторы — нужен раздельный проход"
    models = {"sent": AutoModelForSequenceClassification.from_pretrained(mc["sentiment"]["hf"]).to(device).eval(),
              "emo": AutoModelForSequenceClassification.from_pretrained(mc["emotion"]["hf"]).to(device).eval()}
    labels = {k: [m.config.id2label[i] for i in range(m.config.num_labels)] for k, m in models.items()}
    cols = [f"sent_{x}" for x in labels["sent"]] + [f"emo_{x}" for x in labels["emo"]]
    ns = cfg["segments"]
    segs = {n: pd.read_parquet(s01 / f"segments_rel_N{n}.parquet") for n in ns}
    starts_all: dict[str, set[int]] = {}
    for n in ns:
        for doc, st in zip(segs[n]["doc_id"], segs[n]["sent_start"]):
            starts_all.setdefault(doc, set()).add(int(st))
    shards = sorted(glob.glob(str(s01 / "sentences" / "*.parquet")))
    t0 = time.time()
    tot_tok = 0
    for si, path in enumerate(shards):
        name = Path(path).stem
        if all((out / f"N{n}" / f"{name}.parquet").exists() for n in ns):
            continue
        sent = pd.read_parquet(path, columns=["doc_id", "sent_idx", "text", "n_words"]).sort_values(["doc_id", "sent_idx"])
        tl = np.array([len(x) for x in tok(list(sent["text"]), add_special_tokens=False)["input_ids"]])
        ch = make_chunks(sent, starts_all, tl, mc["max_tokens"] - 2)
        res = run_models(list(ch["text"]), tok, models, device, mc["max_tokens"])
        tot_tok += int(tl.sum())
        for k, lab in labels.items():
            for j, lb in enumerate(lab):
                ch[f"{k}_{lb}"] = res[k][:, j]
        docs = set(ch["doc_id"])
        for n in ns:
            sg = segs[n][segs[n]["doc_id"].isin(docs)]
            agg = aggregate(ch, sg, cols)
            (out / f"N{n}").mkdir(parents=True, exist_ok=True)
            agg.to_parquet(out / f"N{n}" / f"{name}.parquet", index=False)
        el = time.time() - t0
        print(f"[{el:7.0f}s] {si + 1}/{len(shards)} {name}: кусков {len(ch)}, токенов всего {tot_tok:,}, "
              f"{tot_tok / max(el, 1):,.0f} ток/с", flush=True)
    for n in ns:
        parts = sorted((out / f"N{n}").glob("shard-*.parquet"))
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(out / f"clf_N{n}.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(dict(s01=str(s01.relative_to(ROOT)), models={k: mc[k]["hf"] for k in ["sentiment", "emotion"]},
                                                   labels=labels, max_tokens=mc["max_tokens"]), ensure_ascii=False, indent=2))
    print("готово:", out)


if __name__ == "__main__":
    main()
