"""Этап 2. «Серебряная» выборка сегментов N = 20 для LLM-разметки через API.

Те же поля и то же правило отрывков, что у золота (s02_gold.py), поэтому сигналы считаются тем же gold_signals().
Страты и размеры — configs/s02_llm_api.yaml (silver). Пары (doc_id, seg) из золота исключаются.

Запуск:
    .venv/bin/python src/s02_silver.py configs/s02_llm_api.yaml
Выход: data/processed/s02/llm_api/silver_key.parquet (uid вида v00001) — если файл есть, не пересоздаётся.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    ecfg = load_config(ROOT / cfg["emotion_config"])
    sc = cfg["silver"]
    out = ROOT / cfg["out_dir"]
    kf = out / "silver_key.parquet"
    if kf.exists():
        print("уже есть:", kf)
        return
    rng = np.random.default_rng(cfg["seed"])
    s01 = ROOT / ecfg["s01_run"]
    a = pd.read_parquet(ROOT / ecfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(sc["units"]) & a["within_author_cap"]]
    seg = pd.read_parquet(s01 / f"segments_rel_N{sc['N']}.parquet", columns=["doc_id", "seg", "sent_start", "sent_end", "n_words"])
    a = a[a["doc_id"].isin(set(seg["doc_id"]))]
    gold_pairs = set()
    if sc["exclude_gold_segments"]:
        g = pd.read_parquet(ROOT / ecfg["out_dir"] / "gold_key.parquet", columns=["doc_id", "seg"])
        gold_pairs = set(zip(g["doc_id"], g["seg"]))

    picks = []
    for name, st in sc["strata"].items():
        pool = a[a["corpus"] == st["corpus"]]
        if "form" in st:
            pool = pool[pool["form"] == st["form"]]
        if "base" in st:
            pool = pool[pool[st["base"]]]
        pool = pool.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        if st.get("one_per_author", True):
            pool = pool.drop_duplicates("author_id")
        n = st["n"]
        k = int(min(sc["max_seg_per_text"], max(1, np.ceil(n / len(pool)))))
        picks.append(pool.head(int(np.ceil(n / k)))[["doc_id", "corpus", "form", "unit", "author_id"]]
                     .assign(stratum=name, k=k, target=n))
    docs = pd.concat(picks, ignore_index=True)
    segd = dict(tuple(seg[seg["doc_id"].isin(set(docs["doc_id"]))].groupby("doc_id")))
    rows = []
    for r in docs.itertuples():
        sg = segd[r.doc_id]
        sg = sg[[(r.doc_id, s) not in gold_pairs for s in sg["seg"]]]
        if not len(sg):
            continue
        for _, s in sg.iloc[rng.choice(len(sg), min(r.k, len(sg)), replace=False)].iterrows():
            rows.append(dict(doc_id=r.doc_id, corpus=r.corpus, form=r.form, unit=r.unit, stratum=r.stratum,
                             author_id=r.author_id, seg=int(s["seg"]), seg_sent_start=int(s["sent_start"]),
                             seg_sent_end=int(s["sent_end"]), seg_n_words=int(s["n_words"]), target=r.target))
    key = pd.DataFrame(rows)
    # обрезка до целевого размера страты (случайно)
    key = pd.concat([g.head(int(g["target"].iloc[0])) for _, g in
                     key.sample(frac=1.0, random_state=cfg["seed"]).groupby("stratum")])
    key = key.drop(columns="target").reset_index(drop=True)
    assert key["stratum"].notna().all() and key["author_id"].notna().all()

    want = list(set(key["doc_id"]))
    sents = []
    for f in sorted(glob.glob(str(s01 / "sentences" / "*.parquet"))):
        x = pd.read_parquet(f, columns=["doc_id", "sent_idx", "text", "n_words"], filters=[("doc_id", "in", want)])
        if len(x):
            sents.append(x)
    sd = {k: g.set_index("sent_idx").sort_index() for k, g in pd.concat(sents).groupby("doc_id")}
    gc = ecfg["gold"]
    lim, ew = gc["excerpt_if_longer"], gc["excerpt_words"]
    shown_a, shown_z, texts, nw = [], [], [], []
    for r in key.itertuples():
        g = sd[r.doc_id].loc[r.seg_sent_start:r.seg_sent_end - 1]
        if lim and r.seg_n_words > lim:   # то же правило, что в s02_gold.py
            cum = g["n_words"].cumsum().to_numpy()
            start_w = int(rng.integers(0, max(1, cum[-1] - ew)))
            i0 = int(np.searchsorted(cum, start_w, side="right"))
            i1 = int(np.searchsorted(cum, cum[i0] - g["n_words"].iloc[i0] + ew, side="left")) + 1
            i1 = min(max(i1, i0 + 1), len(g))
            g = g.iloc[i0:i1]
        shown_a.append(int(g.index[0]))
        shown_z.append(int(g.index[-1]) + 1)
        texts.append(" ".join(g["text"]))
        nw.append(int(g["n_words"].sum()))
    key["shown_sent_start"], key["shown_sent_end"], key["shown_n_words"], key["text"] = shown_a, shown_z, nw, texts
    key["is_excerpt"] = (key["shown_sent_start"] != key["seg_sent_start"]) | (key["shown_sent_end"] != key["seg_sent_end"])
    key = key.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    key.insert(0, "uid", [f"v{i + 1:05d}" for i in range(len(key))])
    out.mkdir(parents=True, exist_ok=True)
    key.to_parquet(kf, index=False)
    print(key.groupby("stratum").agg(сегментов=("uid", "size"), текстов=("doc_id", "nunique"), авторов=("author_id", "nunique"),
                                     отрывков=("is_excerpt", "sum"), слов=("shown_n_words", "sum"),
                                     медиана_слов=("shown_n_words", "median")).to_string())
    print("всего слов:", int(key["shown_n_words"].sum()))


if __name__ == "__main__":
    main()
