"""Ворота 2, п. 2.2: выборка для LLM-дуг тональности (все 20 сегментов N = 20 каждого текста) и оценка стоимости.

Группы и размеры — configs/s02_llm_arcs.yaml. Рассказы, новеллы, фанфики — в диапазоне длины сказок базы
(5–95-й перцентиль n_words), ≤ 1 текста на автора; повести и романы — каждый сегмент отрывком ≈ 350 слов.
Разметка — тем же s02_llm_api.py (--key arcs), та же инструкция (тональность + эмоция).

Запуск:
    .venv/bin/python src/s02_llm_arcs.py configs/s02_llm_arcs.yaml          # выборка + оценка, без запросов к API
    .venv/bin/python src/s02_llm_api.py  configs/s02_llm_arcs.yaml --key arcs
Выход: data/processed/s02/llm_arcs/arcs_key.parquet
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
    ac = cfg["arcs"]
    out = ROOT / cfg["out_dir"]
    kf = out / "arcs_key.parquet"
    rng = np.random.default_rng(cfg["seed"])
    if kf.exists():
        key = pd.read_parquet(kf)
        print("уже есть:", kf)
    else:
        s01 = ROOT / ecfg["s01_run"]
        a = pd.read_parquet(ROOT / ecfg["analysis_set"])
        a = a[a["include"] & a["unit"].isin(ac["units"]) & a["within_author_cap"]]
        seg = pd.read_parquet(s01 / f"segments_rel_N{ac['N']}.parquet", columns=["doc_id", "seg", "sent_start", "sent_end", "n_words"])
        full = seg.groupby("doc_id")["seg"].size()
        a = a[a["doc_id"].isin(set(full[full == ac["N"]].index))]
        base = a[(a["corpus"] == "afanasyev") & a["base_magic600"]]
        lo, hi = np.percentile(base["n_words"], ac["length_pct"])
        print(f"длина сказок базы, {ac['length_pct'][0]}–{ac['length_pct'][1]}-й перцентиль: {lo:.0f}–{hi:.0f} слов")
        picks = []
        for name, gcf in ac["groups"].items():
            pool = a[a["corpus"] == gcf["corpus"]]
            if "form" in gcf:
                pool = pool[pool["form"] == gcf["form"]]
            if "base" in gcf:
                pool = pool[pool[gcf["base"]]]
            if gcf.get("tale_length"):
                pool = pool[(pool["n_words"] >= lo) & (pool["n_words"] <= hi)]
            pool = pool.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
            if gcf["one_per_author"]:
                pool = pool.drop_duplicates("author_id")
            if gcf["n"] != "all":
                pool = pool.head(int(gcf["n"]))
            picks.append(pool[["doc_id", "corpus", "form", "unit", "author_id", "n_words"]].assign(stratum=name, excerpt=gcf["excerpt"]))
        docs = pd.concat(picks, ignore_index=True)
        sg = seg[seg["doc_id"].isin(set(docs["doc_id"]))].merge(docs, on="doc_id", suffixes=("", "_doc"))
        want = list(set(docs["doc_id"]))
        sents = []
        for f in sorted(glob.glob(str(s01 / "sentences" / "*.parquet"))):
            x = pd.read_parquet(f, columns=["doc_id", "sent_idx", "text", "n_words"], filters=[("doc_id", "in", want)])
            if len(x):
                sents.append(x)
        sd = {k: g.set_index("sent_idx").sort_index() for k, g in pd.concat(sents).groupby("doc_id")}
        ew = ac["excerpt_words"]
        rows = []
        for r in sg.sort_values(["stratum", "doc_id", "seg"]).itertuples():
            g = sd[r.doc_id].loc[r.sent_start:r.sent_end - 1]
            if r.excerpt and r.n_words > ew:      # то же правило отрывка, что в s02_gold.py / s02_silver.py
                cum = g["n_words"].cumsum().to_numpy()
                start_w = int(rng.integers(0, max(1, cum[-1] - ew)))
                i0 = int(np.searchsorted(cum, start_w, side="right"))
                i1 = int(np.searchsorted(cum, cum[i0] - g["n_words"].iloc[i0] + ew, side="left")) + 1
                i1 = min(max(i1, i0 + 1), len(g))
                g = g.iloc[i0:i1]
            rows.append(dict(doc_id=r.doc_id, corpus=r.corpus, form=r.form, unit=r.unit, stratum=r.stratum, author_id=r.author_id,
                             doc_n_words=r.n_words_doc, seg=int(r.seg), seg_sent_start=int(r.sent_start), seg_sent_end=int(r.sent_end),
                             seg_n_words=int(r.n_words), shown_sent_start=int(g.index[0]), shown_sent_end=int(g.index[-1]) + 1,
                             shown_n_words=int(g["n_words"].sum()), text=" ".join(g["text"])))
        key = pd.DataFrame(rows)
        key["is_excerpt"] = (key["shown_sent_start"] != key["seg_sent_start"]) | (key["shown_sent_end"] != key["seg_sent_end"])
        assert key.groupby("doc_id")["seg"].nunique().eq(ac["N"]).all()
        key.insert(0, "uid", [f"a{i + 1:06d}" for i in range(len(key))])
        out.mkdir(parents=True, exist_ok=True)
        key.to_parquet(kf, index=False)
    cm = ac["cost_model"]
    key["est_rub"] = cm["a"] + cm["b"] * key["shown_n_words"]
    summ = key.groupby("stratum").agg(текстов=("doc_id", "nunique"), авторов=("author_id", "nunique"), сегментов=("uid", "size"),
                                      отрывков=("is_excerpt", "sum"), слов=("shown_n_words", "sum"),
                                      медиана_длины_текста=("doc_n_words", "median"), оценка_руб=("est_rub", "sum"))
    print(summ.round(1).to_string())
    print(f"ИТОГО: {len(key)} запросов, {int(key['shown_n_words'].sum())} слов, оценка ≈ {key['est_rub'].sum():.0f} ₽ "
          f"(стоп {cfg['stop_at_rub']} ₽)")


if __name__ == "__main__":
    main()
