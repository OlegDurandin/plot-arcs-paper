"""Этап 2. Покрытие лексиконами по формам и корпусам + потери на лемматизации у Афанасьева.

Запуск:
    .venv/bin/python src/s02_coverage.py configs/s02_emotion.yaml
Пишет reports/02_coverage.md и data/processed/s02/coverage_*.parquet.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s02_lexicon import build_lexicons  # noqa: E402

LETTER = re.compile(r"[^\W\d_]")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (f"{v:,}".replace(",", " ") if isinstance(v, (int, np.integer))
                                                          else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def group_label(a: pd.DataFrame) -> pd.Series:
    lab = a["corpus"] + " / " + a["form"] + " (" + a["unit"] + ")"
    lab[a["base_magic600"].fillna(False).astype(bool)] = "afanasyev / волшебные, база (text)"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    out = ROOT / cfg["out_dir"]
    n = cfg["coverage"]["N"]
    lex = pd.read_parquet(out / "lex" / f"lex_N{n}.parquet")
    a = pd.read_parquet(ROOT / cfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(["text", "linked_work"])]
    a["group"] = group_label(a)
    # Афанасьев: основная база отдельно, все включённые — отдельно
    af_all = a[a["corpus"] == "afanasyev"].copy()
    af_all["group"] = "afanasyev / все типы ≥ 600 (text)"
    a.loc[(a["corpus"] == "afanasyev") & ~a["base_magic600"].fillna(False).astype(bool), "group"] = "afanasyev / не волшебные ≥ 600 (text)"
    a = pd.concat([a, af_all])
    d = lex.merge(a[["doc_id", "group"]], on="doc_id")
    d["rsl_polar"] = d["rsl_pos"] + d["rsl_neg"]
    rows = []
    for g, x in d.groupby("group"):
        w = x["n_words"].sum()
        r = dict(группа=g, текстов=x["doc_id"].nunique(), сегментов=len(x), слов_в_сегм_медиана=int(x["n_words"].median()))
        for lab, col in [("RSL_любые", "rsl_hits"), ("RSL_полярные", "rsl_polar"), ("Linis", "lin_hits"),
                         ("Linis_|≥0,5|", "lin_polar"), ("NRC", "nrc_hits")]:
            r[f"{lab}_доля_слов"] = float(x[col].sum() / w)
            r[f"{lab}_сегм_≥1"] = float((x[col] > 0).mean())
        rows.append(r)
    cov = pd.DataFrame(rows).sort_values("группа")
    cov.to_parquet(out / "coverage_N20.parquet", index=False)

    # ---- Афанасьев: потери на лемматизации (первый разбор pymorphy3 против всех разборов) + сравнение с рассказами
    import pymorphy3
    from razdel import tokenize
    morph = pymorphy3.MorphAnalyzer()
    L = build_lexicons(cfg)
    lexsets = {"RSL": {k for k, v in L["rsl"].items() if v != 0}, "Linis": {k for k, v in L["linis"].items() if abs(v) >= 0.5},
               "NRC": set(L["nrc"])}
    s01 = ROOT / cfg["s01_run"]
    base_ids = set(a.loc[a["group"] == "afanasyev / волшебные, база (text)", "doc_id"])
    rng = np.random.default_rng(cfg["seed"])
    st_ids = set(rng.choice(a.loc[a["group"] == "proza / story (text)", "doc_id"].to_numpy(), 300, replace=False))
    res, unknown = {}, Counter()
    unk_lemma = {}
    for label, ids in [("Афанасьев, база", base_ids), ("рассказы, 300 случайных", st_ids)]:
        toks = []
        for f in sorted(glob.glob(str(s01 / "sentences" / "*.parquet"))):
            s = pd.read_parquet(f, columns=["doc_id", "text"], filters=[("doc_id", "in", list(ids))])
            for t in s["text"]:
                toks += [x.text.lower().replace("ё", "е") for x in tokenize(t) if LETTER.search(x.text)]
        c = Counter(toks)
        tot = sum(c.values())
        stats = dict(токенов=tot, не_в_словаре_pymorphy=0)
        for k in lexsets:
            stats[f"{k}_по_1_разбору"] = 0
            stats[f"{k}_по_любому_разбору"] = 0
        for w, k_ in c.items():
            ps = morph.parse(w)
            known = morph.word_is_known(w)
            if not known:
                stats["не_в_словаре_pymorphy"] += k_
                if label.startswith("Афанасьев"):
                    unknown[w] += k_
                    unk_lemma[w] = ps[0].normal_form
            first = ps[0].normal_form.replace("ё", "е")
            alls = {p.normal_form.replace("ё", "е") for p in ps}
            for nm, sset in lexsets.items():
                if first in sset:
                    stats[f"{nm}_по_1_разбору"] += k_
                if alls & sset:
                    stats[f"{nm}_по_любому_разбору"] += k_
        res[label] = {k: (v / tot if k != "токенов" else v) for k, v in stats.items()}
    lem = pd.DataFrame(res).T.reset_index().rename(columns={"index": "выборка"})
    top_unk = pd.DataFrame([(w, unk_lemma[w], k) for w, k in unknown.most_common(40)], columns=["слово", "лемма pymorphy3", "частота"])
    top_unk.to_csv(out / "afanasyev_unknown_top40.csv", index=False)

    R = ["# Этап 2. Покрытие лексиконами\n",
         f"Сегменты N = {n}, документы из анализа (`include`, единицы text и linked_work). «доля_слов» — доля токенов сегмента, "
         "найденных в лексиконе (по сумме слов группы); «сегм_≥1» — доля сегментов хотя бы с одним найденным словом.\n",
         "Лексиконы (после приведения к леммам pymorphy3): " + json.dumps(L["info"], ensure_ascii=False) + "\n",
         "RSL_любые — все однословные записи RuSentiLex (вкл. нейтральные и контекстно-зависимые); RSL_полярные — только "
         "positive/negative; Linis_|≥0,5| — слова LinisCrowd со средней оценкой не меньше 0,5 по модулю; NRC — слово хотя бы "
         "с одной из 10 категорий.\n",
         "## Покрытие по группам\n", md_table(cov) + "\n",
         "## Афанасьев: потери на лемматизации\n",
         "Сравнение долей токенов, найденных в лексиконе (полярные записи), если брать лемму первого разбора pymorphy3 "
         "(как в корпусе) и если засчитывать любую из гипотез разбора. Разница — верхняя оценка потерь из-за выбора разбора. "
         "«не_в_словаре_pymorphy» — слова, которых нет в словаре (лемма угадана по окончанию; «нонче» → «нонкий»): "
         "их лемма обычно неверна, и в лексиконе такие слова почти никогда не находятся.\n",
         md_table(lem) + "\n",
         "## 40 самых частых слов Афанасьева, неизвестных словарю pymorphy3\n",
         md_table(top_unk) + "\n"]
    (ROOT / "reports" / "02_coverage.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R[5:]))


if __name__ == "__main__":
    main()
