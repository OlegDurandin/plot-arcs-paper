"""Ворота 2/5, п. 2: устойчивость результата «начало волшебной сказки темнее прозы той же длины» (LLM-дуги).

Начало как контраст z-дуги внутри текста (сигнал → z по 20 сегментам):
  (а) основной: сегменты 1–2 минус середина 3–18;
  (б) без зачина: сегменты 2–3 минус середина 4–18 (эффект не держится только на первом сегменте);
сигналы: LLM, классификатор, RSL (z-дуга не зависит от линейной калибровки). Разница «проза − Афанасьев» и
«проза − fairytalez» (бутстрэп по авторам, независимо в группах). Проза — рассказы, новеллы, фанфики в диапазоне длины
сказок базы. И 15 начальных сегментов сказок базы с самой низкой LLM-оценкой (для чтения: недостача / вредительство).

Запуск:  .venv/bin/python src/s02_beginnings.py configs/s02_ensemble.yaml
Выход:   reports/02_tale_beginnings.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
import s02_validate as V  # noqa: E402
from s02_ensemble import boot_groups  # noqa: E402

PROSE = ["story", "novella", "fanfic"]
RU = {"afanasyev": "Афанасьев", "fairytales": "fairytalez", "story": "рассказ", "novella": "новелла", "fanfic": "фанфик"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["bootstrap"]
    acfg = load_config(ROOT / "configs/s02_llm_arcs.yaml")
    ao = ROOT / acfg["out_dir"]
    k = pd.read_parquet(ao / "arcs_key.parquet")
    an = V.read_ann(ao / f"arcs__{acfg['models']['main']}.csv")
    x = k.merge(an, on="uid")
    x = x[x.groupby("doc_id")["skip"].transform("sum") == 0]
    x = x[x["stratum"].isin(["afanasyev", "fairytales"] + PROSE)]
    ids = list(set(x["doc_id"]))
    c = pd.read_parquet(ROOT / "data/processed/s02/clf/clf_N20.parquet", columns=["doc_id", "seg", "sent_positive", "sent_negative"],
                        filters=[("doc_id", "in", ids)])
    l_ = pd.read_parquet(ROOT / "data/processed/s02/lex/lex_N20.parquet", columns=["doc_id", "seg", "n_words", "rsl_pos", "rsl_neg"],
                         filters=[("doc_id", "in", ids)])
    x = x.merge(c, on=["doc_id", "seg"]).merge(l_, on=["doc_id", "seg"])
    x["clf"] = x["sent_positive"] - x["sent_negative"]
    x["rsl"] = (x["rsl_pos"] - x["rsl_neg"]) / x["n_words"].clip(lower=1)
    defs = {"(а) сегменты 1–2 − середина 3–18": ([0, 1], list(range(2, 18))),
            "(б) без зачина: сегменты 2–3 − середина 4–18": ([1, 2], list(range(3, 18)))}
    per = {}
    for sig in ["tone", "clf", "rsl"]:
        for dn, (bseg, mseg) in defs.items():
            rows = []
            for doc, t in x.groupby("doc_id"):
                t = t.sort_values("seg")
                v = t[sig].to_numpy(float)
                sd = v.std()
                z = (v - v.mean()) / sd if sd > 0 else v * 0
                rows.append((t["stratum"].iloc[0], str(t["author_id"].iloc[0]), z[bseg].mean() - z[mseg].mean()))
            per[(sig, dn)] = pd.DataFrame(rows, columns=["st", "au", "c"])
    R = ["# Начало волшебной сказки темнее прозы? Проверки устойчивости (LLM-дуги, N = 20)\n",
         "Контраст начала — z-дуга внутри текста. Проза (рассказ, новелла, фанфик) — в диапазоне длины сказок базы "
         "(643–3 179 слов), ≤ 1 текста на автора. Разница групп — 95 % ДИ, бутстрэп по авторам независимо в каждой группе. "
         "Положительная разница «проза − сказка» = сказка начинается темнее прозы (относительно своей середины).\n"]
    sig_ru = {"tone": "LLM", "clf": "классификатор", "rsl": "RSL"}
    for dn in defs:
        R.append(f"## {dn}\n")
        rows = []
        for sig in ["tone", "clf", "rsl"]:
            p = per[(sig, dn)]
            r = {"сигнал": sig_ru[sig]}
            for st in ["afanasyev", "fairytales"] + PROSE:
                r[RU[st]] = f"{p.loc[p['st'] == st, 'c'].mean():+.2f}"
            for ref in ["afanasyev", "fairytales"]:
                b = p[p["st"] == ref]
                bb = np.array([b["c"].to_numpy()[t].mean() for t in boot_groups(b["au"].to_numpy(), rng, B)])
                for st in PROSE:
                    q = p[p["st"] == st]
                    qb = np.array([q["c"].to_numpy()[t].mean() for t in boot_groups(q["au"].to_numpy(), rng, B)])
                    dd = qb - bb
                    lo, hi = np.percentile(dd, [2.5, 97.5])
                    flag = " ✓" if lo > 0 else (" ✗" if hi < 0 else "")
                    r[f"{RU[st]} − {RU[ref]}"] = f"{q['c'].mean() - b['c'].mean():+.2f} [{lo:+.2f}; {hi:+.2f}]{flag}"
            rows.append(r)
        R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    R.append("✓ — ДИ выше 0 (сказка темнее в начале), ✗ — ниже 0.\n")
    # самые тёмные начала сказок
    a = x[(x["stratum"] == "afanasyev") & (x["seg"] <= 1)].sort_values(["tone", "clf"]).head(15)
    R.append("## 15 начальных сегментов (1–2) сказок базы с самой низкой LLM-оценкой\n")
    R.append("| # | сегмент | слов | LLM | эмоция LLM | текст |\n|---|---|---|---|---|---|")
    emo = an.set_index("uid")["emotion"]
    for i, r in enumerate(a.itertuples(), 1):
        t = r.text.replace("|", "／").replace("\n", " ")
        t = t if len(t) <= 600 else t[:600] + " …"
        R.append(f"| {i} | {r.seg + 1} | {r.shown_n_words} | {r.tone:+.0f} | {emo.get(r.uid, '')} | {t} |")
    R.append("")
    (ROOT / "reports" / "02_tale_beginnings.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
