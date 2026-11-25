"""Ворота 4, п. 4: светлее ли концовка фанфика, чем у рассказа той же длины, — или это артефакт?

(1) позиционный остаток сигнала на серебре (п. 2.1): не завышает ли сигнал концовки фанфиков (и рассказов);
(2) роль тегов Ficbook: end_lift фанфика против рассказа (MixedLM ~ форма + log n_words + (1 | автор)) — все фанфики,
    без «светлых» тегов, только «светлые», без «светлых» и юмора, «тёмные» (контроль валидности сигнала концовки);
(3) LLM-дуги (300 фанфиков и 300 рассказов в диапазоне длины сказок) — если размечены.

Запуск:  .venv/bin/python src/s04_fanfic_endings.py configs/s04_nulls.yaml
Выход:   reports/04_fanfic_endings.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s04_nulls import md  # noqa: E402
from s02_ensemble import boot_groups  # noqa: E402


def effect(x, y, label):
    """end_lift фанфика − рассказа с контролем длины; MixedLM, оптимизатор по умолчанию."""
    x = x[[y, "fk", "lw", "author_id"]].dropna().rename(columns={y: "v"})
    m = smf.mixedlm("v ~ C(fk, Treatment('story')) + lw", x, groups=x["author_id"]).fit(reml=True)
    nm = "C(fk, Treatment('story'))[T.fanfic]"
    lo, hi = m.conf_int().loc[nm]
    return f"{m.params[nm]:+.3f} [{lo:+.3f}; {hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    tg = cfg["fanfic_tags"]
    out = ROOT / cfg["out_dir"] / "full"
    rng = np.random.default_rng(cfg["seed"])
    R = ["# Концовки фанфиков (Ворота 4, п. 4)\n"]

    # (1) позиционный остаток на серебре
    from s02_position_report import load as load_silver
    s = load_silver(load_config(ROOT / "configs/s02_ensemble.yaml"))
    R.append("## 1. Не завышает ли сигнал концовки фанфиков (серебро, позиционный остаток п. 2.1)\n")
    R.append("Остаток = LLM − калибровка сигнала внутри формы; «конец − середина» = последние 2 сегмента минус 3…18. "
             "Положительный — сигнал **занижает** концовку относительно LLM. 95 % ДИ — бутстрэп по авторам.\n")
    rows = []
    for sig in ["Классификатор_тональности", "RSL"]:
        r = {"сигнал": sig}
        vals = {}
        for st in ["fanfic", "story"]:
            x = s[s["stratum"] == st]
            pos, res, g = x["pos"].to_numpy(), x[f"res_{sig}"].to_numpy(), x["author_id"].astype(str).to_numpy()

            def c(t):
                p_, r_ = pos[t], res[t]
                return r_[p_ >= 19].mean() - r_[(p_ >= 3) & (p_ <= 18)].mean()
            bs = [c(t) for t in boot_groups(g, rng, 1000)]
            vals[st] = c(np.arange(len(res)))
            r[st] = f"{vals[st]:+.2f} [{np.percentile(bs, 2.5):+.2f}; {np.percentile(bs, 97.5):+.2f}]"
        r["фанфик − рассказ"] = f"{vals['fanfic'] - vals['story']:+.2f}"
        rows.append(r)
    R.append(md(pd.DataFrame(rows).rename(columns={"fanfic": "фанфик", "story": "рассказ"})) + "\n")

    # (2) теги
    a = pd.read_parquet(ROOT / cfg["analysis_set"], columns=["doc_id", "tags"])
    R.append("## 2. Роль тегов (полный корпус, N = 20, within_author_cap)\n")
    R.append("end_lift (z-дуга: последние 2 сегмента − середина), эффект фанфика относительно рассказа, MixedLM ~ форма + "
             "log n_words + (1 | автор), 95 % ДИ.\n")
    for N in [20, 10]:
        d = pd.read_parquet(out / f"arcs_N{N}.parquet").drop(columns=cfg["signals"]).merge(pd.read_parquet(out / f"metrics_N{N}.parquet"), on="doc_id")
        d = d[d["within_author_cap"] & d["fk"].isin(["fanfic", "story"])].merge(a, on="doc_id", how="left")
        d["lw"] = np.log(d["n_words"])
        d["author_id"] = d["author_id"].astype(str)
        tags = d["tags"].map(lambda t: set(json.loads(t)) if isinstance(t, str) else set())
        d["bright"] = tags.map(lambda t: bool(t & set(tg["bright"])))
        d["bright_plus"] = tags.map(lambda t: bool(t & set(tg["bright"] + tg["bright_plus"])))
        d["dark"] = tags.map(lambda t: bool(t & set(tg["dark"])))
        fan = d["fk"] == "fanfic"
        if N == 20:
            R.append(f"Фанфиков: {int(fan.sum())}; со «светлыми» тегами {d.loc[fan, 'bright'].mean():.1%}, "
                     f"со «светлыми» или юмором {d.loc[fan, 'bright_plus'].mean():.1%}, с «тёмными» {d.loc[fan, 'dark'].mean():.1%}, "
                     f"с обоими (светлые и тёмные) {(d.loc[fan, 'bright'] & d.loc[fan, 'dark']).mean():.1%}.\n")
        groups = [("все фанфики", fan), ("без «светлых» тегов", fan & ~d["bright"]), ("без «светлых» и юмора", fan & ~d["bright_plus"]),
                  ("только «светлые»", fan & d["bright"] & ~d["dark"]), ("только «тёмные»", fan & d["dark"] & ~d["bright"]),
                  ("ни светлых, ни тёмных", fan & ~d["bright_plus"] & ~d["dark"])]
        rows = []
        for nm, msk in groups:
            x = d[msk | (d["fk"] == "story")]
            r = {"N": N, "фанфики": nm, "n фанфиков": int(msk.sum())}
            for sig in cfg["signals"]:
                r[f"{sig}: end_lift, эффект"] = effect(x, f"{sig}_end_lift", nm)
                r[f"{sig}: медиана end_lift фанфиков"] = f"{d.loc[msk, f'{sig}_end_lift'].median():+.3f}"
            rows.append(r)
        R.append(md(pd.DataFrame(rows)) + "\n")
        if N == 20:
            R.append(f"Медиана end_lift рассказов (N = 20): clf {d.loc[~fan, 'clf_end_lift'].median():+.3f}, "
                     f"rsl {d.loc[~fan, 'rsl_end_lift'].median():+.3f}.\n")

    # (3) LLM-дуги
    acfg = load_config(ROOT / "configs/s02_llm_arcs.yaml")
    ak = ROOT / acfg["out_dir"] / "arcs_key.parquet"
    af = ROOT / acfg["out_dir"] / f"arcs__{acfg['models']['main']}.csv"
    R.append("## 3. LLM-дуги\n")
    if af.exists():
        import s02_validate as V
        k = pd.read_parquet(ak, columns=["uid", "doc_id", "stratum", "author_id", "seg", "doc_n_words"])
        an = V.read_ann(af)
        x = k.merge(an, on="uid")
        x = x[x["skip"] == 0]
        full = x.groupby("doc_id")["seg"].transform("size") == 20
        x = x[full & x["stratum"].isin(["fanfic", "story"])]
        rows = []
        for doc, t in x.groupby("doc_id"):
            t = t.sort_values("seg")
            v = t["tone"].to_numpy(float)
            sd = v.std()
            z = (v - v.mean()) / sd if sd > 0 else v * 0
            rows.append(dict(doc_id=doc, fk=t["stratum"].iloc[0], author_id=str(t["author_id"].iloc[0]),
                             lw=np.log(t["doc_n_words"].iloc[0]), end_raw=v[18:].mean() - v[2:18].mean(),
                             end_z=z[18:].mean() - z[2:18].mean()))
        L = pd.DataFrame(rows).merge(a, on="doc_id", how="left")
        if len(L) and L["fk"].nunique() == 2:
            tags = L["tags"].map(lambda t: set(json.loads(t)) if isinstance(t, str) else set())
            L["bright"] = tags.map(lambda t: bool(t & set(tg["bright"])))
            R.append(f"Полностью размечено: фанфиков {int((L['fk'] == 'fanfic').sum())}, рассказов {int((L['fk'] == 'story').sum())} "
                     "(длина 643–3 179 слов, ≤ 1 текста на автора). Эффект фанфика — OLS ~ форма + log n_words (авторы не повторяются).\n")
            rows = []
            for nm, msk in [("все фанфики", L["fk"] == "fanfic"), ("без «светлых» тегов", (L["fk"] == "fanfic") & ~L["bright"])]:
                y = L[msk | (L["fk"] == "story")]
                r = {"фанфики": nm, "n": int(msk.sum())}
                for c in ["end_raw", "end_z"]:
                    m = smf.ols(f"{c} ~ C(fk, Treatment('story')) + lw", y).fit(cov_type="HC1")
                    nmc = "C(fk, Treatment('story'))[T.fanfic]"
                    lo, hi = m.conf_int().loc[nmc]
                    r[{"end_raw": "LLM: конец − середина, баллы", "end_z": "LLM: end_lift (z)"}[c]] = f"{m.params[nmc]:+.3f} [{lo:+.3f}; {hi:+.3f}]"
                rows.append(r)
            R.append(md(pd.DataFrame(rows)) + "\n")
        else:
            R.append("Недостаточно полностью размеченных текстов.\n")
    else:
        R.append("LLM-дуги ещё не размечены.\n")
    (ROOT / "reports" / "04_fanfic_endings.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
