"""Ворота 2, п. 2.2: LLM-дуги против классификатора / RSL (с поправкой на позицию и без) и эмоции LLM по позициям.

1. Поправка на позицию (серебро): калибровка сигнала внутри формы tone ~ a + b·сигнал, затем средний остаток по
   (форма × интервал позиций). Проверка — GroupKFold по авторам: остаток концовок/начал на отложенных авторах
   до и после поправки, ρ сигнала с LLM до и после.
2. LLM-дуги (все 20 сегментов текста): «картина концовок» по форме — контраст «последние 2 сегмента − середина
   (3…18)» по тексту, (а) в баллах шкалы, (б) после z-нормализации дуги внутри текста (форма дуги);
   сигналы: LLM, классификатор (калибр.), классификатор (калибр. + поправка на позицию), RSL (то же).
   Среднее по текстам, 95 % ДИ — бутстрэп по авторам. Согласие дуг: медиана по текстам ρ(LLM, сигнал) по 20 позициям.
   Для повестей и романов LLM видела отрывок ≈ 350 слов, сигналы — целый сегмент (как в основном расчёте).
3. Эмоции LLM (только описательно): распределение по зонам «начало / середина / конец» по формам.

Запуск:  .venv/bin/python src/s02_arcs_compare.py configs/s02_ensemble.yaml
Выход:   reports/02_llm_arcs.md, reports/figures/02_llm_arcs_mean.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
import s02_validate as V  # noqa: E402
from s02_ensemble import boot_groups  # noqa: E402

ST = ["afanasyev", "fairytales", "story", "novella", "fanfic", "povest", "novel"]
ST_RU = {"afanasyev": "волшебная сказка (Афанасьев)", "fairytales": "сказка (fairytalez)", "story": "рассказ",
         "novella": "новелла", "fanfic": "фанфик", "povest": "повесть", "novel": "роман"}
PBINS = [(1, 2), (3, 5), (6, 8), (9, 12), (13, 15), (16, 18), (19, 20)]
SIGS = {"clf": "Классификатор_тональности", "rsl": "RSL"}


def pbin(pos):
    pos = np.asarray(pos)
    out = np.zeros(len(pos), int)
    for i, (a, b) in enumerate(PBINS):
        out[(pos >= a) & (pos <= b)] = i
    return out


def fit_correction(d, sig):
    """калибровка (a, b) по форме + средний остаток по (форма, интервал позиций)."""
    cal, corr = {}, {}
    for st in ST:
        x = d[d["stratum"] == st]
        b, a = np.polyfit(x[sig], x["tone"], 1)
        cal[st] = (a, b)
        r = x["tone"] - (a + b * x[sig])
        pb = pbin(x["pos"])
        corr[st] = np.array([r[pb == i].mean() if (pb == i).any() else 0.0 for i in range(len(PBINS))])
    return cal, corr


def apply(df, sig_values, cal, corr, positional=True):
    out = np.zeros(len(df))
    for st in ST:
        m = (df["stratum"] == st).to_numpy()
        if not m.any():
            continue
        a, b = cal[st]
        out[m] = a + b * sig_values[m]
        if positional:
            out[m] += corr[st][pbin(df["pos"].to_numpy()[m])]
    return out


def edge(r, pos):
    mid = (pos >= 3) & (pos <= 18)
    return r[pos >= 19].mean() - r[mid].mean(), r[pos <= 2].mean() - r[mid].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["bootstrap"]
    lcfg = load_config(ROOT / cfg["llm_config"])
    acfg = load_config(ROOT / "configs" / "s02_llm_arcs.yaml")
    sil = ROOT / lcfg["out_dir"]
    R = ["# LLM-дуги тональности и проверка классификатора с поправкой на позицию (Ворота 2, п. 2.2)\n"]

    # ---------- 1. поправка на позицию (серебро) ----------
    key = pd.read_parquet(sil / "silver_key.parquet", columns=["uid", "seg"])
    d = pd.read_parquet(sil / "silver_signals.parquet").drop(columns=["text"]).merge(
        V.read_ann(sil / f"silver__{lcfg['models']['main']}.csv"), on="uid").merge(key, on="uid")
    d = d[d["skip"] == 0].reset_index(drop=True)
    d["pos"] = d["seg"] + 1
    g = d["author_id"].astype(str).to_numpy()
    R.append("## 1. Поправка на позицию (серебро, GroupKFold по авторам, 5 блоков)\n")
    R.append("Остаток на отложенных авторах: «конец» = последние 2 сегмента − середина (3…18), «начало» = первые 2 − середина; "
             "ρ — Спирмен прогноза с LLM-тональностью внутри формы.\n")
    rows = []
    for nm, sig in SIGS.items():
        pred0, pred1 = np.zeros(len(d)), np.zeros(len(d))
        for tr, te in GroupKFold(5).split(d, groups=g):
            cal, corr = fit_correction(d.iloc[tr], sig)
            pred0[te] = apply(d.iloc[te], d[sig].to_numpy()[te], cal, corr, positional=False)
            pred1[te] = apply(d.iloc[te], d[sig].to_numpy()[te], cal, corr, positional=True)
        for st in ST:
            m = (d["stratum"] == st).to_numpy()
            pos, y = d["pos"].to_numpy()[m], d["tone"].to_numpy()[m]
            e0, b0 = edge(y - pred0[m], pos)
            e1, b1 = edge(y - pred1[m], pos)
            rows.append({"сигнал": nm, "форма": ST_RU[st], "конец: без поправки": f"{e0:+.2f}", "конец: с поправкой": f"{e1:+.2f}",
                         "начало: без поправки": f"{b0:+.2f}", "начало: с поправкой": f"{b1:+.2f}",
                         "ρ без / с поправкой": f"{stats.spearmanr(pred0[m], y).statistic:.2f} / {stats.spearmanr(pred1[m], y).statistic:.2f}"})
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    fits = {nm: fit_correction(d, sig) for nm, sig in SIGS.items()}
    R.append("Поправка (на всём серебре), средний остаток по интервалам позиций " + ", ".join(f"{a}–{b}" for a, b in PBINS) + ":\n")
    rows = []
    for nm in SIGS:
        for st in ST:
            rows.append({"сигнал": nm, "форма": ST_RU[st], **{f"{a}–{b}": f"{v:+.2f}" for (a, b), v in zip(PBINS, fits[nm][1][st])}})
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")

    # ---------- 2. LLM-дуги ----------
    aout = ROOT / acfg["out_dir"]
    ak = pd.read_parquet(aout / "arcs_key.parquet").drop(columns=["text"])
    aa = pd.read_csv(aout / f"arcs__{acfg['models']['main']}.csv")
    ann = V.read_ann(aout / f"arcs__{acfg['models']['main']}.csv")
    cost = pd.to_numeric(aa["cost_rub"], errors="coerce").sum()
    x = ak.merge(ann, on="uid", how="left")
    x["pos"] = x["seg"] + 1
    ok = x.groupby("doc_id").apply(lambda t: (t["skip"] == 0).sum() == 20 and t["tone"].notna().all(), include_groups=False)
    x = x[x["doc_id"].isin(ok[ok].index)].copy()
    clf = pd.read_parquet(ROOT / "data/processed/s02/clf/clf_N20.parquet", columns=["doc_id", "seg", "sent_positive", "sent_negative"],
                          filters=[("doc_id", "in", list(set(x["doc_id"])))])
    lex = pd.read_parquet(ROOT / "data/processed/s02/lex/lex_N20.parquet", columns=["doc_id", "seg", "n_words", "rsl_pos", "rsl_neg"],
                          filters=[("doc_id", "in", list(set(x["doc_id"])))])
    x = x.merge(clf, on=["doc_id", "seg"]).merge(lex, on=["doc_id", "seg"])
    x["Классификатор_тональности"] = x["sent_positive"] - x["sent_negative"]
    x["RSL"] = (x["rsl_pos"] - x["rsl_neg"]) / x["n_words"].clip(lower=1)
    S = {"LLM": x["tone"].to_numpy(float)}
    for nm, sig in SIGS.items():
        cal, corr = fits[nm]
        S[f"{nm} (калибр.)"] = apply(x, x[sig].to_numpy(), cal, corr, positional=False)
        S[f"{nm} (калибр. + поправка)"] = apply(x, x[sig].to_numpy(), cal, corr, positional=True)
    for k_, v in S.items():
        x[k_] = v
    R.append(f"## 2. LLM-дуги: {x['doc_id'].nunique()} текстов полностью размечены (из {ak['doc_id'].nunique()}); "
             f"стоимость {cost:.2f} ₽\n")
    R.append(V.md_table(x.groupby("stratum").agg(текстов=("doc_id", "nunique"), авторов=("author_id", "nunique"))
                        .reindex(ST).rename(index=ST_RU).reset_index().rename(columns={"stratum": "форма"})) + "\n")

    def per_text(col, z):
        rows_ = []
        for doc, t in x.groupby("doc_id", sort=False):
            t = t.sort_values("pos")
            v = t[col].to_numpy(float)
            if z:
                sd = v.std()
                v = (v - v.mean()) / sd if sd > 0 else v * 0
            pos = t["pos"].to_numpy()
            e, b = edge(v, pos)
            rows_.append((doc, t["stratum"].iloc[0], str(t["author_id"].iloc[0]), e, b))
        return pd.DataFrame(rows_, columns=["doc_id", "stratum", "author_id", "end", "beg"])

    for z, title in [(False, "в баллах шкалы −2…+2"), (True, "после z-нормализации дуги внутри текста (форма дуги)")]:
        R.append(f"### Картина концовок: последние 2 сегмента − середина, {title}\n")
        rows = []
        for col in S:
            pt = per_text(col, z)
            r = {"сигнал": col}
            for st in ST:
                q = pt[pt["stratum"] == st]
                if not len(q):
                    continue
                v, gg = q["end"].to_numpy(), q["author_id"].to_numpy()
                bs = [v[t].mean() for t in boot_groups(gg, rng, B)]
                r[ST_RU[st]] = f"{v.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}; {np.percentile(bs, 97.5):+.2f}]"
            rows.append(r)
        R.append(V.md_table(pd.DataFrame(rows)) + "\n")
        if z:
            R.append("### Начала: первые 2 сегмента − середина, после z-нормализации\n")
            rows = []
            for col in S:
                pt = per_text(col, True)
                r = {"сигнал": col}
                for st in ST:
                    q = pt[pt["stratum"] == st]
                    if len(q):
                        v, gg = q["beg"].to_numpy(), q["author_id"].to_numpy()
                        bs = [v[t].mean() for t in boot_groups(gg, rng, B)]
                        r[ST_RU[st]] = f"{v.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}; {np.percentile(bs, 97.5):+.2f}]"
                rows.append(r)
            R.append(V.md_table(pd.DataFrame(rows)) + "\n")

    # различия форм на LLM-дугах: разница с волшебной сказкой (Афанасьев), бутстрэп по авторам внутри каждой формы
    R.append("### Сказка против прозы на LLM-дугах: разница «форма − Афанасьев» [95 % ДИ, бутстрэп по авторам]\n")
    R.append("Рассказы, новеллы, фанфики отобраны в диапазоне длины сказок базы (5–95-й перцентиль), поэтому сравнение "
             "с ними сопоставлено по длине; повести и романы — отрывками ≈ 350 слов, по длине не сопоставимы.\n")
    rows = []
    for col, z, nm in [("LLM", False, "концовка, баллы"), ("LLM", True, "концовка, z"), ("LLM", True, "начало, z")]:
        pt = per_text(col, z)
        key_ = "beg" if nm.startswith("начало") else "end"
        base = pt[pt["stratum"] == "afanasyev"]
        bv, bg = base[key_].to_numpy(), base["author_id"].to_numpy()
        bb = np.array([bv[t].mean() for t in boot_groups(bg, rng, B)])
        r = {"метрика": nm, "Афанасьев": f"{bv.mean():+.2f}"}
        for st in ST[1:]:
            q = pt[pt["stratum"] == st]
            v, gg = q[key_].to_numpy(), q["author_id"].to_numpy()
            qb = np.array([v[t].mean() for t in boot_groups(gg, rng, B)])
            dd = qb - bb[:len(qb)]
            r[ST_RU[st]] = f"{v.mean() - bv.mean():+.2f} [{np.percentile(dd, 2.5):+.2f}; {np.percentile(dd, 97.5):+.2f}]"
        rows.append(r)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")

    R.append("### Согласие дуг: медиана по текстам ρ Спирмена (LLM, сигнал) по 20 позициям [IQR]\n")
    rows = []
    for col in [c for c in S if c != "LLM"]:
        r = {"сигнал": col}
        for st in ST:
            rr = []
            for _, t in x[x["stratum"] == st].groupby("doc_id"):
                if t["LLM"].std() > 0 and t[col].std() > 0:
                    rr.append(stats.spearmanr(t["LLM"], t[col]).statistic)
            if rr:
                r[ST_RU[st]] = f"{np.median(rr):.2f} [{np.percentile(rr, 25):.2f}; {np.percentile(rr, 75):.2f}]"
        rows.append(r)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")

    # рисунок: средние дуги
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    axes = axes.ravel()
    sty = {"LLM": ("#0b0b0b", 2.2, "-"), "clf (калибр.)": ("#2a5fb0", 1.4, "--"), "clf (калибр. + поправка)": ("#2a5fb0", 1.6, "-"),
           "rsl (калибр.)": ("#c0392b", 1.2, "--"), "rsl (калибр. + поправка)": ("#c0392b", 1.4, "-")}
    for ax, st in zip(axes, ST):
        t = x[x["stratum"] == st]
        for col, (c, lw, ls) in sty.items():
            ax.plot(range(1, 21), t.groupby("pos")[col].mean().reindex(range(1, 21)), color=c, lw=lw, ls=ls, label=col)
        ax.set_title(f"{ST_RU[st]} ({t['doc_id'].nunique()})", fontsize=10)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.grid(alpha=0.25)
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[-1].axis("off")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("тональность, шкала LLM")
    axes[4].set_ylabel("тональность, шкала LLM")
    fig.tight_layout()
    fp = ROOT / "reports" / "figures" / "02_llm_arcs_mean.png"
    fig.savefig(fp, dpi=130)
    R.append(f"![Средние дуги](figures/{fp.name})\n")

    # ---------- 3. эмоции LLM (описательно) ----------
    R.append("## 3. Эмоции LLM по зонам текста (только описательно)\n")
    x["зона"] = np.where(x["pos"] <= 2, "начало (1–2)", np.where(x["pos"] >= 19, "конец (19–20)", "середина (3–18)"))
    emo = ann.set_index("uid")["emotion"]
    x["emotion"] = x["uid"].map(emo)
    tab = (x[x["emotion"] != ""].groupby(["stratum", "зона"])["emotion"].value_counts(normalize=True).unstack(fill_value=0) * 100)
    tab = tab.reindex(ST, level=0).round(1).reset_index()
    tab["stratum"] = tab["stratum"].map(ST_RU)
    R.append("Доля сегментов с данной доминирующей эмоцией, %.\n")
    R.append(V.md_table(tab.rename(columns={"stratum": "форма"})) + "\n")
    (ROOT / "reports" / "02_llm_arcs.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
