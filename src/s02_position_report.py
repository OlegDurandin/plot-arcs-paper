"""Этап 2, Ворота 2 п. 2.1 (методический раздел): позиционная ошибка сигналов тональности на серебре.

Остаток = LLM-тональность − калибровка сигнала внутри формы (tone ~ a + b·сигнал; снимает сдвиг уровня и масштаб).
  * рисунок: средний остаток по позиции 1…20 для каждой формы, классификатор и RSL, 95 % ДИ — бутстрэп по авторам;
  * примеры: 15 концовок сказок (последние 2 сегмента) и 15 начал новелл и повестей (первые 2) с наибольшим остатком;
  * связь остатка концовок сказок с длиной сегмента.

Запуск:  .venv/bin/python src/s02_position_report.py configs/s02_ensemble.yaml
Выход:   reports/02_position_error.md, reports/figures/02_position_error_by_form.png
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
import s02_validate as V  # noqa: E402
from s02_ensemble import boot_groups  # noqa: E402

ST = ["afanasyev", "fairytales", "story", "novella", "fanfic", "povest", "novel"]
ST_RU = {"afanasyev": "волшебная сказка (Афанасьев)", "fairytales": "сказка (fairytalez)", "story": "рассказ",
         "novella": "новелла", "fanfic": "фанфик", "povest": "повесть (отрывки)", "novel": "роман (отрывки)"}
SIG = {"Классификатор_тональности": "классификатор", "RSL": "RSL"}
COL = {"Классификатор_тональности": "#2a5fb0", "RSL": "#c0392b"}


def load(cfg):
    lcfg = load_config(ROOT / cfg["llm_config"])
    out = ROOT / lcfg["out_dir"]
    key = pd.read_parquet(out / "silver_key.parquet", columns=["uid", "seg", "text"])
    sig = pd.read_parquet(out / "silver_signals.parquet").drop(columns=["text"])
    ann = V.read_ann(out / f"silver__{lcfg['models']['main']}.csv")
    d = sig.merge(ann, on="uid").merge(key, on="uid")
    d = d[d["skip"] == 0].reset_index(drop=True)
    d["pos"] = d["seg"] + 1
    for s_ in SIG:
        d[f"res_{s_}"] = np.nan
        d[f"cal_{s_}"] = np.nan
        for st in ST:
            m = d["stratum"] == st
            b, a = np.polyfit(d.loc[m, s_], d.loc[m, "tone"], 1)
            d.loc[m, f"cal_{s_}"] = a + b * d.loc[m, s_]
            d.loc[m, f"res_{s_}"] = d.loc[m, "tone"] - d.loc[m, f"cal_{s_}"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["bootstrap"]
    d = load(cfg)
    R = ["# Позиционная ошибка сигналов тональности (Ворота 2, п. 2.1; методический раздел)\n",
         f"Серебро: {len(d)} сегментов N = 20 (deepseek-v4.1-flash). Остаток = LLM-тональность − калибровка сигнала "
         "внутри формы (tone ~ a + b·сигнал; снимает сдвиг уровня и масштаб формы), в баллах шкалы −2…+2. "
         "Положительный остаток — сигнал **занижает** тональность на этой позиции. 95 % ДИ — бутстрэп по авторам "
         f"(B = {B}).\n"]

    # ---------- рисунок по позициям ----------
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharey=True)
    axes = axes.ravel()
    tab = []
    for ax, st in zip(axes, ST):
        x = d[d["stratum"] == st]
        for s_ in SIG:
            means, lo, hi = [], [], []
            for p in range(1, 21):
                y = x[x["pos"] == p]
                v, g = y[f"res_{s_}"].to_numpy(), y["author_id"].astype(str).to_numpy()
                means.append(v.mean())
                bs = [v[t].mean() for t in boot_groups(g, rng, B)] if len(v) > 1 else [np.nan]
                lo.append(np.nanpercentile(bs, 2.5)); hi.append(np.nanpercentile(bs, 97.5))
                tab.append({"форма": ST_RU[st], "сигнал": SIG[s_], "позиция": p, "n": len(v), "остаток": v.mean(),
                            "ДИ_низ": lo[-1], "ДИ_верх": hi[-1]})
            xs = np.arange(1, 21) + (-0.12 if s_ == "RSL" else 0.12)
            ax.fill_between(xs, lo, hi, color=COL[s_], alpha=0.12, lw=0)
            ax.plot(xs, means, color=COL[s_], lw=1.8, marker="o", ms=3, label=SIG[s_])
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(f"{ST_RU[st]} (n = {len(x)})", fontsize=10)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    axes[0].legend(fontsize=9, loc="upper left")
    for ax in axes[4:7]:
        ax.set_xlabel("позиция сегмента (N = 20)")
    axes[0].set_ylabel("остаток LLM − калибровка, баллы")
    axes[4].set_ylabel("остаток LLM − калибровка, баллы")
    fig.tight_layout()
    fp = ROOT / "reports" / "figures" / "02_position_error_by_form.png"
    fig.savefig(fp, dpi=130)
    pd.DataFrame(tab).to_csv(ROOT / "reports" / "02_position_error_by_form.csv", index=False)
    R.append("## 1. Остаток по позиции, все формы\n")
    R.append(f"![Остаток по позиции](figures/{fp.name})\n")
    R.append("Таблица значений (позиция × форма × сигнал, n, остаток, ДИ): `reports/02_position_error_by_form.csv`. "
             "Позиции сэмплированы случайно, n на позицию у Афанасьева 24–45 — ДИ отдельной позиции широкие; "
             "контрасты «первые/последние 2 против середины» — в `reports/02_ensemble_position.md`.\n")

    # ---------- примеры ----------
    def examples(mask, title, n=15):
        s_ = "Классификатор_тональности"
        x = d[mask].sort_values(f"res_{s_}", ascending=False).head(n)
        R.append(f"## {title}\n")
        R.append("| # | форма | позиция | слов | LLM | классификатор (сырой) | калибр. | RSL | остаток | текст |\n|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(x.itertuples(), 1):
            t = r.text.replace("|", "／").replace("\n", " ")
            t = t if len(t) <= 700 else t[:350] + " … " + t[-330:]
            R.append(f"| {i} | {ST_RU[r.stratum]} | {r.pos} | {r.shown_n_words} | {r.tone:+.0f} | {getattr(r, 'Классификатор_тональности'):+.2f} | "
                     f"{getattr(r, 'cal_Классификатор_тональности'):+.2f} | {r.RSL:+.3f} | {getattr(r, 'res_Классификатор_тональности'):+.2f} | {t} |")
        R.append("")
        return x
    tales = d["stratum"].isin(["afanasyev", "fairytales"])
    ends = examples(tales & (d["pos"] >= 19), "2. 15 концовок сказок (позиции 19–20) с наибольшим остатком классификатора")
    begs = examples(d["stratum"].isin(["novella", "povest"]) & (d["pos"] <= 2),
                    "3. 15 начал новелл и повестей (позиции 1–2) с наибольшим остатком классификатора")
    for nm, x in [("ends", ends), ("begs", begs)]:
        x[["uid", "doc_id", "stratum", "pos", "shown_n_words", "tone", "Классификатор_тональности", "RSL",
           "res_Классификатор_тональности", "text"]].to_csv(ROOT / "reports" / f"02_position_examples_{nm}.csv", index=False)

    # ---------- длина сегмента и остаток концовок ----------
    R.append("## 4. Связан ли остаток концовок сказок с длиной сегмента\n")
    rows = []
    for st in ["afanasyev", "fairytales"]:
        x = d[d["stratum"] == st].copy()
        x["зона"] = np.where(x["pos"] >= 19, "конец", np.where(x["pos"] <= 2, "начало", "середина"))
        for s_ in SIG:
            e = x[x["зона"] == "конец"]
            rho = stats.spearmanr(e["shown_n_words"], e[f"res_{s_}"]).statistic
            # остаток концовки внутри терцилей длины (терцили по всем сегментам формы)
            q = np.quantile(x["shown_n_words"], [1 / 3, 2 / 3])
            x["длина"] = pd.cut(x["shown_n_words"], [0, q[0], q[1], 1e9], labels=["короткие", "средние", "длинные"])
            r = {"форма": ST_RU[st], "сигнал": SIG[s_],
                 "медиана слов: конец / середина": f"{e['shown_n_words'].median():.0f} / {x.loc[x['зона'] == 'середина', 'shown_n_words'].median():.0f}",
                 "терцили длины, слов": f"≤{q[0]:.0f} / {q[0]:.0f}–{q[1]:.0f} / >{q[1]:.0f}",
                 "ρ(длина, остаток) в концовках": f"{rho:+.2f} (n = {len(e)})"}
            for b_, g in x.groupby("длина", observed=True):
                ce, cm = g.loc[g["зона"] == "конец", f"res_{s_}"], g.loc[g["зона"] == "середина", f"res_{s_}"]
                r[f"конец − середина: {b_}"] = f"{ce.mean() - cm.mean():+.2f} (n конц. {len(ce)})"
            # регрессия: остаток ~ конец + log(слов), кластер по авторам
            import statsmodels.formula.api as smf
            x["end"] = (x["зона"] == "конец").astype(int)
            x["lw"] = np.log(x["shown_n_words"].clip(lower=1))
            x["rr"] = x[f"res_{s_}"]
            m = smf.ols("rr ~ end + lw", x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["author_id"].astype(str))[0]})
            lo, hi = m.conf_int().loc["end"]
            r["эффект концовки при контроле log(слов)"] = f"{m.params['end']:+.2f} [{lo:+.2f}; {hi:+.2f}]"
            rows.append(r)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    (ROOT / "reports" / "02_position_error.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
