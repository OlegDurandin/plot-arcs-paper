"""Этап 10: рисунки основного текста (Ворота 6, 2026-11-26). Новых расчётов сигналов нет — только готовые дуги.

Рис. 1 — архетипы как артефакт сглаживания: первые моды SVD настоящих и перемешанных дуг (N = 20, классификатор,
         все тексты within_author_cap) при скользящем среднем и без сглаживания. Запасной вариант (1alt) — центроиды
         Уорда при k = 6 (как у Reagan et al.) на той же подвыборке, что в 04_archetype_null.
Рис. 2 — средние дуги (z внутри текста) волшебной сказки и прозы той же длины: LLM (95 % ДИ — бутстрэп по авторам,
         полоса перемешанных) и классификатор на тех же текстах поверх.
Подписи и числа для них — в reports/10_figures.md.

Запуск:  .venv/bin/python src/s10_figures.py configs/s10_figures.yaml
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s04_archetype_null import match, paa, prep, svd_modes, ward  # noqa: E402
import s04_archetype_null as AN  # noqa: E402
from s04_nulls import md, zscore  # noqa: E402
from s05_ending import llm_arcs  # noqa: E402
from s05_mean_arcs import LLM_RU, mean_ci, shuffle_band  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
import matplotlib.ticker as mticker  # noqa: E402

COMMA = mticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))   # русская десятичная запятая


def ru(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def commas(axes):
    for ax in np.ravel(axes):
        ax.yaxis.set_major_formatter(COMMA)


SM_RU = {"ma": "скользящее среднее (15 % текста)", "none": "без сглаживания", "gauss": "гауссово ядро"}


def save(fig, cfg, name):
    paths = []
    for ext in cfg["formats"]:
        p = ROOT / f"{cfg['out_prefix']}{name}.{ext}"
        fig.savefig(p, dpi=cfg["dpi"], bbox_inches="tight")
        paths.append(p.relative_to(ROOT).as_posix())
    plt.close(fig)
    return paths


def fig1(cfg, L):
    c, col = cfg["fig1"], cfg["colors"]
    ncfg = load_config(ROOT / cfg["nulls_config"])
    an = ncfg["archetype_null"]
    AN.SIGMA = an.get("gauss_sigma_frac", AN.SIGMA)
    frac = ncfg["arc"]["ma_frac"]
    rng = np.random.default_rng(cfg["seed"])
    arcs = pd.read_parquet(ROOT / ncfg["out_dir"] / "full" / f"arcs_N{c['N']}.parquet")
    arcs = arcs[arcs["within_author_cap"]].reset_index(drop=True)
    X = np.vstack(arcs[c["signal"]].to_numpy()).astype(np.float64)
    ok = X.std(1) > 0
    X = X[ok]
    shuf = [np.take_along_axis(X, np.argsort(rng.random(X.shape), 1), 1) for _ in range(c["n_shuffle_reps"])]
    k = c["svd_modes"]
    fig, axes = plt.subplots(len(c["smoothing"]), k, figsize=(7.2, 4.4), sharex=True)
    xs = np.arange(1, c["N"] + 1)
    L.append(f"## Рис. 1. Архетипы — артефакт сглаживания\n\nДанные: {len(X)} дуг N = {c['N']}, классификатор, within_author_cap; "
             f"перемешанные — те же тексты, сегменты переставлены, {c['n_shuffle_reps']} повторов.\n")
    rows = []
    for r_i, sm in enumerate(c["smoothing"]):
        vr, Vr, _ = svd_modes(prep(X, sm, frac), k)
        Vs, vs = [], []
        for Xs in shuf:
            v_, V_, _ = svd_modes(prep(Xs, sm, frac), k)
            # знак моды перемешанных — по совпадению с настоящей
            V_ = V_ * np.sign((V_ * Vr).sum(1, keepdims=True))
            Vs.append(V_); vs.append(v_)
        Vs, vs = np.array(Vs), np.array(vs)
        for j in range(k):
            ax = axes[r_i, j]
            for t in range(len(Vs)):
                ax.plot(xs, Vs[t, j], color=col["clf"], lw=1, alpha=0.7, label="перемешанные дуги" if t == 0 else None)
            ax.plot(xs, Vr[j], color=col["llm"], lw=2, label="настоящие дуги")
            ax.axhline(0, color="#cfceca", lw=0.6)
            rr = [abs(np.corrcoef(Vr[j], Vs[t, j])[0, 1]) for t in range(len(Vs))]
            ax.set_title(f"мода {j + 1} · |r| = {ru(np.mean(rr))}", fontsize=8.5)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=0.2)
            rows.append(dict(сглаживание=SM_RU[sm], мода=j + 1, доля_настоящие=round(float(vr[j]), 3),
                             доля_перемешанные=round(float(vs[:, j].mean()), 3), r_формы=round(float(np.mean(rr)), 2)))
        axes[r_i, 0].set_ylabel(SM_RU[sm].replace(" (", "\n("), fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("позиция сегмента (N = 20)")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    commas(axes)
    fig.tight_layout()
    paths = save(fig, cfg, "fig1_archetypes_svd")
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("Подпись (черновик): «Первые три моды SVD z-дуг тональности (классификатор, N = 20) для настоящих текстов "
             "(синие) и тех же текстов с переставленными сегментами (оранжевые, 5 повторов). При скользящем среднем моды "
             "перемешанных дуг повторяют моды настоящих (|r| = 0,96–0,99): «архетипы» создаёт сглаживание. Без "
             "сглаживания совпадения нет (|r| = 0,14–0,21). Доля дисперсии первых трёх мод: со сглаживанием 0,23 / 0,18 / "
             "0,15 у настоящих и 0,17 / 0,16 / 0,14 у перемешанных.» Числа — в таблице выше.\n")
    L.append("Файлы: " + ", ".join(f"`{p}`" for p in paths) + "\n")
    # запасной вариант: центроиды Уорда k = 6
    ward_idx = np.concatenate([g.sample(min(len(g), an["ward_per_form"]), random_state=cfg["seed"]).index.to_numpy()
                               for _, g in arcs.groupby("fk")])
    widx = np.searchsorted(np.where(ok)[0], np.intersect1d(ward_idx, np.where(ok)[0]))
    kk = c["alt_ward_k"]
    Pr = paa(prep(X, "ma", frac)[widx], an["paa_points"])
    _, cen_r, sz_r, sil_r, _ = ward(Pr, an["k_range"], kk)
    cens, rr, sils = [], [], []
    for Xs in shuf:
        _, cs, _, sil_s, _ = ward(paa(prep(Xs, "ma", frac)[widx], an["paa_points"]), an["k_range"], kk)
        r_, order = match(cen_r, cs)
        cens.append(cs[order]); rr.append(r_); sils.append(sil_s)
    rr = np.array(rr)
    fig, axes = plt.subplots(1, kk, figsize=(7.2, 1.9), sharey=True)
    xp = np.arange(1, an["paa_points"] + 1)
    for i in range(kk):
        ax = axes[i]
        for t, cs in enumerate(cens):
            ax.plot(xp, cs[i], color=col["clf"], lw=1, alpha=0.7, label="перемешанные" if (i == 0 and t == 0) else None)
        ax.plot(xp, cen_r[i], color=col["llm"], lw=2, label="настоящие" if i == 0 else None)
        ax.axhline(0, color="#cfceca", lw=0.6)
        ax.set_title(f"{sz_r[i]:.0%} · r = {ru(rr[:, i].mean())}".replace("%", " %"), fontsize=8)
        ax.set_xticks([1, 5, 10])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("z (PAA, 10 точек)", fontsize=8)
    commas(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    paths = save(fig, cfg, "fig1alt_archetypes_ward")
    L.append(f"### Запасной вариант Рис. 1: центроиды Уорда, k = {kk}, скользящее среднее\n\n{len(widx)} текстов (≤ "
             f"{an['ward_per_form']} на форму). Силуэт: настоящие {sil_r:.3f}, перемешанные {np.mean(sils):.3f}. "
             f"r сопоставленных центроидов: {', '.join(f'{v:.2f}' for v in rr.mean(0))} (среднее {rr.mean():.2f}).\n")
    L.append("Файлы: " + ", ".join(f"`{p}`" for p in paths) + "\n")


def fig2(cfg, L):
    c, col = cfg["fig2"], cfg["colors"]
    acfg = load_config(ROOT / cfg["arcs_config"])
    ncfg = load_config(ROOT / cfg["nulls_config"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    docs, T = llm_arcs(acfg)
    Zl = zscore(T)
    clf = pd.read_parquet(ROOT / ncfg["out_dir"] / "full" / "arcs_N20.parquet", columns=["doc_id", "clf"]).set_index("doc_id")["clf"]
    fig, axes = plt.subplots(1, len(c["groups"]), figsize=(7.2, 2.6), sharey=True)
    xs = np.arange(1, 21)
    rows = []
    for ax, g in zip(axes, c["groups"]):
        m_ = ((docs["stratum"] == g).to_numpy()) & ~np.isnan(Zl).any(1) & docs["doc_id"].isin(clf.index).to_numpy()
        ids = docs.loc[m_, "doc_id"].to_numpy()
        au = docs.loc[m_, "author_id"].astype(str).to_numpy()
        Zc = zscore(np.vstack(clf.loc[ids].to_numpy()).astype(np.float64))
        okc = ~np.isnan(Zc).any(1)
        Z_l, Z_c, au = Zl[m_][okc], Zc[okc], au[okc]
        ml, lol, hil = mean_ci(Z_l, au, rng, c["bootstrap"])
        mc, loc, hic = mean_ci(Z_c, au, rng, c["bootstrap"])
        band = shuffle_band(Z_l, rng, c["n_shuffle"])
        bandc = shuffle_band(Z_c, rng, c["n_shuffle"])
        p_l = (np.sum((band ** 2).sum(1) >= (ml ** 2).sum()) + 1) / (c["n_shuffle"] + 1)
        p_c = (np.sum((bandc ** 2).sum(1) >= (mc ** 2).sum()) + 1) / (c["n_shuffle"] + 1)
        ax.fill_between(xs, *np.percentile(band, [2.5, 97.5], 0), color=col["shuffled"], alpha=0.25, lw=0,
                        label="перемешанные, 95 %")
        ax.fill_between(xs, loc, hic, color=col["clf"], alpha=0.15, lw=0)
        ax.plot(xs, mc, color=col["clf"], lw=1.5, label="классификатор")
        ax.fill_between(xs, lol, hil, color=col["llm"], alpha=0.18, lw=0)
        ax.plot(xs, ml, color=col["llm"], lw=2, label="LLM")
        ax.axhline(0, color="#cfceca", lw=0.6)
        name = LLM_RU[g].replace("волшебная сказка (Афанасьев)", "волшебная сказка\n(Афанасьев)").replace("сказка (fairytalez)", "сказка\n(fairytalez)")
        ax.set_title(f"{name}, n = {len(Z_l)}", fontsize=8.5)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("позиция сегмента")
        ax.grid(alpha=0.2)
        rows.append(dict(группа=LLM_RU[g], текстов=len(Z_l), p_LLM_против_перемешанных=round(float(p_l), 3),
                         p_классификатор=round(float(p_c), 3), LLM_позиция_20=round(float(ml[-1]), 2),
                         классификатор_позиция_20=round(float(mc[-1]), 2)))
    axes[0].set_ylabel("средняя z-дуга")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    commas(axes)
    fig.tight_layout()
    paths = save(fig, cfg, "fig2_mean_arcs")
    L.append("## Рис. 2. Средние дуги: волшебная сказка и проза той же длины, LLM и классификатор\n\n"
             "z внутри текста, N = 20; тексты выборки LLM-дуг (проза — в диапазоне длины сказок базы, ≤ 1 текста на "
             f"автора); классификатор — на тех же текстах; ДИ 95 % — пуассоновский бутстрэп по авторам ({c['bootstrap']}); "
             f"полоса — средняя LLM-дуга перемешанных текстов ({c['n_shuffle']} перестановок).\n")
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("Подпись (черновик): «Средние дуги тональности (z внутри текста, N = 20) по LLM (синие, 95 % ДИ) и "
             "классификатору на тех же текстах (оранжевые); серая полоса — 95 % средней LLM-дуги тех же текстов с "
             "переставленными сегментами. У волшебной сказки LLM видит тёмное начало и резкий подъём в последнем сегменте; "
             "классификатор подъёма не видит.» p в таблице — SS средней дуги против перемешанных (на этой выборке; "
             "p = 0,975 в тексте — классификатор на полном корпусе, этап 5).\n")
    L.append("Файлы: " + ", ".join(f"`{p}`" for p in paths) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    L = ["# Этап 10: рисунки основного текста (черновые подписи)\n",
         "Состав утверждён на Воротах 6 (2026-11-26). Шкала Рис. 2 — z (решение пользователя). Язык подписей — русский.\n"]
    fig1(cfg, L)
    fig2(cfg, L)
    (ROOT / "reports" / "10_figures.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
