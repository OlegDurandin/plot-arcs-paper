"""Этап 6. RQ2 — расстояние до волшебной сказки (решения Ворот 5, 2026-11-26).

Основной анализ — только в пределах одного сигнала: LLM-дуги прозы до LLM-базы (средняя LLM-дуга сказок базы
Афанасьева). Расстояние «классификаторная дуга прозы → LLM-база» смешивает два измерителя и не считается.
  Варианты (configs/s06_distance.yaml): все 20 позиций (евклидово); без позиции 20 (z пересчитан по 19 позициям);
  позиции 1–19 с DTW (полоса Сакоэ — Чибы). Доля расстояния, приходящаяся на концовку, — из разложения D² по позициям.
  Агрегатный уровень (главный): D = расстояние средней дуги группы до базы; ДИ — пуассоновский бутстрэп по авторам
  (база и группа независимо); p — перестановка меток «база / группа»; ориентиры: D перемешанных дуг группы и
  D двух случайных половин самих сказок (шум выборки).
  Длина: внутри коротких форм (рассказ, новелла, фанфик той же длины, что сказки) — расстояние текста до базы
  ~ форма + log(n_words) (бутстрэп по текстам; ≤ 1 текста на автора) и D по терцилям длины; повести и романы — отдельные
  точки (отрывки ≈ 350 слов на сегмент).
Вторичный анализ (--secondary): классификатор на полном корпусе до классификаторной базы — с оговоркой о позиционной
ошибке (концовки сказок занижены, reports/02_position_error.md); модель с контролем длины и наивная — без него.

Запуск:  .venv/bin/python src/s06_distance.py configs/s06_distance.yaml [--secondary [--sample]]
Выход:   reports/06_distance.md (+ 06_distance_secondary[_sample].md), reports/figures/06_*.png, data/processed/s06/*.parquet
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
import statsmodels.formula.api as smf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s03_report import FORM_RU  # noqa: E402
from s04_nulls import ORDER, md, zscore  # noqa: E402
from s05_ending import llm_arcs  # noqa: E402
from s05_mean_arcs import COL, LLM_RU, author_sums  # noqa: E402

VAR_RU = {"all20": "все 20 позиций", "wo20": "без позиции 20", "dtw1_19": "позиции 1–19, DTW"}


def dtw(X, b, w):
    """DTW-расстояние каждой строки X (n, L) до b (L,), квадрат разности, полоса |i − j| ≤ w. Векторно по n."""
    X = np.atleast_2d(X)
    n, L = X.shape
    D = np.full((n, L + 1, L + 1), np.inf)
    D[:, 0, 0] = 0.0
    for i in range(1, L + 1):
        for j in range(max(1, i - w), min(L, i + w) + 1):
            c = (X[:, i - 1] - b[j - 1]) ** 2
            D[:, i, j] = c + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return np.sqrt(D[:, L, L])


def dist(X, b, use_dtw, w):
    X = np.atleast_2d(X)
    return dtw(X, b, w) if use_dtw else np.sqrt(((X - b) ** 2).sum(1))


def boot_means(Z, authors, rng, B):
    S, n = author_sums(Z, authors)
    W = rng.poisson(1.0, (B, len(n))).astype(float)
    return (W @ S) / (W @ n)[:, None], S.sum(0) / n.sum()


def corr(D, Db):
    """Расстояние между выборочными средними смещено вверх шумом выборки (у бутстрэп-копий — вдвое).
    Поправка на смещение на шкале D²: θ = 2·D² − mean(D*²); ДИ — базовый бутстрэп на той же шкале
    [2·D² − q97,5(D*²); 2·D² − q2,5(D*²)]; возвращается √ (отрицательное → 0). Подходит и для DTW."""
    D2, Db2 = D ** 2, np.asarray(Db) ** 2
    th = 2 * D2 - Db2.mean()
    lo, hi = 2 * D2 - np.percentile(Db2, 97.5), 2 * D2 - np.percentile(Db2, 2.5)
    r = lambda x: float(np.sqrt(max(0.0, x)))  # noqa: E731
    return r(th), r(lo), r(hi)


def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}; {hi:.{d}f}]"


def variant_Z(T, v):
    a, b = v["positions"]
    return zscore(T[:, a - 1:b])


# ---------------------------------------------------------------- основной анализ: LLM
def primary(cfg):
    acfg = load_config(ROOT / cfg["arcs_config"])
    rng = np.random.default_rng(cfg["seed"])
    B, P, R, w = cfg["bootstrap"], cfg["n_perm"], cfg["n_shuffle"], cfg["dtw_window"]
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    figd = ROOT / "reports" / "figures"
    docs, T = llm_arcs(acfg)
    groups = [cfg["base_group"]] + cfg["groups"]
    L = ["# Этап 6. RQ2 — расстояние до волшебной сказки\n",
         "Решения Ворот 5 (2026-11-26). База — средняя LLM-дуга сказок базы Афанасьева (deepseek-v4.1-flash, N = 20). "
         "Основной анализ — только LLM-дуги (один измеритель для базы и прозы): рассказы, новеллы, фанфики — 300 текстов той "
         "же длины, что сказки базы (643–3 179 слов), ≤ 1 текста на автора; повести и романы — по 100, каждый сегмент — "
         "отрывок ≈ 350 слов (не целый сегмент — точки для сравнения, не для выводов о длине); fairytalez — для ориентира. "
         f"Дуга — z внутри текста. Расстояние между выборочными средними смещено вверх шумом выборки, поэтому главная "
         "величина — **D с поправкой на смещение** (на шкале D²: 2·D² − среднее бутстрэп-D*²; ДИ — базовый бутстрэп на той "
         f"же шкале; пуассоновский бутстрэп по авторам, база и группа независимо, B = {B}); сырое D — для справки. "
         "p — перестановка меток "
         f"«база / группа» ({P}).\n",
         "Варианты расстояния: **все 20 позиций** (евклидово); **без позиции 20** (z пересчитан по позициям 1–19 — "
         f"последний сегмент не влияет и на нормировку); **позиции 1–19, DTW** (полоса ±{w} позиции).\n"]
    store, per_text = [], []
    curves = {}
    for vk, v in cfg["variants"].items():
        Z = variant_Z(T, v)
        ok = ~np.isnan(Z).any(1)
        d, Zv = docs[ok].reset_index(drop=True), Z[ok]
        au = d["author_id"].astype(str).to_numpy()
        mb = (d["stratum"] == cfg["base_group"]).to_numpy()
        bs_b, base = boot_means(Zv[mb], au[mb], rng, B)
        # ориентир: две случайные половины сказок (расстояние между ними — шум выборки базы)
        half = []
        ib = np.where(mb)[0]
        for _ in range(200):
            p = rng.permutation(ib)
            h1, h2 = Zv[p[:len(p) // 2]].mean(0), Zv[p[len(p) // 2:]].mean(0)
            half.append(dist(h1, h2, v["dtw"], w)[0])
        rows = []
        L.append(f"## Вариант «{VAR_RU[vk]}»: {int(mb.sum())} сказок базы; текстов с нулевой дисперсией дуги исключено — "
                 f"{int((~ok).sum())}\n")
        for g in cfg["groups"]:
            m_ = (d["stratum"] == g).to_numpy()
            bs_g, mg = boot_means(Zv[m_], au[m_], rng, B)
            D = dist(mg, base, v["dtw"], w)[0]
            Db = np.array([dist(bs_g[k], bs_b[k], v["dtw"], w)[0] for k in range(B)])
            pool = np.vstack([Zv[mb], Zv[m_]])
            nb = int(mb.sum())
            D0 = np.empty(P)
            for r in range(P):
                ii = rng.permutation(len(pool))
                D0[r] = dist(pool[ii[nb:]].mean(0), pool[ii[:nb]].mean(0), v["dtw"], w)[0]
            Ds = np.array([dist(np.take_along_axis(Zv[m_], np.argsort(rng.random(Zv[m_].shape), 1), 1).mean(0), base,
                                v["dtw"], w)[0] for _ in range(R)])
            Dc, lo_, hi_ = corr(D, Db)
            row = {"группа": LLM_RU[g], "текстов": int(m_.sum()), "D (с поправкой на шум) [95 % ДИ]": fmt(Dc, lo_, hi_),
                   "D сырое": f"{D:.3f}", "p": f"{(np.sum(D0 >= D) + 1) / (P + 1):.4f}",
                   "D сырое, перемешанные дуги": f"{np.median(Ds):.3f}"}
            share = np.nan
            if vk == "all20":
                diff = mg - base
                # доля с поправкой на шум: из квадрата разницы на каждой позиции вычитается её бутстрэп-дисперсия
                vb = np.var(bs_g, 0) + np.var(bs_b, 0)
                c2 = np.clip(diff ** 2 - vb, 0, None)
                share = c2[-1] / c2.sum()
                row["доля D² на позицию 20"] = f"{share:.2f} (без поправки {diff[-1] ** 2 / (diff ** 2).sum():.2f})"
                curves[g] = (diff, *np.percentile(np.stack([bs_g[k] - bs_b[k] for k in range(B)]), [2.5, 97.5], 0))
            rows.append(row)
            store.append(dict(variant=vk, group=g, n=int(m_.sum()), D_raw=D, D=Dc, D_lo=lo_, D_hi=hi_,
                              p=(np.sum(D0 >= D) + 1) / (P + 1), D_shuffled=np.median(Ds), share_pos20=share))
            dt = dist(Zv[m_], base, v["dtw"], w)
            per_text.append(pd.DataFrame({"variant": vk, "doc_id": d.loc[m_, "doc_id"].to_numpy(), "group": g,
                                          "n_words": d.loc[m_, "n_words"].to_numpy(), "author_id": au[m_], "d": dt}))
        L.append(md(pd.DataFrame(rows)) + "\n")
        L.append(f"Ориентиры: |база| (расстояние плоской дуги до базы) = {dist(np.zeros_like(base), base, v['dtw'], w)[0]:.3f}; "
                 f"сырое D двух случайных половин сказок базы друг до друга — медиана {np.median(half):.3f} "
                 f"(95 % — {np.percentile(half, 95):.3f}) — столько даёт один шум выборки.\n")
        store.append(dict(variant=vk, group="_flat", D=dist(np.zeros_like(base), base, v["dtw"], w)[0]))
        store.append(dict(variant=vk, group="_halves", D=np.median(half), D_hi=np.percentile(half, 95)))
    S = pd.DataFrame(store)
    PT = pd.concat(per_text, ignore_index=True)

    # сводка: сколько расстояния уходит без концовки
    L.append("## Сколько расстояния даёт концовка\n")
    rows = []
    for g in cfg["groups"]:
        x = S[S["group"] == g].set_index("variant")
        rows.append({"группа": LLM_RU[g], "D, все 20": fmt(*x.loc['all20', ['D', 'D_lo', 'D_hi']]),
                     "D, без 20": fmt(*x.loc['wo20', ['D', 'D_lo', 'D_hi']]),
                     "D, 1–19 DTW": fmt(*x.loc['dtw1_19', ['D', 'D_lo', 'D_hi']]),
                     "доля D² на позицию 20": f"{x.loc['all20', 'share_pos20']:.2f}",
                     "p без 20": f"{x.loc['wo20', 'p']:.4f}", "p 1–19 DTW": f"{x.loc['dtw1_19', 'p']:.4f}"})
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("D — с поправкой на шум, [95 % ДИ]. Доля D² на позицию 20 — вклад последней позиции в квадрат расстояния "
             "средней дуги группы до базы (вариант «все 20»; с поправкой на шум по позициям); при равномерном вкладе было "
             "бы 1/20 = 0,05. Варианты «все 20» и «без 20» нормированы по-разному (z по 20 и по 19 позициям) — сравнивать "
             "их D напрямую можно лишь приближённо; доля D² отвечает на вопрос точнее.\n")

    # длина внутри коротких форм
    L.append("## Растёт ли расстояние с длиной (короткие формы той же длины, что сказки)\n")
    L.append("Расстояние текста до базы ~ форма + log(n_words), МНК; ДИ наклона — бутстрэп по текстам (≤ 1 текста на "
             "автора). Расстояние отдельного текста в основном шум (SNR дуги ≈ 2, этап 4), поэтому важнее агрегатный "
             "D по терцилям длины (ниже).\n")
    rows, terc = [], []
    sf = cfg["short_forms"]
    for vk in cfg["variants"]:
        x = PT[(PT["variant"] == vk) & PT["group"].isin(sf)].reset_index(drop=True)
        x["lw"] = np.log(x["n_words"])
        f = smf.ols("d ~ C(group) + lw", data=x).fit()
        sl = []
        for _ in range(B):
            xb = x.iloc[rng.integers(0, len(x), len(x))]
            sl.append(smf.ols("d ~ C(group) + lw", data=xb).fit().params["lw"])
        rows.append({"вариант": VAR_RU[vk], "текстов": len(x), "наклон по log(n_words)": fmt(f.params["lw"], *np.percentile(sl, [2.5, 97.5])),
                     "на 2-кратный рост длины": f"{f.params['lw'] * np.log(2):+.3f}", "среднее расстояние текста": f"{x['d'].mean():.3f}"})
    L.append(md(pd.DataFrame(rows)) + "\n")
    # агрегатный D по терцилям длины (короткие формы вместе)
    d_ = docs.copy()
    sfm = d_["stratum"].isin(sf).to_numpy()
    q = np.quantile(np.log(d_.loc[sfm, "n_words"]), np.linspace(0, 1, cfg["length_bins"] + 1))
    rows = []
    for vk, v in cfg["variants"].items():
        Z = variant_Z(T, v)
        ok = ~np.isnan(Z).any(1)
        au = d_["author_id"].astype(str).to_numpy()
        mb = ok & (d_["stratum"] == cfg["base_group"]).to_numpy()
        bs_b, base = boot_means(Z[mb], au[mb], rng, B)
        lw = np.log(d_["n_words"].to_numpy())
        for k in range(cfg["length_bins"]):
            m_ = ok & sfm & (lw >= q[k]) & ((lw <= q[k + 1]) if k == cfg["length_bins"] - 1 else (lw < q[k + 1]))
            bs_g, mg = boot_means(Z[m_], au[m_], rng, B)
            D = dist(mg, base, v["dtw"], w)[0]
            Db = np.array([dist(bs_g[i], bs_b[i], v["dtw"], w)[0] for i in range(B)])
            lo_w, hi_w = np.exp(q[k]), np.exp(q[k + 1])
            Dc, l2, h2 = corr(D, Db)
            rows.append({"вариант": VAR_RU[vk], "терциль длины": f"{lo_w:.0f}–{hi_w:.0f} слов", "текстов": int(m_.sum()),
                         "D (с поправкой) [95 % ДИ]": fmt(Dc, l2, h2), "D сырое": f"{D:.3f}"})
            terc.append(dict(variant=vk, bin=k, lo=lo_w, hi=hi_w, D=Dc, D_lo=l2, D_hi=h2, D_raw=D))
    L.append("### D средней дуги до базы по терцилям длины (рассказ, новелла, фанфик вместе)\n")
    L.append(md(pd.DataFrame(rows)) + "\n")
    TR = pd.DataFrame(terc)

    # рисунки
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    ax = axes[0]
    xs = np.arange(1, 21)
    for g in cfg["groups"]:
        if g not in curves:
            continue
        diff, lo, hi = curves[g]
        ax.fill_between(xs, lo, hi, color=COL[g], alpha=0.12, lw=0)
        ax.plot(xs, diff, color=COL[g], lw=1.5, marker="o", ms=2.5, label=LLM_RU[g])
    ax.axhline(0, color="#777", lw=0.6)
    ax.set_xlabel("позиция (N = 20)")
    ax.set_ylabel("средняя z-дуга группы − база")
    ax.set_title("Где расстояние до базы: разница по позициям", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25)
    ax = axes[1]
    vks = list(cfg["variants"])
    width = 0.8 / len(cfg["groups"])
    for j, g in enumerate(cfg["groups"]):
        x = S[S["group"] == g].set_index("variant").loc[vks]
        pos = np.arange(len(vks)) + (j - (len(cfg["groups"]) - 1) / 2) * width
        ax.errorbar(pos, x["D"], yerr=[x["D"] - x["D_lo"], x["D_hi"] - x["D"]], fmt="o", ms=5, color=COL[g], lw=1.2,
                    capsize=2, label=LLM_RU[g])
    ax.set_xticks(range(len(vks)))
    ax.set_xticklabels([VAR_RU[v] for v in vks], fontsize=8)
    ax.set_ylabel("D до базы, с поправкой на шум (95 % ДИ)")
    ax.set_title("Расстояние до волшебной сказки, три варианта", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y")
    ax = axes[2]
    for i, vk in enumerate(vks):
        t = TR[TR["variant"] == vk]
        mid = np.sqrt(t["lo"] * t["hi"])
        ax.errorbar(mid * (1 + 0.03 * (i - 1)), t["D"], yerr=[t["D"] - t["D_lo"], t["D_hi"] - t["D"]], fmt="o-", ms=4,
                    lw=1.2, capsize=2, color=["#2a5fb0", "#8e44ad", "#16a085"][i], label=VAR_RU[vk])
    ax.set_xscale("log")
    ax.set_xlabel("длина, слов (терцили коротких форм)")
    ax.set_ylabel("D до базы, с поправкой на шум")
    ax.set_title("Расстояние и длина внутри коротких форм", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fp = figd / "06_distance_llm.png"
    fig.savefig(fp, dpi=120)
    plt.close(fig)
    L.append(f"![расстояние до базы](figures/{fp.name})\n")
    L += residual(cfg, docs, T, S, rng)
    S.to_parquet(out / "distance_llm.parquet", index=False)
    PT.to_parquet(out / "distance_llm_texts.parquet", index=False)
    TR.to_parquet(out / "distance_llm_length.parquet", index=False)
    (ROOT / "reports" / "06_distance.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def residual(cfg, docs, T, S, rng):
    """Решение Ворот 5/6, п. 3: чем объясняется остаточное расстояние без позиции 20 (z по позициям 1–19).
    (а) доля D² на позициях 1–2 (тёмное начало сказки) с поправкой на шум; (б) D только по позициям 3–19 и его
    перестановочный p; (в) TOST эквивалентности: порог δ = половина D рассказа в том же варианте (задан до расчёта),
    эквивалентность — если верхняя односторонняя 95 % граница D (с поправкой на смещение) < δ."""
    B, P, w = cfg["bootstrap"], cfg["n_perm"], cfg["dtw_window"]
    L = ["## Остаток без позиции 20: начало сказки? (Ворота 5/6, п. 3)\n",
         "z по позициям 1–19 (вариант «без позиции 20»). (а) Доля D² на позициях 1–2 — с поправкой на шум по позициям "
         "(квадрат разницы минус её бутстрэп-дисперсия); при равномерном вкладе было бы 2/19 = 0,11. (б) D по позициям 3–19 "
         "(без начала и без конца) и p перестановки меток. (в) TOST: порог эквивалентности δ = ½ · D рассказа "
         "(вариант «без позиции 20», с поправкой), задан до расчёта; эквивалентность — если верхняя односторонняя 95 % "
         "граница D группы < δ.\n"]
    Z = zscore(T[:, :19])
    ok = ~np.isnan(Z).any(1)
    d, Zv = docs[ok].reset_index(drop=True), Z[ok]
    au = d["author_id"].astype(str).to_numpy()
    mb = (d["stratum"] == cfg["base_group"]).to_numpy()
    bs_b, base = boot_means(Zv[mb], au[mb], rng, B)
    delta = 0.5 * float(S[(S["variant"] == "wo20") & (S["group"] == "story")]["D"].iloc[0])
    rows = []
    for g in cfg["groups"]:
        m_ = (d["stratum"] == g).to_numpy()
        bs_g, mg = boot_means(Zv[m_], au[m_], rng, B)
        diff = mg - base
        vb = np.var(bs_g, 0) + np.var(bs_b, 0)
        c2 = np.clip(diff ** 2 - vb, 0, None)
        sh = c2[:2].sum() / c2.sum() if c2.sum() > 0 else np.nan
        # бутстрэп доли (каждая копия — с той же поправкой)
        shb = []
        for k in range(B):
            dk = bs_g[k] - bs_b[k]
            ck = np.clip(dk ** 2 - vb, 0, None)
            shb.append(ck[:2].sum() / ck.sum() if ck.sum() > 0 else np.nan)
        shb = np.array(shb)
        # D по позициям 3–19 и перестановочный p
        D3 = np.sqrt((diff[2:] ** 2).sum())
        pool = np.vstack([Zv[mb], Zv[m_]])
        nb = int(mb.sum())
        D0 = np.empty(P)
        for r in range(P):
            ii = rng.permutation(len(pool))
            D0[r] = np.sqrt(((pool[ii[nb:]].mean(0) - pool[ii[:nb]].mean(0))[2:] ** 2).sum())
        Db3 = np.sqrt((((bs_g - bs_b)[:, 2:]) ** 2).sum(1))
        D3c, D3lo, D3hi = corr(D3, Db3)
        # TOST: верхняя односторонняя 95 % граница D (вариант «без позиции 20», все 19 позиций)
        D = np.sqrt((diff ** 2).sum())
        Db = np.sqrt(((bs_g - bs_b) ** 2).sum(1))
        up = float(np.sqrt(max(0.0, 2 * D ** 2 - np.percentile(Db ** 2, 5))))
        rows.append({"группа": LLM_RU[g], "доля D² на позициях 1–2": f"{sh:.2f} [{np.nanpercentile(shb, 2.5):.2f}; {np.nanpercentile(shb, 97.5):.2f}]",
                     "D по позициям 3–19 (с поправкой)": fmt(D3c, D3lo, D3hi), "p, 3–19": f"{(np.sum(D0 >= D3) + 1) / (P + 1):.4f}",
                     "D без позиции 20: верхняя 95 % граница": f"{up:.3f}",
                     f"эквивалентность (δ = {delta:.3f})": "да" if up < delta else "не показана"})
    L.append(md(pd.DataFrame(rows)) + "\n")
    return L


# ---------------------------------------------------------------- вторичный анализ: классификатор, полный корпус
def secondary(cfg, sample):
    sc = cfg["secondary"]
    ncfg = load_config(ROOT / cfg["nulls_config"])
    rng = np.random.default_rng(cfg["seed"])
    B, w = cfg["bootstrap"], cfg["dtw_window"]
    arcs = pd.read_parquet(ROOT / ncfg["out_dir"] / "full" / f"arcs_N{sc['N']}.parquet")
    arcs = arcs[arcs["within_author_cap"]].reset_index(drop=True)
    if sample:
        keep = (arcs["fk"] == "tale_ru") | (rng.random(len(arcs)) < sc["sample_frac"])
        arcs = arcs[keep].reset_index(drop=True)
    X = np.vstack(arcs[sc["signal"]].to_numpy()).astype(np.float64)
    L = [f"# Этап 6, вторичный анализ: классификатор, {'выборка ' + str(sc['sample_frac']) if sample else 'полный корпус'}\n",
         "**Оговорка.** База — классификаторная средняя дуга сказок Афанасьева; у классификатора систематическая позиционная "
         "ошибка: концовки сказок занижены (reports/02_position_error.md), и средняя дуга сказки по нему неотличима от "
         "перемешанной (этап 5, p = 0,975). Поэтому это расстояние до *классификаторного образа* сказки, а не до сказки; "
         "в выводы RQ2 — только как проверка поведения расстояния с длиной на полном диапазоне длин. Классификаторные дуги "
         "до LLM-базы не сравниваются (разные измерители).\n"]
    store = []
    for vk, v in cfg["variants"].items():
        Z = variant_Z(X, v)
        ok = ~np.isnan(Z).any(1)
        d, Zv = arcs[ok].reset_index(drop=True), Z[ok]
        au = d["author_id"].astype(str).to_numpy()
        mb = (d["fk"] == "tale_ru").to_numpy()
        bs_b, base = boot_means(Zv[mb], au[mb], rng, B)
        d["d"] = dist(Zv, base, v["dtw"], w)
        d["lw"] = np.log(d["n_words"])
        L.append(f"## Вариант «{VAR_RU[vk]}» ({len(d)} дуг)\n")
        rows = []
        for f in [f for f in ORDER if f != "tale_ru" and (d["fk"] == f).any()]:
            m_ = (d["fk"] == f).to_numpy()
            bs_g, mg = boot_means(Zv[m_], au[m_], rng, 300)
            D = dist(mg, base, v["dtw"], w)[0]
            Db = np.array([dist(bs_g[k], bs_b[k], v["dtw"], w)[0] for k in range(300)])
            Dc, l2, h2 = corr(D, Db)
            rows.append({"форма": FORM_RU[f], "текстов": int(m_.sum()), "D средней дуги до базы (с поправкой)": fmt(Dc, l2, h2),
                         "медиана длины, слов": int(d.loc[m_, "n_words"].median())})
            store.append(dict(variant=vk, form=f, D=Dc, D_lo=l2, D_hi=h2, D_raw=D))
        L.append(md(pd.DataFrame(rows)) + "\n")
        x = d[~mb].copy()
        x["form"] = x["fk"]
        res = []
        for name, fml in [("наивная: d ~ форма", "d ~ C(form, Treatment('story'))"),
                          ("с длиной: d ~ форма + log(n_words)", "d ~ C(form, Treatment('story')) + lw")]:
            f_ = smf.ols(fml, data=x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["author_id"])[0]})
            r = {"модель": name}
            for k_, val in f_.params.items():
                if k_.startswith("C(form"):
                    lab = FORM_RU[k_.split("[T.")[1].rstrip("]")]
                    r[lab] = f"{val:+.3f} ± {1.96 * f_.bse[k_]:.3f}"
            if "lw" in f_.params:
                r["log(n_words)"] = f"{f_.params['lw']:+.3f} ± {1.96 * f_.bse['lw']:.3f}"
            res.append(r)
        L.append("Расстояние текста до базы, МНК с кластерными по авторам ошибками (± 1,96·SE); эффекты формы — "
                 "относительно рассказа:\n")
        L.append(md(pd.DataFrame(res).fillna("")) + "\n")
    S = pd.DataFrame(store)
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    suf = "_sample" if sample else ""
    S.to_parquet(out / f"distance_secondary{suf}.parquet", index=False)
    (ROOT / "reports" / f"06_distance_secondary{suf}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--secondary", action="store_true")
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.secondary:
        secondary(cfg, a.sample)
    else:
        primary(cfg)


if __name__ == "__main__":
    main()
