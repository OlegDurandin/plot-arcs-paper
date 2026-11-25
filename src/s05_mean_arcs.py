"""Этап 5. RQ1 на агрегатах: отличаются ли средние дуги форм друг от друга и от средней дуги перемешанных текстов.

Дуга — z внутри текста, без сглаживания (сглаживание — только на рисунках). Тексты within_author_cap.
  1. Средняя дуга формы: 95 % ДИ — пуассоновский бутстрэп по авторам (вес автора ~ Poisson(1)).
  2. Нулевая полоса: средняя дуга тех же текстов после перестановки сегментов внутри каждого текста (n_shuffle).
     Общий тест «дуга формы ≠ перемешанной»: SS = Σ_позиций (средняя дуга)², p — доля перестановок с SS ≥ наблюдаемого.
  3. Различия форм с контролем длины: пары текстов, сопоставленные по log(n_words) (|Δ| ≤ caliper, без возвращения);
     кривая средней парной разницы по позициям (ДИ — бутстрэп пар), расстояние D = √Σ(разница)²;
     p — перестановка знака внутри пар (n_perm). Проверки: веса по шуму (N20), N = 10, RSL.
  4. LLM-дуги (главный сигнал для сравнений со сказками): средние дуги групп выборки LLM-дуг, ДИ и нулевая полоса;
     различие «проза − Афанасьев» — D и перестановка меток групп.

Запуск:  .venv/bin/python src/s05_mean_arcs.py configs/s05_mean_arcs.yaml
Выход:   reports/05_mean_arcs.md, reports/figures/05_*.png, data/processed/s05/mean_arcs.parquet
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
from s03_report import FORM_RU  # noqa: E402
from s04_nulls import ORDER, md, smooth_gauss, zscore  # noqa: E402
import s02_validate as V  # noqa: E402

LLM_RU = {"afanasyev": "волшебная сказка (Афанасьев)", "fairytales": "сказка (fairytalez)", "story": "рассказ",
          "novella": "новелла", "fanfic": "фанфик", "povest": "повесть (отрывки)", "novel": "роман (отрывки)"}
COL = {"tale_ru": "#c0392b", "tale_world": "#e67e22", "story": "#2a5fb0", "novella": "#8e44ad", "fanfic": "#27ae60",
       "short_series": "#16a085", "povest": "#7f8c8d", "novel": "#2c3e50",
       "afanasyev": "#c0392b", "fairytales": "#e67e22"}


def author_sums(Z, authors, w=None):
    a, inv = np.unique(np.asarray(authors), return_inverse=True)
    w = np.ones(len(Z)) if w is None else np.asarray(w, float)
    S = np.zeros((len(a), Z.shape[1]))
    np.add.at(S, inv, Z * w[:, None])
    n = np.bincount(inv, weights=w, minlength=len(a))
    return S, n


def mean_ci(Z, authors, rng, B, w=None):
    S, n = author_sums(Z, authors, w)
    m = S.sum(0) / n.sum()
    W = rng.poisson(1.0, (B, len(n))).astype(float)
    bs = (W @ S) / (W @ n)[:, None]
    return m, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def shuffle_band(Z, rng, R):
    ms = np.empty((R, Z.shape[1]))
    for r in range(R):
        P = np.take_along_axis(Z, np.argsort(rng.random(Z.shape), 1), 1)
        ms[r] = P.mean(0)
    return ms


def match_pairs(la, lb, caliper, rng):
    order = np.argsort(lb)
    bl = lb[order]
    used = np.zeros(len(bl), bool)
    pa, pb = [], []
    for i in rng.permutation(len(la)):
        j = np.searchsorted(bl, la[i])
        best = None
        for k in range(max(0, j - 50), min(len(bl), j + 50)):
            if not used[k] and abs(bl[k] - la[i]) <= caliper and (best is None or abs(bl[k] - la[i]) < abs(bl[best] - la[i])):
                best = k
        if best is not None:
            used[best] = True
            pa.append(i); pb.append(order[best])
    return np.array(pa, int), np.array(pb, int)


def summarize(m):
    N = len(m)
    e = max(1, round(0.1 * N))
    mid = m[e:N - e].mean()
    return m[:e].mean() - mid, m[N - e:].mean() - mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    ncfg = load_config(ROOT / cfg["nulls_config"])
    rng = np.random.default_rng(cfg["seed"])
    B, R, P = cfg["bootstrap"], cfg["n_shuffle"], cfg["n_perm"]
    s4 = ROOT / ncfg["out_dir"] / "full"
    out = ROOT / "data/processed/s05"
    out.mkdir(parents=True, exist_ok=True)
    figd = ROOT / "reports" / "figures"
    L = ["# Этап 5. RQ1 — средние дуги форм против перемешанных и друг против друга\n",
         "Дуга — z внутри текста, без сглаживания (сглаживание σ = 0,05 длины — только на рисунках). Тексты within_author_cap "
         f"(≤ 20 на автора). ДИ — пуассоновский бутстрэп по авторам (B = {B}); нулевая полоса — {R} перестановок сегментов "
         f"внутри текстов; различия форм — пары, сопоставленные по log(n_words) (|Δ| ≤ {cfg['caliper_log']}), p — перестановка "
         f"знака внутри пар ({P}). Для сравнений со сказками главный сигнал — LLM-дуги (раздел 3); классификатор и RSL "
         "на полном корпусе — для прозы и как проверка.\n"]
    rows_store = []
    w20 = pd.read_parquet(s4 / "weights_N20.parquet", columns=["doc_id", "clf_w", "rsl_w"]).set_index("doc_id")

    for N in cfg["N"]:
        arcs = pd.read_parquet(s4 / f"arcs_N{N}.parquet")
        arcs = arcs[arcs["within_author_cap"]].reset_index(drop=True)
        arcs["lw"] = np.log(arcs["n_words"])
        forms = [f for f in ORDER if (arcs["fk"] == f).any()]
        for s_ in cfg["signals"]:
            X = np.vstack(arcs[s_].to_numpy()).astype(np.float64)
            Z = zscore(X)
            ok = ~np.isnan(Z).any(1)
            d, Z = arcs[ok].reset_index(drop=True), Z[ok]
            L.append(f"## N = {N}, сигнал {s_}: {len(d)} дуг\n")
            # 1–2. средние дуги и нулевая полоса
            rows, curves = [], {}
            for f in forms:
                m_ = (d["fk"] == f).to_numpy()
                Zf = Z[m_]
                mean, lo, hi = mean_ci(Zf, d.loc[m_, "author_id"].astype(str), rng, B)
                band = shuffle_band(Zf, rng, R)
                ss, ss0 = (mean ** 2).sum(), (band ** 2).sum(1)
                p = (np.sum(ss0 >= ss) + 1) / (R + 1)
                outside = int(((mean < np.percentile(band, 2.5, 0)) | (mean > np.percentile(band, 97.5, 0))).sum())
                beg, end = summarize(mean)
                bs_lo = summarize(lo)
                curves[f] = (mean, lo, hi, np.percentile(band, 2.5, 0), np.percentile(band, 97.5, 0))
                rows.append({"форма": FORM_RU[f], "текстов": int(m_.sum()), "SS средней дуги": f"{ss:.4f}",
                             "SS перемешанных, 95 %": f"{np.percentile(ss0, 95):.4f}", "p": f"{p:.3f}",
                             f"позиций вне полосы (из {N})": outside, "начало − середина": f"{beg:+.3f}", "конец − середина": f"{end:+.3f}"})
                for i in range(N):
                    rows_store.append(dict(N=N, signal=s_, form=f, pos=i + 1, mean=mean[i], lo=lo[i], hi=hi[i],
                                           null_lo=curves[f][3][i], null_hi=curves[f][4][i]))
            L.append("### Средняя дуга формы против перемешанных\n")
            L.append(md(pd.DataFrame(rows)) + "\n")
            if N == 20:
                # взвешенная по шуму средняя дуга — проверка
                ww = w20.reindex(d["doc_id"])[f"{s_}_w"].to_numpy()
                rows = []
                for f in forms:
                    m_ = (d["fk"] == f).to_numpy() & np.isfinite(ww)
                    mean_w, _, _ = mean_ci(Z[m_], d.loc[m_, "author_id"].astype(str), rng, 200, ww[m_])
                    b0, e0 = summarize(curves[f][0])
                    b1, e1 = summarize(mean_w)
                    rows.append({"форма": FORM_RU[f], "r(невзвеш., взвеш.)": f"{np.corrcoef(curves[f][0], mean_w)[0, 1]:.3f}",
                                 "начало: без весов → с весами": f"{b0:+.3f} → {b1:+.3f}", "конец: без весов → с весами": f"{e0:+.3f} → {e1:+.3f}"})
                L.append("Проверка: средние дуги, взвешенные по шуму (вес — по размеру сегмента, без формы):\n")
                L.append(md(pd.DataFrame(rows)) + "\n")
            # рисунок
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
            axes = axes.ravel()
            xs = np.arange(1, N + 1)
            for ax, f in zip(axes, forms):
                mean, lo, hi, n0, n1 = curves[f]
                ax.fill_between(xs, n0, n1, color="#999", alpha=0.25, lw=0, label="перемешанные, 95 %")
                ax.fill_between(xs, lo, hi, color=COL[f], alpha=0.25, lw=0)
                ax.plot(xs, mean, color=COL[f], lw=1.2, marker="o", ms=2.5, label="средняя дуга (95 % ДИ)")
                ax.plot(xs, smooth_gauss(mean[None, :], cfg["gauss_sigma_frac"])[0], color=COL[f], lw=2, alpha=0.6, label="сглажено (σ = 0,05)")
                ax.axhline(0, color="#777", lw=0.6)
                ax.set_title(f"{FORM_RU[f]} ({int((d['fk'] == f).sum())})", fontsize=10)
                ax.grid(alpha=0.25)
            for ax in axes[len(forms):]:
                ax.axis("off")
            axes[0].legend(fontsize=8, loc="lower left")
            for ax in (axes[0], axes[4]):
                ax.set_ylabel("средняя z-дуга")
            fig.suptitle(f"Средние дуги форм, N = {N}, {s_}", fontsize=11)
            fig.tight_layout()
            fp = figd / f"05_mean_arcs_N{N}_{s_}.png"
            fig.savefig(fp, dpi=120)
            plt.close(fig)
            L.append(f"![средние дуги](figures/{fp.name})\n")

            # 3. сопоставленные пары
            L.append("### Различия форм на сопоставленных по длине парах\n")
            rows = []
            fig, axes = plt.subplots(1, len(cfg["pairs"]), figsize=(2.4 * len(cfg["pairs"]), 3.2), sharey=True)
            for ax, (fa, fb) in zip(axes, cfg["pairs"]):
                A, Bm = (d["fk"] == fa).to_numpy(), (d["fk"] == fb).to_numpy()
                if not A.any() or not Bm.any():
                    ax.axis("off")
                    continue
                ia, ib = np.where(A)[0], np.where(Bm)[0]
                pa, pb = match_pairs(d["lw"].to_numpy()[ia], d["lw"].to_numpy()[ib], cfg["caliper_log"], rng)
                Dm = Z[ia[pa]] - Z[ib[pb]]
                diff = Dm.mean(0)
                D = np.sqrt((diff ** 2).sum())
                sg = rng.choice([-1.0, 1.0], (P, len(Dm)))
                D0 = np.sqrt((((sg @ Dm) / len(Dm)) ** 2).sum(1))
                p = (np.sum(D0 >= D) + 1) / (P + 1)
                bi = rng.integers(0, len(Dm), (B, len(Dm)))
                bdiff = np.stack([Dm[b].mean(0) for b in bi[:300]])
                lo, hi = np.percentile(bdiff, [2.5, 97.5], 0)
                nsig = int(((lo > 0) | (hi < 0)).sum())
                bA, eA = summarize(diff)
                rows.append({"пара (A − B)": f"{FORM_RU[fa]} − {FORM_RU[fb]}", "пар": len(Dm), "D": f"{D:.3f}",
                             "D перестановок, 95 %": f"{np.percentile(D0, 95):.3f}", "p": f"{p:.4f}",
                             f"позиций с ДИ ≠ 0 (из {N})": nsig, "начало A − B": f"{bA:+.3f}", "конец A − B": f"{eA:+.3f}"})
                ax.fill_between(xs, lo, hi, color="#2a5fb0", alpha=0.2, lw=0)
                ax.plot(xs, diff, color="#2a5fb0", lw=1.4)
                ax.axhline(0, color="#777", lw=0.6)
                ax.set_title(f"{FORM_RU[fa].split(' (')[0]} −\n{FORM_RU[fb].split(' (')[0]} ({len(Dm)})", fontsize=8)
                ax.grid(alpha=0.25)
            axes[0].set_ylabel("разница средних z-дуг")
            fig.suptitle(f"Разница дуг в парах, сопоставленных по длине (N = {N}, {s_}); полоса — 95 % ДИ", fontsize=10)
            fig.tight_layout()
            fp = figd / f"05_pair_diffs_N{N}_{s_}.png"
            fig.savefig(fp, dpi=120)
            plt.close(fig)
            L.append(md(pd.DataFrame(rows)) + "\n")
            L.append(f"![разницы пар](figures/{fp.name})\n")
            print(f"N={N} {s_} готово", flush=True)

    # 4. LLM-дуги
    acfg = load_config(ROOT / cfg["arcs_config"])
    ao = ROOT / acfg["out_dir"]
    k = pd.read_parquet(ao / "arcs_key.parquet", columns=["uid", "doc_id", "stratum", "author_id", "seg", "doc_n_words"])
    an = V.read_ann(ao / f"arcs__{acfg['models']['main']}.csv")
    x = k.merge(an, on="uid")
    x = x[x.groupby("doc_id")["skip"].transform("sum") == 0].sort_values(["doc_id", "seg"])
    docs = x.groupby("doc_id").agg(stratum=("stratum", "first"), author_id=("author_id", "first"))
    T = x["tone"].to_numpy(float).reshape(-1, 20)
    Zl = zscore(T)
    okl = ~np.isnan(Zl).any(1)
    docs, Zl = docs[okl], Zl[okl]
    L.append(f"## LLM-дуги (deepseek-v4.1-flash, N = 20): {len(docs)} текстов — главный сигнал для сравнений со сказками\n")
    L.append("Рассказы, новеллы, фанфики — в диапазоне длины сказок базы (643–3 179 слов), ≤ 1 текста на автора; повести и "
             "романы — каждый сегмент отрывком ≈ 350 слов. Текстов с нулевой дисперсией LLM-дуги (все сегменты одной "
             f"оценки) — {int((~okl).sum())}, исключены.\n")
    rows, curves = [], {}
    for g in cfg["llm_groups"]:
        m_ = (docs["stratum"] == g).to_numpy()
        mean, lo, hi = mean_ci(Zl[m_], docs.loc[m_, "author_id"].astype(str), rng, B)
        band = shuffle_band(Zl[m_], rng, R)
        ss, ss0 = (mean ** 2).sum(), (band ** 2).sum(1)
        p = (np.sum(ss0 >= ss) + 1) / (R + 1)
        beg, end = summarize(mean)
        curves[g] = (mean, lo, hi, np.percentile(band, 2.5, 0), np.percentile(band, 97.5, 0))
        rows.append({"группа": LLM_RU[g], "текстов": int(m_.sum()), "SS": f"{ss:.4f}", "SS перемешанных, 95 %": f"{np.percentile(ss0, 95):.4f}",
                     "p": f"{p:.3f}", "позиций вне полосы": int(((mean < curves[g][3]) | (mean > curves[g][4])).sum()),
                     "начало − середина": f"{beg:+.3f}", "конец − середина": f"{end:+.3f}"})
        for i in range(20):
            rows_store.append(dict(N=20, signal="llm", form=g, pos=i + 1, mean=mean[i], lo=lo[i], hi=hi[i],
                                   null_lo=curves[g][3][i], null_hi=curves[g][4][i]))
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("### Различие средних LLM-дуг: группа − Афанасьев (D = √Σ разница²; p — перестановка меток групп)\n")
    rows = []
    za = Zl[(docs["stratum"] == "afanasyev").to_numpy()]
    for g in cfg["llm_groups"][1:]:
        zg = Zl[(docs["stratum"] == g).to_numpy()]
        diff = zg.mean(0) - za.mean(0)
        D = np.sqrt((diff ** 2).sum())
        pool = np.vstack([za, zg])
        D0 = np.empty(P)
        for r in range(P):
            ii = rng.permutation(len(pool))
            D0[r] = np.sqrt(((pool[ii[len(za):]].mean(0) - pool[ii[:len(za)]].mean(0)) ** 2).sum())
        bA, eA = summarize(diff)
        rows.append({"группа − Афанасьев": LLM_RU[g], "D": f"{D:.3f}", "D перестановок, 95 %": f"{np.percentile(D0, 95):.3f}",
                     "p": f"{(np.sum(D0 >= D) + 1) / (P + 1):.4f}", "начало: разница": f"{bA:+.3f}", "конец: разница": f"{eA:+.3f}"})
    L.append(md(pd.DataFrame(rows)) + "\n")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    axes = axes.ravel()
    xs = np.arange(1, 21)
    for ax, g in zip(axes, cfg["llm_groups"]):
        mean, lo, hi, n0, n1 = curves[g]
        ax.fill_between(xs, n0, n1, color="#999", alpha=0.25, lw=0, label="перемешанные, 95 %")
        ax.fill_between(xs, lo, hi, color=COL[g], alpha=0.25, lw=0)
        ax.plot(xs, mean, color=COL[g], lw=1.2, marker="o", ms=2.5, label="средняя дуга (95 % ДИ)")
        ax.plot(xs, smooth_gauss(mean[None, :], cfg["gauss_sigma_frac"])[0], color=COL[g], lw=2, alpha=0.6)
        ax.axhline(0, color="#777", lw=0.6)
        ax.set_title(f"{LLM_RU[g]} ({int((docs['stratum'] == g).sum())})", fontsize=10)
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    axes[0].legend(fontsize=8, loc="lower left")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("средняя z-дуга (LLM)")
    fig.suptitle("Средние LLM-дуги, N = 20", fontsize=11)
    fig.tight_layout()
    fp = figd / "05_mean_arcs_llm.png"
    fig.savefig(fp, dpi=120)
    plt.close(fig)
    L.append(f"![LLM-дуги](figures/{fp.name})\n")
    pd.DataFrame(rows_store).to_parquet(out / "mean_arcs.parquet", index=False)
    (ROOT / "reports" / "05_mean_arcs.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
