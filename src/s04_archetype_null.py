"""Ворота 4, п. 2: создаёт ли «архетипы» само сглаживание? Тот же конвейер на настоящих и перемешанных дугах.

Конвейер (как в плане этапа 5): z-дуга внутри текста → сглаживание (скользящее среднее 15 % длины; контроль — без
сглаживания) → (1) SVD по всем текстам (within_author_cap): доля дисперсии мод и их форма; (2) PAA до 10 точек +
иерархическая кластеризация Уорда на стратифицированной подвыборке, k — по силуэту на настоящих дугах.
Перемешанные дуги: те же тексты, сегменты переставлены (n_shuffle_reps повторов). Сравнение: доля дисперсии мод,
корреляция форм мод и центроидов (сопоставление кластеров — венгерский алгоритм по корреляции), силуэт.

Запуск:  .venv/bin/python src/s04_archetype_null.py configs/s04_nulls.yaml
Выход:   reports/04_archetype_null.md, reports/figures/04_archetype_null_*.png
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
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s04_nulls import smooth_gauss, smooth_ma, zscore, md  # noqa: E402


SIGMA = 0.05


def prep(X, smoothing, frac):
    Z = zscore(X)
    if smoothing == "ma":
        return smooth_ma(Z, frac)
    if smoothing == "gauss":
        return smooth_gauss(Z, SIGMA)
    return Z


def paa(Z, m):
    N = Z.shape[1]
    if N == m:
        return Z
    return Z.reshape(len(Z), m, N // m).mean(2)


def svd_modes(Z, k):
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    V = Vt[:k] * np.sign(Vt[:k, [0]] + 1e-12)       # знак — по первой точке
    return var[:k], V, Z.mean(0)


def ward(P, k_range, k=None):
    L = linkage(P, "ward")
    sil = {}
    if k is None:
        for kk in range(k_range[0], k_range[1] + 1):
            lab = fcluster(L, kk, "maxclust")
            sil[kk] = silhouette_score(P, lab)
        k = max(sil, key=sil.get)
    lab = fcluster(L, k, "maxclust")
    cen = np.vstack([P[lab == c].mean(0) for c in np.unique(lab)])
    sizes = np.array([(lab == c).mean() for c in np.unique(lab)])
    return k, cen, sizes, silhouette_score(P, lab), sil


def match(A, B):
    C = np.corrcoef(A, B)[:len(A), len(A):]
    r, c = linear_sum_assignment(-C)
    return C[r, c], c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    an = cfg["archetype_null"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / cfg["out_dir"] / "full"
    frac = cfg["arc"]["ma_frac"]
    global SIGMA
    SIGMA = an.get("gauss_sigma_frac", SIGMA)
    R = ["# Архетипы на перемешанных дугах (Ворота 4, п. 2)\n",
         f"Конвейер: z-дуга → сглаживание (скользящее среднее {frac} длины; контроль — без сглаживания) → SVD (все тексты "
         f"within_author_cap) и PAA ({an['paa_points']} точек) + Уорд (до {an['ward_per_form']} текстов на форму, вся база; "
         f"k — по силуэту на настоящих дугах, k ∈ {an['k_range']}). Перемешанные — те же тексты, сегменты переставлены, "
         f"{an['n_shuffle_reps']} повторов (в таблицах — среднее [мин; макс] по повторам).\n"]
    figs = []
    for N in an["N"]:
        arcs = pd.read_parquet(out / f"arcs_N{N}.parquet")
        arcs = arcs[arcs["within_author_cap"]].reset_index(drop=True)
        ward_idx = np.concatenate([g.sample(min(len(g), an["ward_per_form"]), random_state=cfg["seed"]).index.to_numpy()
                                   for _, g in arcs.groupby("fk")])
        for s_ in an["signals"]:
            X = np.vstack(arcs[s_].to_numpy()).astype(np.float64)
            ok = X.std(1) > 0
            X = X[ok]
            widx = np.intersect1d(ward_idx, np.where(ok)[0])
            widx = np.searchsorted(np.where(ok)[0], widx)
            shuf = [np.take_along_axis(X, np.argsort(rng.random(X.shape), 1), 1) for _ in range(an["n_shuffle_reps"])]
            for sm in an["smoothing"]:
                Zr = prep(X, sm, frac)
                vr, Vr, mr = svd_modes(Zr, an["svd_modes"])
                vs, cs = [], []
                for Xs in shuf:
                    v_, V_, _ = svd_modes(prep(Xs, sm, frac), an["svd_modes"])
                    vs.append(v_)
                    cs.append([abs(np.corrcoef(Vr[i], V_[i])[0, 1]) for i in range(an["svd_modes"])])
                vs, cs = np.array(vs), np.array(cs)
                R.append(f"## N = {N}, {s_}, сглаживание: {dict(ma='скользящее среднее', gauss=f'гауссово ядро σ = {SIGMA} длины', none='нет')[sm]}\n")
                rows = []
                for i in range(an["svd_modes"]):
                    rows.append({"мода SVD": i + 1, "доля дисперсии: настоящие": f"{vr[i]:.3f}",
                                 "перемешанные": f"{vs[:, i].mean():.3f} [{vs[:, i].min():.3f}; {vs[:, i].max():.3f}]",
                                 "|r| формы моды (наст. vs перем.)": f"{cs[:, i].mean():.2f} [{cs[:, i].min():.2f}; {cs[:, i].max():.2f}]"})
                R.append(md(pd.DataFrame(rows)) + "\n")
                R.append(f"Сумма долей первых {an['svd_modes']} мод: настоящие {vr.sum():.3f}, перемешанные {vs.sum(1).mean():.3f}. "
                         f"Средняя дуга (по позициям) у настоящих: от {mr.min():+.2f} до {mr.max():+.2f} z.\n")
                Pr = paa(Zr[widx], an["paa_points"])
                k, cen_r, sz_r, sil_r, sil_all = ward(Pr, an["k_range"])
                rr, sils, cens = [], [], []
                for Xs in shuf:
                    Ps = paa(prep(Xs, sm, frac)[widx], an["paa_points"])
                    _, cen_s, sz_s, sil_s, _ = ward(Ps, an["k_range"], k)
                    r_, order = match(cen_r, cen_s)
                    rr.append(r_); sils.append(sil_s); cens.append(cen_s[order])
                rr = np.array(rr)
                R.append(f"Уорд: {len(widx)} текстов; k = {k} (силуэт по k: " + ", ".join(f"{kk}: {v:.3f}" for kk, v in sil_all.items()) + ").\n")
                rows = [{"кластер": i + 1, "доля (настоящие)": f"{sz_r[i]:.2f}",
                         "r центроидов наст. vs перем.": f"{rr[:, i].mean():.2f} [{rr[:, i].min():.2f}; {rr[:, i].max():.2f}]"} for i in range(k)]
                R.append(md(pd.DataFrame(rows)) + "\n")
                R.append(f"Силуэт: настоящие {sil_r:.3f}, перемешанные {np.mean(sils):.3f} [{np.min(sils):.3f}; {np.max(sils):.3f}]. "
                         f"Средний r сопоставленных центроидов: {rr.mean():.2f}.\n")
                kf = an.get("k_fixed")
                if kf and kf != k:
                    _, cen_f, sz_f, sil_f, _ = ward(Pr, an["k_range"], kf)
                    rf, sf = [], []
                    for Xs in shuf:
                        _, cs_f, _, sil_sf, _ = ward(paa(prep(Xs, sm, frac)[widx], an["paa_points"]), an["k_range"], kf)
                        r_, _ = match(cen_f, cs_f)
                        rf.append(r_); sf.append(sil_sf)
                    rf = np.array(rf)
                    R.append(f"Контроль k = {kf}: силуэт настоящие {sil_f:.3f}, перемешанные {np.mean(sf):.3f}; r сопоставленных "
                             f"центроидов по кластерам: " + ", ".join(f"{rf[:, i].mean():.2f}" for i in range(kf)) +
                             f" (среднее {rf.mean():.2f}); доли кластеров (настоящие): " + ", ".join(f"{v:.2f}" for v in sz_f) + ".\n")
                if sm in ("ma", "gauss"):
                    fig, axes = plt.subplots(2, k, figsize=(2.3 * k, 4.6), sharey=True)
                    axes = np.atleast_2d(axes)
                    xs = np.arange(1, an["paa_points"] + 1)
                    for i in range(k):
                        axes[0, i].plot(xs, cen_r[i], color="#2a5fb0", lw=2)
                        axes[0, i].set_title(f"{i + 1}: {sz_r[i]:.0%}", fontsize=9)
                        for cs_ in cens:
                            axes[1, i].plot(xs, cs_[i], color="#c0392b", lw=1, alpha=0.6)
                        axes[1, i].set_title(f"r = {rr[:, i].mean():.2f}", fontsize=9)
                        for ax in axes[:, i]:
                            ax.axhline(0, color="#aaa", lw=0.6); ax.grid(alpha=0.25); ax.set_xticks([1, 5, 10])
                    axes[0, 0].set_ylabel("настоящие\n(z, PAA)")
                    axes[1, 0].set_ylabel(f"перемешанные\n({an['n_shuffle_reps']} повт.)")
                    fig.suptitle(f"Центроиды Уорда: N = {N}, {s_}, {sm}", fontsize=10)
                    fig.tight_layout()
                    fp = ROOT / "reports" / "figures" / f"04_archetype_null_N{N}_{s_}_{sm}.png"
                    fig.savefig(fp, dpi=120)
                    plt.close(fig)
                    figs.append(fp.name)
                    R.append(f"![центроиды](figures/{fp.name})\n")
            print(f"N={N} {s_} готово", flush=True)
    (ROOT / "reports" / "04_archetype_null.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
