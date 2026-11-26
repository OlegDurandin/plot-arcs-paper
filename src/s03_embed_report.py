"""Ворота 5/6, п. 5: тематические таблицы на эмбеддингах полного корпуса (multilingual-e5-small, N = 20, без усечения).

По каждому тексту (20 сегментов, косинусное расстояние 1 − cos):
  speed      — среднее расстояние соседних сегментов (сырая скорость);
  pairwise   — среднее попарное расстояние сегментов текста (внутритекстовый разброс);
  ratio      — speed / pairwise: нормированная скорость (основная метрика, решение Ворот 3). При случайном порядке
               сегментов её ожидание = 1, поэтому ratio < 1 — последовательная связность (соседи ближе случайных пар);
  coh_p      — перестановочный p связности: доля из n_perm перестановок сегментов с средним соседним расстоянием ≤ наблюдаемого.
Таблицы — тексты within_author_cap; формы — внутри диапазонов размера сегмента (содержательных лемм, медиана по
тексту, как в 03_robustness.md); эффект формы — регрессия метрика ~ форма + B-сплайн(log размера), ошибки — кластерные
по авторам; сравнение с подвыборкой 03_robustness.md.

Запуск:  .venv/bin/python src/s03_embed_report.py configs/s03_robust.yaml [--sample]
Выход:   data/processed/s03/robust/emb_full_metrics[_sample].parquet, reports/03_embeddings_full[_sample].md
"""
from __future__ import annotations

import argparse
import glob
import sys
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s03_report import FORM_RU  # noqa: E402
from s03_robust import form_effects, md  # noqa: E402

N_PERM = 100
SEED = 42


def shard_metrics(path, n_perm):
    d = pd.read_parquet(path, columns=["doc_id", "seg", "emb"]).sort_values(["doc_id", "seg"])
    sz = d.groupby("doc_id")["seg"].transform("size")
    d = d[sz == 20]
    rng = np.random.default_rng([SEED, zlib.crc32(Path(path).name.encode())])   # воспроизводимо по шарду
    E = np.vstack(d["emb"].to_numpy()).astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    docs = d["doc_id"].to_numpy()[::20]
    E3 = E.reshape(len(docs), 20, -1)
    iu = np.triu_indices(20, 1)
    P = np.argsort(rng.random((n_perm, 20)), 1)
    rows = []
    for i, doc in enumerate(docs):
        X = E3[i]
        D = np.clip(1 - X @ X.T, 0, None)
        speed = float(np.diagonal(D, 1).mean())
        pair = float(D[iu].mean())
        null = D[P[:, :-1], P[:, 1:]].mean(1)
        rows.append((doc, speed, pair, speed / pair if pair > 0 else np.nan, float((np.sum(null <= speed) + 1) / (n_perm + 1))))
    return pd.DataFrame(rows, columns=["doc_id", "speed", "pairwise", "ratio", "coh_p"])


def main():
    global N_PERM
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tcfg = load_config(ROOT / cfg["topics_config"])
    rc = cfg["embeddings_full_report"]
    N_PERM = rc["n_perm"]
    out = ROOT / cfg["out_dir"]
    suf = "_sample" if a.sample else ""
    mf = out / f"emb_full_metrics{suf}.parquet"
    if mf.exists():
        M = pd.read_parquet(mf)
        print("метрики уже есть:", mf)
    else:
        fs = sorted(glob.glob(str(out / "emb_full" / "shard-*.parquet")))
        if a.sample:
            fs = fs[::max(1, len(fs) // rc["sample_shards"])]       # равномерно по шардам (они упорядочены по корпусам)
        with Pool(cfg["workers"]) as p:
            parts = p.map(partial(shard_metrics, n_perm=N_PERM), fs, chunksize=1)
        M = pd.concat(parts, ignore_index=True)
        M.to_parquet(mf, index=False)
    ncfg = load_config(ROOT / "configs/s04_nulls.yaml")
    meta = pd.read_parquet(ROOT / ncfg["out_dir"] / "full" / "arcs_N20.parquet",
                           columns=["doc_id", "fk", "author_id", "within_author_cap", "n_words"])
    size = (pd.read_parquet(ROOT / tcfg["out_dir"] / "full" / "theta_K50" / "theta_N20.parquet", columns=["doc_id", "n_content"])
            .groupby("doc_id")["n_content"].median().rename("size"))
    X = M.merge(meta, on="doc_id").join(size, on="doc_id")
    n_all = len(X)
    X = X[X["within_author_cap"]].reset_index(drop=True)
    bins = [float(b) for b in cfg["size_bins"]]
    labs = [f"{int(a_)}–{int(b_)}" if b_ < 1e8 else f">{int(a_)}" for a_, b_ in zip(bins[:-1], bins[1:])]
    X["bin"] = pd.cut(X["size"], bins, labels=labs)
    X["ls"] = np.log(X["size"].clip(lower=1))
    forms = [f for f in rc["forms"] if (X["fk"] == f).any()]
    L = [f"# Тематический канал на эмбеддингах {'(выборка ' + str(rc['sample_shards']) + ' шардов)' if a.sample else 'полного корпуса'}\n",
         "Решение Ворот 5/6, п. 5 (2026-11-26). multilingual-e5-small, сегменты N = 20 целиком (без усечения: куски по "
         "предложениям, длинные предложения — окнами по токенам), косинусное расстояние. Тексты within_author_cap "
         f"({len(X)} из {n_all} с метриками). Размер сегмента — содержательных лемм, медиана по тексту.\n",
         "Метрики: **сырая скорость** (среднее расстояние соседних сегментов); **внутритекстовое попарное** (средний "
         "разброс сегментов текста); **нормированная скорость** = скорость / попарное — основная (Ворота 3); ожидание при "
         "случайном порядке сегментов = 1, поэтому значение < 1 — **последовательная связность**. Связность текста "
         f"проверяется перестановкой сегментов ({N_PERM} перестановок, p ≤ {rc['alpha']}).\n",
         "Состав: " + ", ".join(f"{FORM_RU[f]} {int((X['fk'] == f).sum())}" for f in forms) + ".\n"]

    def by_bin(col, title, fmt="{:.3f}"):
        L.append(f"### {title}: медианы по формам внутри диапазонов размера сегмента (в скобках — текстов; «—» < 30)\n")
        rows = []
        for b in labs:
            x = X[X["bin"] == b]
            if not len(x):
                continue
            r = {"лемм в сегменте": b}
            for f in forms:
                y = x.loc[x["fk"] == f, col].dropna()
                r[FORM_RU[f]] = f"{fmt.format(y.median())} ({len(y)})" if len(y) >= 30 else "—"
            rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")

    for col, title in [("ratio", "Нормированная скорость (скорость / попарное)"), ("speed", "Сырая скорость"),
                       ("pairwise", "Внутритекстовое попарное расстояние")]:
        by_bin(col, title)
    # регрессия
    L.append("### Эффект формы относительно рассказа: метрика ~ форма + B-сплайн(log размера сегмента), 95 % ДИ "
             "(кластерные по авторам)\n")
    rows = []
    for col, nm in [("ratio", "нормированная скорость"), ("speed", "сырая скорость"), ("pairwise", "попарное")]:
        fe = form_effects(X.rename(columns={col: "m"}), "m", cfg["spline_df"])
        r = {"метрика": nm, "рассказ (медиана)": f"{X.loc[X['fk'] == 'story', col].median():.3f}"}
        for f in forms:
            if f in fe:
                e, lo, hi = fe[f]
                r[FORM_RU[f]] = f"{e:+.3f} [{lo:+.3f}; {hi:+.3f}]"
        rows.append(r)
    L.append(md(pd.DataFrame(rows)) + "\n")
    # связность
    X["coh"] = X["coh_p"] <= rc["alpha"]
    L.append(f"### Последовательная связность: доля текстов с нормированной скоростью < 1 и доля связных по перестановке "
             f"(p ≤ {rc['alpha']}; при отсутствии связности ожидается ≈ {rc['alpha']:.0%})\n")
    rows = []
    for b in labs:
        x = X[X["bin"] == b]
        if not len(x):
            continue
        r = {"лемм в сегменте": b}
        for f in forms:
            y = x[x["fk"] == f]
            r[FORM_RU[f]] = f"{(y['ratio'] < 1).mean():.0%} / {y['coh'].mean():.0%}" if len(y) >= 30 else "—"
        rows.append(r)
    r = {"лемм в сегменте": "все"}
    for f in forms:
        y = X[X["fk"] == f]
        r[FORM_RU[f]] = f"{(y['ratio'] < 1).mean():.0%} / {y['coh'].mean():.0%}"
    rows.append(r)
    L.append(md(pd.DataFrame(rows)) + "\n")
    # сверка с подвыборкой
    sp = ROOT / cfg["out_dir"] / "metrics.parquet"
    if sp.exists():
        S = pd.read_parquet(sp)
        L.append("### Сверка с подвыборкой (03_robustness.md): нормированная скорость, Афанасьев против рассказа\n")
        rows = []
        for b in ["0–20", "20–40", "40–80"]:
            s_ = S[S["bin"].astype(str) == b]
            x = X[X["bin"] == b]
            r = {"лемм в сегменте": b}
            for nm, dd, col in [("подвыборка", s_, "emb|speed / попарное · эмбеддинги"), ("полный корпус", x, "ratio")]:
                t_ = dd.loc[dd["fk"] == "tale_ru", col].dropna()
                st = dd.loc[dd["fk"] == "story", col].dropna()
                r[f"{nm}: Афанасьев"] = f"{t_.median():.3f} ({len(t_)})"
                r[f"{nm}: рассказ"] = f"{st.median():.3f} ({len(st)})"
                r[f"{nm}: разница"] = f"{100 * (t_.median() / st.median() - 1):+.1f} %"
            rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")
    (ROOT / "reports" / f"03_embeddings_full{suf}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
