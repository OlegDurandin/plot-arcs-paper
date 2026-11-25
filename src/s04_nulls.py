"""Этап 4. Нулевые модели, контроль длины и шум дуги (классификатор и RSL).

Шаги (каждый кэшируется в data/processed/s04[/sample]):
  arcs    — дуги N = 10 и 20 по каждому включённому тексту: clf = P(pos) − P(neg), rsl = (pos − neg) / слов.
  metrics — метрики формы z-дуги и перемешанная нулевая модель (n_perm перестановок сегментов на текст):
            lag1 — автокорреляция с лагом 1 (последовательная связность; у перемешанной дуги ≈ −1/(N−1));
            rough — средний |Δz| несглаженной z-дуги; turns — число поворотов сглаженной z-дуги;
            end_lift / beg_lift — среднее z концовки / начала (edge_frac сегментов) минус середина;
            peak_pos — положение максимума сглаженной z-дуги (доля текста); amp — std сырой дуги (до z);
            dist_base — евклидово расстояние сглаженной z-дуги до центроида базы (Афанасьев, base_magic600).
            Для каждой метрики (кроме amp, peak_pos) — z и p относительно перестановок того же текста.
  noise   — (выборка) бутстрэп предложений внутри сегментов N = 20: SNR = дисперсия дуги между сегментами /
            средняя бутстрэп-дисперсия сегмента. Классификатор — по предложениям (приближение сегментного сигнала
            средним по словам), RSL — точно (лексиконные счётчики предложений).
  report  — корреляции метрик с log(n_words) (все и внутри форм), интервалы длины, сопоставленные пары,
            MixedLM метрика ~ форма + log(n_words) + (1 | author) против наивной ~ форма; доля дуг, отличимых
            от перемешанных; SNR по формам и размеру сегмента.

Запуск:  .venv/bin/python src/s04_nulls.py configs/s04_nulls.yaml {arcs,metrics,noise,report} [--sample]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, in_sample, load_config  # noqa: E402
from s03_report import FORM_RU, form_key  # noqa: E402

METRICS = ["lag1", "rough", "turns", "end_lift", "beg_lift", "peak_pos", "amp", "dist_base"]
NULL_METRICS = ["lag1", "rough", "turns", "end_lift", "beg_lift", "dist_base"]
ORDER = ["tale_ru", "tale_world", "story", "novella", "fanfic", "short_series", "povest", "novel"]


def odir(cfg, sample):
    p = ROOT / cfg["out_dir"] / ("sample" if sample else "full")
    p.mkdir(parents=True, exist_ok=True)
    return p


def analysis(cfg):
    a = pd.read_parquet(ROOT / cfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(cfg["units"])].copy()
    a["fk"] = form_key(a)
    return a[a["fk"].notna()]


# ------------------------------------------------------------------ arcs
def cmd_arcs(cfg, sample):
    out = odir(cfg, sample)
    a = analysis(cfg)
    if sample:
        a = a[[in_sample(d, cfg["sample_frac"]) or fk == "tale_ru" for d, fk in zip(a["doc_id"], a["fk"])]]
    ids = list(a["doc_id"])
    for N in cfg["N"]:
        f = out / f"arcs_N{N}.parquet"
        if f.exists():
            continue
        c = pd.read_parquet(ROOT / cfg["clf_dir"] / f"clf_N{N}.parquet", columns=["doc_id", "seg", "sent_positive", "sent_negative", "n_words"],
                            filters=[("doc_id", "in", ids)])
        l_ = pd.read_parquet(ROOT / cfg["lex_dir"] / f"lex_N{N}.parquet", columns=["doc_id", "seg", "rsl_pos", "rsl_neg", "n_words"],
                             filters=[("doc_id", "in", ids)])
        d = c.merge(l_, on=["doc_id", "seg"], suffixes=("", "_lex"))
        assert (d["n_words"] == d["n_words_lex"]).all()
        d["clf"] = d["sent_positive"] - d["sent_negative"]
        d["rsl"] = (d["rsl_pos"] - d["rsl_neg"]) / d["n_words"].clip(lower=1)
        d = d.sort_values(["doc_id", "seg"])
        full = d.groupby("doc_id")["seg"].transform("size") == N
        d = d[full]
        g = d.groupby("doc_id", sort=True)
        arcs = pd.DataFrame({"doc_id": list(g.groups.keys())})
        for s_ in cfg["signals"]:
            arcs[s_] = list(d[s_].to_numpy(np.float32).reshape(-1, N))
        arcs["seg_words_median"] = g["n_words"].median().to_numpy()
        arcs = arcs.merge(a[["doc_id", "fk", "author_id", "n_words", "within_author_cap", "is_complete", "corpus", "unit"]], on="doc_id")
        arcs.to_parquet(f, index=False)
        print(f"N={N}: дуг {len(arcs)}", arcs["fk"].value_counts().to_dict())


# ------------------------------------------------------------------ metrics
def smooth_ma(Z, frac):
    """скользящее среднее по последней оси; окно — нечётное ≥ 3 из доли текста; края — по доступным точкам."""
    N = Z.shape[-1]
    w = max(3, int(round(frac * N)) | 1)
    h = w // 2
    out = np.empty_like(Z)
    for i in range(N):
        out[..., i] = Z[..., max(0, i - h):min(N, i + h + 1)].mean(-1)
    return out


def smooth_gauss(Z, sigma_frac):
    """гауссово ядро по позиции в долях текста (центры сегментов (i + 0,5) / N); на краях ядро перенормируется —
    ширина сглаживания одинакова в долях текста при любом N."""
    N = Z.shape[-1]
    x = (np.arange(N) + 0.5) / N
    K = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma_frac) ** 2)
    K = K / K.sum(1, keepdims=True)
    return Z @ K.T


def zscore(X):
    m = X.mean(-1, keepdims=True)
    s = X.std(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, (X - m) / s, np.nan)


def shape_metrics(Z, cfg, base_centroid):
    """Z: (..., N) z-дуги. Возвращает dict метрик формы (без amp)."""
    ac = cfg["arc"]
    N = Z.shape[-1]
    e = max(1, int(round(ac["edge_frac"] * N)))
    S = smooth_ma(Z, ac["ma_frac"])
    dz = np.diff(Z, axis=-1)
    lag1 = (Z[..., :-1] * Z[..., 1:]).mean(-1)          # z-дуга: среднее 0, дисперсия 1 → автокорреляция
    ds = np.diff(S, axis=-1)
    sg = np.sign(np.where(np.abs(ds) > ac["turn_min_delta"], ds, 0))
    turns = np.zeros(Z.shape[:-1])
    for i in range(sg.shape[-1] - 1):
        turns += (sg[..., i] * sg[..., i + 1] < 0)
    mid = Z[..., e:N - e].mean(-1)
    res = dict(lag1=lag1, rough=np.abs(dz).mean(-1), turns=turns, end_lift=Z[..., N - e:].mean(-1) - mid,
               beg_lift=Z[..., :e].mean(-1) - mid, peak_pos=np.argmax(np.nan_to_num(S, nan=-9), -1) / (N - 1),
               dist_base=np.sqrt(((S - base_centroid) ** 2).sum(-1)))
    return res


def cmd_metrics(cfg, sample):
    out = odir(cfg, sample)
    rng = np.random.default_rng(cfg["seed"])
    nc = cfg["perm_null"]
    for N in cfg["N"]:
        f = out / f"metrics_N{N}.parquet"
        if f.exists():
            continue
        arcs = pd.read_parquet(out / f"arcs_N{N}.parquet")
        a_base = analysis(cfg)
        base_ids = set(a_base.loc[a_base["base_magic600"].fillna(False).astype(bool), "doc_id"])
        res = arcs[["doc_id"]].copy()
        t0 = time.time()
        for s_ in cfg["signals"]:
            X = np.vstack(arcs[s_].to_numpy()).astype(np.float64)
            Z = zscore(X)
            bm = arcs["doc_id"].isin(base_ids).to_numpy() & ~np.isnan(Z).any(1)
            cen = np.nanmean(smooth_ma(Z[bm], cfg["arc"]["ma_frac"]), 0)
            obs = shape_metrics(Z, cfg, cen)
            res[f"{s_}_amp"] = X.std(1)
            for m in METRICS:
                if m in obs:
                    res[f"{s_}_{m}"] = obs[m]
            # перемешанная нулевая модель — порциями
            nz = {m: np.full(len(X), np.nan) for m in NULL_METRICS}
            npv = {m: np.full(len(X), np.nan) for m in NULL_METRICS}
            for i0 in range(0, len(X), nc["chunk_texts"]):
                Zc = Z[i0:i0 + nc["chunk_texts"]]
                perm = np.argsort(rng.random((len(Zc), nc["n_perm"], N)), axis=-1)
                P = np.take_along_axis(np.repeat(Zc[:, None, :], nc["n_perm"], 1), perm, -1)
                pm = shape_metrics(P, cfg, cen)
                for m in NULL_METRICS:
                    mu, sd = pm[m].mean(1), pm[m].std(1)
                    o = obs[m][i0:i0 + len(Zc)]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        nz[m][i0:i0 + len(Zc)] = np.where(sd > 0, (o - mu) / sd, np.nan)
                    # p (односторонний): доля перестановок с метрикой ≥ наблюдаемой (lag1, |lift|) или ≤ (rough, turns, dist)
                    if m in ("rough", "turns", "dist_base"):
                        npv[m][i0:i0 + len(Zc)] = ((pm[m] <= o[:, None]).sum(1) + 1) / (nc["n_perm"] + 1)
                    elif m in ("end_lift", "beg_lift"):
                        npv[m][i0:i0 + len(Zc)] = ((np.abs(pm[m]) >= np.abs(o)[:, None]).sum(1) + 1) / (nc["n_perm"] + 1)
                    else:
                        npv[m][i0:i0 + len(Zc)] = ((pm[m] >= o[:, None]).sum(1) + 1) / (nc["n_perm"] + 1)
            for m in NULL_METRICS:
                res[f"{s_}_{m}_nullz"] = nz[m]
                res[f"{s_}_{m}_p"] = npv[m]
            print(f"N={N} {s_}: {len(X)} дуг, {time.time() - t0:.0f} с", flush=True)
        res.to_parquet(f, index=False)


# ------------------------------------------------------------------ noise
def cmd_noise(cfg, sample):
    """SNR дуг N = 20 на стратифицированной выборке: бутстрэп предложений внутри сегментов."""
    out = odir(cfg, sample)
    f = out / "noise_N20.parquet"
    if f.exists():
        print("уже есть:", f)
        return
    nc = cfg["noise"]
    rng = np.random.default_rng(cfg["seed"])
    arcs = pd.read_parquet(out / "arcs_N20.parquet")
    pool = arcs[arcs["within_author_cap"]] if cfg["use_author_cap"] else arcs
    pick = pd.concat([g.sample(min(len(g), nc["per_form"].get(fk, 0)), random_state=cfg["seed"])
                      for fk, g in pool.groupby("fk") if nc["per_form"].get(fk, 0) > 0])
    ids = list(pick["doc_id"])
    ecfg = load_config(ROOT / cfg["emotion_config"])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s02_lexicon import build_lexicons, shard_features, _init
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _init(build_lexicons(ecfg))
    mc = ecfg["models"]["sentiment"]
    dev = nc["device"] if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mc["hf"], revision=mc.get("revision"))
    model = AutoModelForSequenceClassification.from_pretrained(mc["hf"], revision=mc.get("revision")).to(dev).eval()
    lab = [model.config.id2label[j] for j in range(model.config.num_labels)]
    ip, ineg = lab.index("positive"), lab.index("negative")
    seg = pd.read_parquet(ROOT / cfg["s01_run"] / "segments_rel_N20.parquet", filters=[("doc_id", "in", ids)])
    segd = {d: g.sort_values("seg") for d, g in seg.groupby("doc_id")}
    rows = []
    t0 = time.time()
    for fi, sh in enumerate(sorted(glob.glob(str(ROOT / cfg["s01_run"] / "sentences" / "*.parquet")))):
        s = pd.read_parquet(sh, columns=["doc_id", "sent_idx", "text", "n_words"], filters=[("doc_id", "in", ids)])
        if not len(s):
            continue
        lf = shard_features(sh)
        lf = lf[lf["doc_id"].isin(set(s["doc_id"]))][["doc_id", "sent_idx", "rsl_pos", "rsl_neg"]]
        s = s.merge(lf, on=["doc_id", "sent_idx"])
        texts = list(s["text"])
        order = np.argsort([len(t) for t in texts])
        sc = np.zeros(len(texts), np.float32)
        with torch.no_grad():
            for i in range(0, len(order), nc["clf_batch"]):
                idx = order[i:i + nc["clf_batch"]]
                b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=nc["clf_max_tokens"], return_tensors="pt").to(dev)
                p = torch.softmax(model(**b).logits, -1).cpu().numpy()
                sc[idx] = p[:, ip] - p[:, ineg]
        s["clf_s"] = sc
        for d, g in s.groupby("doc_id"):
            g = g.set_index("sent_idx").sort_index()
            clf_m, clf_v, rsl_m, rsl_v = [], [], [], []
            for r in segd[d].itertuples():
                x = g.loc[r.sent_start:r.sent_end - 1]
                w = x["n_words"].to_numpy(float)
                cs, rp, rn = x["clf_s"].to_numpy(float), x["rsl_pos"].to_numpy(float), x["rsl_neg"].to_numpy(float)
                k = len(x)
                bi = rng.integers(0, k, (nc["n_boot"], k))
                W = w[bi]
                with np.errstate(invalid="ignore", divide="ignore"):
                    bc = (cs[bi] * W).sum(1) / W.sum(1)
                    br = (rp[bi] - rn[bi]).sum(1) / W.sum(1)
                clf_m.append((cs * w).sum() / max(w.sum(), 1)); clf_v.append(np.nanvar(bc))
                rsl_m.append((rp - rn).sum() / max(w.sum(), 1)); rsl_v.append(np.nanvar(br))
            rows.append(dict(doc_id=d, clf_between=float(np.var(clf_m)), clf_noise=float(np.mean(clf_v)),
                             rsl_between=float(np.var(rsl_m)), rsl_noise=float(np.mean(rsl_v))))
        if fi % 25 == 0:
            print(f"[{time.time() - t0:6.0f}s] шард {fi + 1}: текстов {len(rows)}/{len(ids)}", flush=True)
    nz = pd.DataFrame(rows)
    for s_ in ["clf", "rsl"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            nz[f"{s_}_snr"] = nz[f"{s_}_between"] / nz[f"{s_}_noise"].replace(0, np.nan)
    nz = nz.merge(pick[["doc_id", "fk", "author_id", "n_words", "seg_words_median"]], on="doc_id")
    nz.to_parquet(f, index=False)
    print("готово:", f, len(nz))


# ------------------------------------------------------------------ report
def md(df):
    cols = list(df.columns)
    o = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        o.append("| " + " | ".join("" if (not isinstance(v, str) and pd.isna(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(o)


def cmd_report(cfg, sample):
    from scipy import stats
    import statsmodels.formula.api as smf
    out = odir(cfg, sample)
    lc = cfg["length"]
    tag = " (ВЫБОРКА)" if sample else ""
    L = [f"# Этап 4. Нулевые модели, контроль длины, шум дуги{tag}\n",
         "Сигналы: clf = P(pos) − P(neg) (seara/rubert-tiny2), rsl = RuSentiLex (pos − neg) / слов. Метрики — на z-дуге "
         f"(z внутри текста); сглаживание — скользящее среднее, окно {cfg['arc']['ma_frac']} длины текста (≥ 3 сегментов); "
         f"концовка/начало — {cfg['arc']['edge_frac']:.0%} сегментов. Нулевая модель — {cfg["perm_null"]['n_perm']} перестановок сегментов "
         "на текст. Сравнения форм — на текстах within_author_cap (≤ 20 на автора).\n"]
    for N in cfg["N"]:
        arcs = pd.read_parquet(out / f"arcs_N{N}.parquet").drop(columns=cfg["signals"])
        met = pd.read_parquet(out / f"metrics_N{N}.parquet")
        d = arcs.merge(met, on="doc_id")
        if cfg["use_author_cap"]:
            d = d[d["within_author_cap"]]
        d["lw"] = np.log(d["n_words"])
        d["lsw"] = np.log(d["seg_words_median"].clip(lower=1))
        forms = [f for f in ORDER if (d["fk"] == f).any()]
        L.append(f"## N = {N}: {len(d)} дуг — " + ", ".join(f"{FORM_RU[f]} {int((d['fk'] == f).sum())}" for f in forms) + "\n")

        # 1. перемешанная нулевая модель
        L.append("### Отличима ли дуга от перемешанной\n")
        L.append(f"Доля текстов с p < {cfg["perm_null"]['alpha']} относительно перестановок: lag1 (связность выше, чем у перемешанной), "
                 "rough (изрезанность ниже), turns (поворотов меньше); в скобках — медиана z относительно перестановок.\n")
        rows = []
        for s_ in cfg["signals"]:
            for m in ["lag1", "rough", "turns"]:
                r = {"сигнал": s_, "метрика": m}
                for f in forms:
                    x = d[d["fk"] == f]
                    r[FORM_RU[f]] = f"{(x[f'{s_}_{m}_p'] < cfg["perm_null"]['alpha']).mean():.2f} ({x[f'{s_}_{m}_nullz'].median():+.2f})"
                rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")
        L.append(f"Ожидаемая доля при отсутствии структуры ≈ {cfg["perm_null"]['alpha']} (плюс дискретность p при {cfg["perm_null"]['n_perm']} перестановках).\n")

        # 2. корреляция метрик с длиной
        L.append("### Корреляция метрик дуги с длиной (ρ Спирмена с log n_words)\n")
        rows = []
        for s_ in cfg["signals"]:
            for m in METRICS:
                c = f"{s_}_{m}"
                r = {"сигнал": s_, "метрика": m, "все": stats.spearmanr(d[c], d["lw"], nan_policy="omit").statistic,
                     "все: с log(слов в сегменте)": stats.spearmanr(d[c], d["lsw"], nan_policy="omit").statistic}
                for f in forms:
                    x = d[d["fk"] == f]
                    r[FORM_RU[f]] = stats.spearmanr(x[c], x["lw"], nan_policy="omit").statistic if len(x) > 30 else np.nan
                rows.append(r)
        L.append(md(pd.DataFrame(rows).round(3)) + "\n")

        # 3. наивная и скорректированная оценка формы (MixedLM)
        L.append("### Эффект формы относительно рассказа: наивно (~ форма) и с контролем длины (~ форма + log n_words + (1 | автор))\n")
        rows = []
        for s_ in cfg["signals"]:
            for m in ["lag1", "rough", "turns", "end_lift", "amp", "dist_base"]:
                c = f"{s_}_{m}"
                x = d[[c, "fk", "lw", "author_id"]].dropna().rename(columns={c: "y"})
                x["author_id"] = x["author_id"].astype(str)
                if sample:
                    pass
                naive = smf.ols("y ~ C(fk, Treatment('story'))", x).fit()
                try:
                    mm = smf.mixedlm("y ~ C(fk, Treatment('story')) + lw", x, groups=x["author_id"]).fit(reml=True)   # lbfgs в statsmodels 0.15 даёт вырожденные нули — проверено
                    ok = True
                except Exception:
                    ok = False
                r = {"сигнал": s_, "метрика": m}
                for f in forms:
                    if f == "story":
                        continue
                    nm = f"C(fk, Treatment('story'))[T.{f}]"
                    a_ = naive.params.get(nm, np.nan)
                    b_ = mm.params.get(nm, np.nan) if ok else np.nan
                    lo, hi = (mm.conf_int().loc[nm] if ok and nm in mm.params else (np.nan, np.nan))
                    r[FORM_RU[f]] = f"{a_:+.3f} → {b_:+.3f} [{lo:+.3f}; {hi:+.3f}]"
                rows.append(r)
        L.append("В ячейке: наивная оценка → с контролем длины [95 % ДИ].\n")
        L.append(md(pd.DataFrame(rows)) + "\n")

        # 4. внутри квантильных интервалов длины
        q = np.quantile(d["lw"], np.linspace(0, 1, lc["n_bins"] + 1))
        d["lbin"] = pd.cut(d["lw"], q, include_lowest=True, labels=[f"{int(np.exp(a))}–{int(np.exp(b))}" for a, b in zip(q[:-1], q[1:])])
        L.append(f"### Медианы метрик внутри квантильных интервалов длины (слов; формы с ≥ {lc['min_per_form_bin']} текстами)\n")
        rows = []
        for s_ in cfg["signals"]:
            for m in ["lag1", "rough", "end_lift", "dist_base"]:
                for b, x in d.groupby("lbin", observed=True):
                    r = {"сигнал": s_, "метрика": m, "длина": str(b)}
                    for f in forms:
                        y = x.loc[x["fk"] == f, f"{s_}_{m}"].dropna()
                        r[FORM_RU[f]] = f"{y.median():+.3f} ({len(y)})" if len(y) >= lc["min_per_form_bin"] else "—"
                    rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")

        # 5. сопоставленные пары
        L.append(f"### Сопоставленные по длине пары форм (ближайший сосед по log n_words, |Δ| ≤ {lc['caliper_log']}, без возвращения)\n")
        rows = []
        rng = np.random.default_rng(cfg["seed"])
        for fa, fb in lc["matched_pairs"]:
            A, Bp = d[d["fk"] == fa], d[d["fk"] == fb]
            if not len(A) or not len(Bp):
                continue
            bl = np.sort(Bp["lw"].to_numpy())
            bidx = np.argsort(Bp["lw"].to_numpy())
            used = np.zeros(len(bl), bool)
            pa, pb = [], []
            for i in rng.permutation(len(A)):
                v = A["lw"].iloc[i]
                j = np.searchsorted(bl, v)
                best = None
                for k in range(max(0, j - 50), min(len(bl), j + 50)):
                    if not used[k] and abs(bl[k] - v) <= lc["caliper_log"] and (best is None or abs(bl[k] - v) < abs(bl[best] - v)):
                        best = k
                if best is not None:
                    used[best] = True
                    pa.append(i); pb.append(bidx[best])
            r = {"пара": f"{FORM_RU[fa]} — {FORM_RU[fb]}", "пар": len(pa)}
            for s_ in cfg["signals"]:
                for m in ["lag1", "end_lift", "dist_base"]:
                    xa, xb = A[f"{s_}_{m}"].to_numpy()[pa], Bp[f"{s_}_{m}"].to_numpy()[pb]
                    dd = xa - xb
                    r[f"{s_} {m}: Δ"] = f"{np.nanmean(dd):+.3f} ± {1.96 * np.nanstd(dd) / np.sqrt(np.isfinite(dd).sum()):.3f}"
            rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")

    # 6. шум
    nf = out / "noise_N20.parquet"
    if nf.exists():
        nz = pd.read_parquet(nf)
        nz["segbin"] = pd.cut(nz["seg_words_median"], [0, 50, 100, 200, 500, 1000, 3000, 1e9],
                              labels=["≤50", "51–100", "101–200", "201–500", "501–1000", "1001–3000", ">3000"])
        L.append(f"## Шум дуги (N = 20, бутстрэп предложений, {cfg['noise']['n_boot']} повторов; {len(nz)} текстов)\n")
        L.append("SNR = дисперсия дуги между сегментами / средняя бутстрэп-дисперсия сегмента; SNR ≈ 1 — дуга не отличима "
                 "от шума выборки предложений. Классификатор здесь — по предложениям (приближение).\n")
        rows = []
        for s_ in ["clf", "rsl"]:
            r = {"сигнал": s_}
            for f in [f for f in ORDER if (nz["fk"] == f).any()]:
                x = nz.loc[nz["fk"] == f, f"{s_}_snr"]
                r[FORM_RU[f]] = f"{x.median():.1f} (> {cfg['noise']['snr_threshold']:g}: {(x > cfg['noise']['snr_threshold']).mean():.0%})"
            rows.append(r)
        L.append("Медиана SNR (доля текстов выше порога):\n")
        L.append(md(pd.DataFrame(rows)) + "\n")
        rows = []
        for s_ in ["clf", "rsl"]:
            for b, x in nz.groupby("segbin", observed=True):
                r = {"сигнал": s_, "слов в сегменте": str(b), "текстов": len(x), "медиана SNR": x[f"{s_}_snr"].median()}
                rows.append(r)
        L.append("SNR по размеру сегмента (все формы):\n")
        L.append(md(pd.DataFrame(rows)) + "\n")
        from scipy import stats as st_
        L.append("ρ(log SNR, log слов в сегменте): " + ", ".join(
            f"{s_} {st_.spearmanr(np.log(nz[f'{s_}_snr']), np.log(nz['seg_words_median']), nan_policy='omit').statistic:+.2f}" for s_ in ["clf", "rsl"]) + "\n")
    (ROOT / "reports" / f"04_nulls{'_sample' if sample else ''}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("cmd", choices=["arcs", "metrics", "noise", "report"])
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    {"arcs": cmd_arcs, "metrics": cmd_metrics, "noise": cmd_noise, "report": cmd_report}[args.cmd](cfg, args.sample)


if __name__ == "__main__":
    main()
