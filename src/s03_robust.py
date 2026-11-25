"""Этап 3, Ворота 3 (п. 5): устойчивость метрик тематической траектории.

На одной подвыборке (вся база Афанасьева + до 2 000 текстов каждой другой формы, N = 20):
  1. энтропия θ сегментов по формам (концентрированность тем) — внутри диапазонов размера сегмента;
  2. метрики траектории LDA (K = 50 и 100) в вариантах: все темы / без «дискурсивных» тем / без «сказочных» тем /
     без обоих (θ перенормируется на оставшиеся темы);
  3. канал на эмбеддингах (multilingual-e5-small): скорость, объём, извилистость траектории сегментов в пространстве
     эмбеддингов (косинусное расстояние);
  4. сравнение форм: медианы внутри диапазонов размера сегмента и регрессия метрика ~ форма + B-сплайн(log размера)
     с кластерными ошибками по авторам (эффект формы относительно рассказа).

Запуск:
    .venv/bin/python src/s03_robust.py configs/s03_robust.yaml embed    # эмбеддинги сегментов (кэш)
    .venv/bin/python src/s03_robust.py configs/s03_robust.yaml report   # метрики и отчёт
Выход: data/processed/s03/robust/*.parquet, reports/03_robustness.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s03_report import FORM_RU, form_key  # noqa: E402
from s03_topics import shortest_tour, traj_metrics  # noqa: E402


def subsample(cfg, tcfg) -> pd.DataFrame:
    out = ROOT / cfg["out_dir"]
    f = out / "subsample.parquet"
    if f.exists():
        return pd.read_parquet(f)
    a = pd.read_parquet(ROOT / tcfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(["text", "linked_work"])].copy()
    a["fk"] = form_key(a)
    tr = pd.read_parquet(ROOT / tcfg["out_dir"] / "full" / f"topic_traj_N{cfg['N']}_K{cfg['K'][0]}.parquet", columns=["doc_id"])
    a = a[a["doc_id"].isin(set(tr["doc_id"])) & a["fk"].notna()]
    parts = [a[a["fk"] == "tale_ru"]]
    for i, fk in enumerate(cfg["subsample"]["forms"]):
        p = a[a["fk"] == fk]
        parts.append(p.sample(min(len(p), cfg["subsample"]["n_per_form"]), random_state=cfg["seed"] + i))
    s = pd.concat(parts)[["doc_id", "fk", "author_id", "n_words"]].reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    s.to_parquet(f, index=False)
    return s


# ------------------------------------------------------------------ embeddings
def cmd_embed(cfg, tcfg, test=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    ec = cfg["embeddings"]
    out = ROOT / cfg["out_dir"]
    ef = out / ("emb_segments_test.parquet" if test else "emb_segments.parquet")
    if ef.exists():
        print("уже есть:", ef)
        return
    sub = subsample(cfg, tcfg)
    ids = set(sub["doc_id"])
    s01 = ROOT / tcfg["s01_run"]
    seg = pd.read_parquet(s01 / f"segments_rel_N{cfg['N']}.parquet")
    seg = seg[seg["doc_id"].isin(ids)]
    segd = {d: g.sort_values("seg") for d, g in seg.groupby("doc_id")}
    dev = ec["device"] if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ec["model"])
    model = AutoModel.from_pretrained(ec["model"]).to(dev).eval()
    rows = []
    rng = np.random.default_rng(cfg["seed"])
    files = sorted(glob.glob(str(s01 / "sentences" / "*.parquet")))
    if test:
        files = files[:3]
    import time
    t0 = time.time()
    for fi, f in enumerate(files):
        s = pd.read_parquet(f, columns=["doc_id", "sent_idx", "text", "n_words"], filters=[("doc_id", "in", list(ids))])
        if not len(s):
            continue
        chunks, owner, weight = [], [], []
        for d, g in s.groupby("doc_id"):
            g = g.set_index("sent_idx").sort_index()
            for r in segd[d].itertuples():
                x = g.loc[r.sent_start:r.sent_end - 1]
                tl = [len(t) for t in tok(list(x["text"]), add_special_tokens=False)["input_ids"]]
                cs = []
                buf, bt, bw = [], 0, 0
                for txt, t_, w_ in zip(x["text"], tl, x["n_words"]):
                    if buf and bt + t_ > ec["max_tokens"] - 8:
                        cs.append((" ".join(buf), bw))
                        buf, bt, bw = [], 0, 0
                    buf.append(txt); bt += t_; bw += w_
                if buf:
                    cs.append((" ".join(buf), bw))
                m = ec["max_chunks_per_segment"]
                if len(cs) > m:
                    cs = [cs[i] for i in np.linspace(0, len(cs) - 1, m).round().astype(int)]
                for txt, w_ in cs:
                    chunks.append(ec["prefix"] + txt); owner.append((d, int(r.seg))); weight.append(max(w_, 1))
        order = np.argsort([len(c) for c in chunks])
        E = np.zeros((len(chunks), model.config.hidden_size), np.float32)
        with torch.no_grad():
            for i in range(0, len(order), ec["batch_size"]):
                idx = order[i:i + ec["batch_size"]]
                b = tok([chunks[j] for j in idx], padding=True, truncation=True, max_length=ec["max_tokens"], return_tensors="pt").to(dev)
                h = model(**b).last_hidden_state
                m_ = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m_).sum(1) / m_.sum(1)
                E[idx] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        df = pd.DataFrame({"key": owner, "w": weight})
        df["i"] = np.arange(len(df))
        for (d, sg), g in df.groupby("key", sort=False):
            v = (E[g["i"].to_numpy()] * g["w"].to_numpy()[:, None]).sum(0)
            rows.append((d, sg, (v / np.linalg.norm(v)).astype(np.float32)))
        if fi % 20 == 0 or test:
            print(f"[{time.time() - t0:6.0f}s] шард {fi + 1}/{len(files)}: сегментов {len(rows)}", flush=True)
    em = pd.DataFrame(rows, columns=["doc_id", "seg", "emb"])
    em["emb"] = em["emb"].map(list)
    em.to_parquet(ef, index=False)
    print("готово:", ef, len(em))


# ------------------------------------------------------------------ metrics
def emb_metrics(X):
    n = len(X)
    D = np.clip(1 - X @ X.T, 0, None)
    steps = np.array([D[i, i + 1] for i in range(n - 1)])
    c = X.mean(0)
    c = c / np.linalg.norm(c)
    tour = shortest_tour(D)
    return dict(speed=float(steps.mean()), volume=float(np.clip(1 - X @ c, 0, None).mean()),
                circuitousness=float(steps.sum() / tour) if tour > 0 else np.nan)


def _job(args):
    kind, doc, X = args
    m = traj_metrics(X) if kind == "lda" else emb_metrics(X)
    m.pop("path_length", None)
    # внутритекстовый разброс: среднее попарное расстояние между сегментами текста (основная нормировка скорости,
    # решение Ворот 3; Хеллингер для LDA, 1 − cos для эмбеддингов)
    if kind == "lda":
        R_ = np.sqrt(np.clip(X, 0, None))
        D = np.sqrt(np.clip(1.0 - R_ @ R_.T, 0, None))
    else:
        D = np.clip(1 - X @ X.T, 0, None)
    iu = np.triu_indices(len(X), 1)
    m["pairwise"] = float(D[iu].mean())
    m["doc_id"] = doc
    return m


def entropy_norm(X):
    P = np.clip(X, 1e-12, None)
    return -(P * np.log(P)).sum(1) / np.log(X.shape[1])


def tale_topics(tcfg, K, ratio):
    t = json.loads((ROOT / tcfg["out_dir"] / "full" / f"topics_K{K}.json").read_text())
    ps, pc = pd.DataFrame(t["prevalence_by_stratum"]).T, pd.DataFrame(t["prevalence_by_corpus"]).T
    others = ps.loc[[s for s in ps.index if s != "tale"]].mean()
    tl = pc.loc[["afanasyev", "fairytales"]].max()
    return sorted(int(k) for k in tl.index if tl[k] > ratio * others[k]), t["top_words"]


def form_effects(df, metric, spline_df):
    """metric ~ C(fk, Treatment('story')) + bs(log size); кластерные ошибки по авторам."""
    import statsmodels.formula.api as smf
    d = df[[metric, "fk", "ls", "author_id"]].dropna().copy()
    d["author_id"] = d["author_id"].astype(str)
    m = smf.ols(f"{metric} ~ C(fk, Treatment('story')) + bs(ls, df={spline_df})", d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["author_id"])[0]})
    res = {}
    for fk in d["fk"].unique():
        if fk == "story":
            continue
        nm = f"C(fk, Treatment('story'))[T.{fk}]"
        lo, hi = m.conf_int().loc[nm]
        res[fk] = (m.params[nm], lo, hi)
    return res


def cmd_report(cfg, tcfg):
    out = ROOT / cfg["out_dir"]
    sub = subsample(cfg, tcfg)
    ids = list(sub["doc_id"])
    bins = [float(b) for b in cfg["size_bins"]]
    order = ["tale_ru", "tale_world", "story", "novella", "fanfic", "povest", "novel"]
    L = ["# Этап 3. Устойчивость метрик траектории (Ворота 3, п. 5)\n",
         "Подвыборка: " + ", ".join(f"{FORM_RU[f]} {int((sub['fk'] == f).sum())}" for f in order) +
         f" текстов; N = {cfg['N']}. Сравнение форм — (а) медианы внутри диапазона размера сегмента (содержательных лемм "
         f"на сегмент, медиана по тексту); (б) эффект формы относительно рассказа в регрессии метрика ~ форма + "
         f"B-сплайн(log размера, df = {cfg['spline_df']}), 95 % ДИ с кластерными ошибками по авторам.\n"]
    allm = {}
    size = None
    for K in cfg["K"]:
        th = pd.read_parquet(ROOT / tcfg["out_dir"] / "full" / f"theta_K{K}" / f"theta_N{cfg['N']}.parquet",
                             filters=[("doc_id", "in", ids)]).sort_values(["doc_id", "seg"])
        th = th[th.groupby("doc_id")["seg"].transform("size") == cfg["N"]]
        X = np.vstack(th["theta"].to_numpy()).astype(np.float64)
        docs = th["doc_id"].to_numpy()[::cfg["N"]]
        X3 = X.reshape(len(docs), cfg["N"], K)
        if size is None:   # размер сегмента — медиана содержательных лемм по сегментам текста (как в s03_report)
            size = pd.DataFrame({"doc_id": docs, "size": np.median(th["n_content"].to_numpy().reshape(-1, cfg["N"]), axis=1)})
        tt, tw = tale_topics(tcfg, K, cfg["tale_topic_ratio"])
        dt = cfg["discourse_topics"].get(K, [])
        # энтропия θ
        ent = pd.DataFrame({"doc_id": np.repeat(docs, cfg["N"]), "H": entropy_norm(X)}).groupby("doc_id")["H"].mean()
        allm[(K, "энтропия θ")] = ent
        variants = {"все темы": []}
        if dt:
            variants["без дискурсивных"] = dt
        variants["без сказочных"] = tt
        if dt:
            variants["без обоих"] = sorted(set(dt) | set(tt))
        L.append(f"## LDA K = {K}\n")
        L.append(f"Сказочные темы ({len(tt)}): " + "; ".join(f"{k} ({', '.join(tw[str(k)][:3])})" for k in tt) + ".  ")
        L.append(("Дискурсивные темы: " + "; ".join(f"{k} ({', '.join(tw[str(k)][:3])})" for k in dt) + ".\n") if dt
                 else "Дискурсивные темы для этого K не заданы.\n")
        share = pd.DataFrame({"doc_id": np.repeat(docs, cfg["N"]),
                              "tale": X[:, tt].sum(1), "disc": X[:, dt].sum(1) if dt else 0.0}).groupby("doc_id").mean()
        allm[(K, "доля сказочных тем")] = share["tale"]
        if dt:
            allm[(K, "доля дискурсивных тем")] = share["disc"]
        for vn, drop in variants.items():
            keep = [k for k in range(K) if k not in drop]
            Y = X3[:, :, keep]
            Y = Y / Y.sum(2, keepdims=True)
            if vn != "все темы":
                allm[(K, f"энтропия θ · {vn}")] = pd.Series(entropy_norm(Y.reshape(-1, len(keep))).reshape(len(docs), -1).mean(1), index=docs)
            with Pool(cfg["workers"]) as p:
                ms = pd.DataFrame(p.map(_job, [("lda", d, Y[i]) for i, d in enumerate(docs)], chunksize=64)).set_index("doc_id")
            for c in ["speed", "volume", "circuitousness", "pos_concentration"]:
                allm[(K, f"{c} · {vn}")] = ms[c]
            allm[(K, f"speed / volume · {vn}")] = ms["speed"] / ms["volume"].replace(0, np.nan)   # объём 0 — вырожденная θ
            allm[(K, f"speed / попарное · {vn}")] = ms["speed"] / ms["pairwise"].replace(0, np.nan)
            allm[(K, f"попарное · {vn}")] = ms["pairwise"]
    ef = out / "emb_segments.parquet"
    if ef.exists():
        em = pd.read_parquet(ef).sort_values(["doc_id", "seg"])
        em = em[em.groupby("doc_id")["seg"].transform("size") == cfg["N"]]
        E = np.vstack(em["emb"].to_numpy()).astype(np.float64)
        edocs = em["doc_id"].to_numpy()[::cfg["N"]]
        E3 = E.reshape(len(edocs), cfg["N"], -1)
        with Pool(cfg["workers"]) as p:
            ms = pd.DataFrame(p.map(_job, [("emb", d, E3[i]) for i, d in enumerate(edocs)], chunksize=64)).set_index("doc_id")
        for c in ["speed", "volume", "circuitousness"]:
            allm[("emb", f"{c} · эмбеддинги")] = ms[c]
        # нормировка масштаба: средняя косинусная дистанция между сегментами РАЗНЫХ текстов той же формы
        # (20 000 случайных пар); компактный «сказочный» регион пространства иначе уменьшает все расстояния
        fk_of = sub.set_index("doc_id")["fk"]
        rng = np.random.default_rng(cfg["seed"])
        edoc = np.repeat(edocs, cfg["N"])
        efk = fk_of.reindex(edoc).to_numpy()
        bt = {}
        for f in np.unique(efk):
            idx = np.where(efk == f)[0]
            a_, b_ = rng.choice(idx, 20000), rng.choice(idx, 20000)
            m_ = edoc[a_] != edoc[b_]
            bt[f] = float((1 - (E[a_[m_]] * E[b_[m_]]).sum(1)).mean())
        (out / "emb_between_text_dist.json").write_text(json.dumps(bt, ensure_ascii=False, indent=1))
        L.append("Межтекстовое косинусное расстояние сегментов внутри формы (масштаб нормировки): " +
                 ", ".join(f"{FORM_RU[f]} {bt[f]:.3f}" for f in order if f in bt) + ".\n")
        allm[("emb", "speed · эмбеддинги, норм. на межтекстовое")] = ms["speed"] / fk_of.reindex(ms.index).map(bt)
        allm[("emb", "speed / volume · эмбеддинги")] = ms["speed"] / ms["volume"].replace(0, np.nan)
        allm[("emb", "speed / попарное · эмбеддинги")] = ms["speed"] / ms["pairwise"].replace(0, np.nan)
        allm[("emb", "попарное · эмбеддинги")] = ms["pairwise"]
    M = pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in allm.items()})
    M = M.join(sub.set_index("doc_id")[["fk", "author_id"]]).join(size.set_index("doc_id"))
    M["ls"] = np.log(M["size"].clip(lower=1))
    M["bin"] = pd.cut(M["size"], bins, labels=[f"{int(a)}–{int(b)}" if b < 1e8 else f">{int(a)}" for a, b in zip(bins[:-1], bins[1:])])
    M.reset_index().rename(columns={"index": "doc_id"}).to_parquet(out / "metrics.parquet", index=False)

    def block(cols, title):
        L.append(f"### {title}\n")
        # (а) диапазоны, где есть Афанасьев
        tb = M[M["fk"] == "tale_ru"]["bin"].value_counts()
        use_bins = [b for b in tb.index if tb[b] >= 30]
        rows = []
        for c in cols:
            for b in sorted(use_bins, key=str):
                x = M[M["bin"] == b]
                r = {"метрика": c.split("|", 1)[1], "лемм в сегм.": str(b)}
                for f in order:
                    y = x.loc[x["fk"] == f, c].dropna()
                    r[FORM_RU[f]] = f"{y.median():.3f} ({len(y)})" if len(y) >= 30 else "—"
                rows.append(r)
        L.append(md(pd.DataFrame(rows)) + "\n")
        rows = []
        for c in cols:
            fe = form_effects(M.rename(columns={c: "m"}), "m", cfg["spline_df"])
            r = {"метрика": c.split("|", 1)[1], "рассказ (уровень)": f"{M.loc[M['fk'] == 'story', c].median():.3f}"}
            for f in order:
                if f in fe:
                    e, lo, hi = fe[f]
                    r[FORM_RU[f]] = f"{e:+.3f} [{lo:+.3f}; {hi:+.3f}]"
            rows.append(r)
        L.append("Эффект формы относительно рассказа (регрессия со сплайном по размеру сегмента):\n")
        L.append(md(pd.DataFrame(rows)) + "\n")

    for K in cfg["K"]:
        block([c for c in M.columns if c.startswith(f"{K}|") and ("энтропия" in c or "доля" in c)], f"K = {K}: энтропия θ и доли тем")
        block([c for c in M.columns if c.startswith(f"{K}|speed")], f"K = {K}: скорость — варианты состава тем")
        block([c for c in M.columns if c.startswith(f"{K}|pos_concentration")], f"K = {K}: позиционная концентрация — варианты")
        block([c for c in M.columns if c.startswith(f"{K}|volume") or c.startswith(f"{K}|circuitousness")],
              f"K = {K}: объём и извилистость — варианты")
    if ef.exists():
        block([c for c in M.columns if c.startswith("emb|")], "Эмбеддинги (multilingual-e5-small), косинусное расстояние")
    # ---------- итог для Ворот 3: три варианта, внутри диапазонов размера сегмента
    if ef.exists():
        K0 = cfg["K"][0]
        rows_ = [("энтропия θ, LDA все темы", f"{K0}|энтропия θ"), ("энтропия θ, LDA без сказочных тем", f"{K0}|энтропия θ · без сказочных"),
                 ("скорость, LDA все темы", f"{K0}|speed · все темы"), ("скорость, LDA без сказочных тем", f"{K0}|speed · без сказочных"),
                 ("скорость, эмбеддинги (сырая)", "emb|speed · эмбеддинги"),
                 ("**скорость / внутритекстовое попарное, LDA все темы** (основная)", f"{K0}|speed / попарное · все темы"),
                 ("**скорость / внутритекстовое попарное, LDA без сказочных тем**", f"{K0}|speed / попарное · без сказочных"),
                 ("**скорость / внутритекстовое попарное, эмбеддинги** (основная)", "emb|speed / попарное · эмбеддинги"),
                 ("скорость, эмбеддинги / межтекстовое расстояние формы (проверка)", "emb|speed · эмбеддинги, норм. на межтекстовое"),
                 ("знаменатель: внутритекстовое попарное, LDA", f"{K0}|попарное · все темы"),
                 ("знаменатель: внутритекстовое попарное, эмбеддинги", "emb|попарное · эмбеддинги")]
        forms_ = ["tale_ru", "tale_world", "story", "novella", "fanfic"]
        L.append(f"## Итог для Ворот 3 (K = {K0}): медианы внутри диапазонов размера сегмента; последний столбец — отклонение "
                 "Афанасьева от рассказа, %\n")
        L.append("Энтропия для эмбеддингов не определена (не распределение). Повести и романы не входят: их сегменты "
                 "(> 160 лемм) не пересекаются по размеру со сказками.\n")
        tr_ = []
        for lab, c in rows_:
            for b in ["0–20", "20–40", "40–80"]:
                x_ = M[M["bin"] == b]
                med = {f: x_.loc[x_["fk"] == f, c].median() for f in forms_}
                r = {"метрика": lab, "лемм в сегм.": b}
                for f in forms_:
                    r[FORM_RU[f]] = f"{med[f]:.3f}"
                r["Афанасьев vs рассказ"] = f"{100 * (med['tale_ru'] / med['story'] - 1):+.0f} %"
                tr_.append(r)
        L.append(md(pd.DataFrame(tr_)) + "\n")
    (ROOT / "reports" / "03_robustness.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def md(df):
    cols = list(df.columns)
    o = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        o.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("cmd", choices=["embed", "report"])
    ap.add_argument("--test", action="store_true", help="embed: только первые 3 шарда, отдельный файл")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config(ROOT / cfg["topics_config"])
    {"embed": lambda: cmd_embed(cfg, tcfg, args.test), "report": lambda: cmd_report(cfg, tcfg)}[args.cmd]()


if __name__ == "__main__":
    main()
