"""Этап 3. Тематический канал (Plot Arceology): LDA на сбалансированной выборке форм, темы сегментов, траектории.

Запуск (по шагам; каждый шаг кэшируется):
    .venv/bin/python src/s03_topics.py configs/s03_topics.yaml build  [--sample]
    .venv/bin/python src/s03_topics.py configs/s03_topics.yaml train  [--sample]
    .venv/bin/python src/s03_topics.py configs/s03_topics.yaml infer  [--sample] [--K 50]
    .venv/bin/python src/s03_topics.py configs/s03_topics.yaml traj   [--sample] [--K 50]
--sample: маленькая обучающая выборка, K из sample_K, вывод тем — на выборочном прогоне этапа 1.

Словарь: леммы NOUN/ADJF/ADJS/VERB/INFN, без PROPN, без лемм вне словаря pymorphy3 (выдуманные имена), без стоп-слов,
лемма — не меньше чем в min_texts текстах и не больше чем в max_doc_share обучающих документов.
Обучающий документ — 200 подряд идущих содержательных лемм одного текста; поровну документов на каждую группу форм.
Траектория текста (N сегментов): скорость — средняя Хеллингер-дистанция соседних сегментов; объём — средняя дистанция
до центроида текста; извилистость — длина пути / кратчайший обход тех же точек (эвристика коммивояжёра,
ближайший сосед + 2-opt; как circuitousness у Toubia et al. 2021); позиционная концентрация — Σ_k w_k (1 − H(p_k)/log N),
где p_k — распределение массы темы k по позициям, w_k — доля темы в тексте.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

_MORPH = None
CFG: dict = {}
VOCAB: set = set()


def _morph():
    global _MORPH
    if _MORPH is None:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


@lru_cache(maxsize=1_000_000)
def lemma_ok(lemma: str) -> bool:
    vc = CFG["vocab"]
    if len(lemma) < vc["min_lemma_len"] or lemma in STOP:
        return False
    if vc["drop_unknown_lemmas"] and not _morph().word_is_known(lemma):
        return False
    return True


STOP: set = set()


def content_tokens(lemmas: str, pos: str) -> list[str]:
    keep = POS_KEEP
    out = []
    for l_, p_ in zip(lemmas.split(), pos.split()):
        if p_ in keep and lemma_ok(l_):
            out.append(l_)
    return out


POS_KEEP: set = set()


def _setup(cfg):
    global CFG, STOP, POS_KEEP
    CFG = cfg
    STOP = set(cfg["vocab"]["stopwords"])
    POS_KEEP = set(cfg["vocab"]["pos_keep"])


def s01_dir(cfg, sample: bool) -> Path:
    if not sample:
        return ROOT / cfg["s01_run"]
    c = sorted(glob.glob(str(ROOT / "data/processed/s01_*_sample")), key=lambda p: Path(p).stat().st_mtime)
    return Path(c[-1])


def stratum_of(a: pd.DataFrame, strata: dict) -> pd.Series:
    st = pd.Series(None, index=a.index, dtype=object)
    for name, members in strata.items():
        for m in members:
            st[(a["corpus"] == m) | (a["form"] == m)] = name
    return st


# ------------------------------------------------------------------ build
def read_content(s01: Path, ids: set[str]) -> dict[str, list[list[str]]]:
    """doc_id -> список предложений (списки содержательных лемм), по порядку."""
    out: dict[str, list] = {}
    for f in sorted(glob.glob(str(s01 / "sentences" / "*.parquet"))):
        s = pd.read_parquet(f, columns=["doc_id", "sent_idx", "lemmas", "pos"], filters=[("doc_id", "in", list(ids))])
        if not len(s):
            continue
        s = s.sort_values(["doc_id", "sent_idx"])
        for doc, g in s.groupby("doc_id", sort=False):
            out[doc] = [content_tokens(l_, p_) for l_, p_ in zip(g["lemmas"], g["pos"])]
    return out


def cmd_build(cfg, sample: bool):
    tc = cfg["train"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / cfg["out_dir"] / ("sample" if sample else "full")
    out.mkdir(parents=True, exist_ok=True)
    per = tc["sample_chunks_per_stratum"] if sample else tc["chunks_per_stratum"]
    a = pd.read_parquet(ROOT / cfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(["text", "linked_work"])].copy()
    a["stratum"] = stratum_of(a, tc["strata"])
    a = a[a["stratum"].notna()]
    s01 = s01_dir(cfg, sample)
    have = set(pd.read_parquet(s01 / "docs.parquet", columns=["doc_id"])["doc_id"])
    a = a[a["doc_id"].isin(have)]
    # кандидаты: перемешать, ≤ max_texts_per_author на автора в группе; Афанасьев — все включённые
    cands = []
    for st, g in a.groupby("stratum"):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        if st == "tale":
            g = pd.concat([g[g["corpus"] == "afanasyev"], g[g["corpus"] != "afanasyev"]])   # Афанасьев — первым
        if st != "tale":   # у сказок author_id — сборник/собиратель (fairytalez) или сюжет (Афанасьев), не автор
            g = g.groupby("author_id", sort=False).head(tc["max_texts_per_author"])
        cands.append(g.head(max(per, 200) * 3))
    cands = pd.concat(cands)
    print("кандидатов текстов по группам:", cands.groupby("stratum").size().to_dict(), flush=True)
    content = read_content(s01, set(cands["doc_id"]))
    docs = []
    for st, g in cands.groupby("stratum", sort=False):
        n = 0
        for r in g.itertuples():
            toks = [t for sent in content.get(r.doc_id, []) for t in sent]
            ch = [toks[i:i + tc["chunk_tokens"]] for i in range(0, len(toks) - tc["chunk_tokens"] + 1, tc["chunk_tokens"])]
            if len(ch) > tc["max_chunks_per_text"]:
                idx = np.sort(rng.choice(len(ch), tc["max_chunks_per_text"], replace=False))
                ch = [ch[i] for i in idx]
            for k, c in enumerate(ch):
                docs.append(dict(stratum=st, corpus=r.corpus, doc_id=r.doc_id, chunk=k, tokens=" ".join(c)))
            n += len(ch)
            if n >= per:
                break
    d = pd.DataFrame(docs)
    d = d.groupby("stratum", group_keys=False).head(per)
    # словарь
    df_doc = Counter()
    df_text: dict[str, set] = {}
    for doc, toks in zip(d["doc_id"], d["tokens"]):
        u = set(toks.split())
        df_doc.update(u)
        for t in u:
            df_text.setdefault(t, set()).add(doc)
    vc = cfg["vocab"]
    vocab = sorted(t for t, c in df_doc.items()
                   if len(df_text[t]) >= vc["min_texts"] and c / len(d) <= vc["max_doc_share"])
    removed_frequent = sorted((t for t, c in df_doc.items() if c / len(d) > vc["max_doc_share"]), key=lambda t: -df_doc[t])
    d.to_parquet(out / "train_docs.parquet", index=False)
    (out / "vocab.json").write_text(json.dumps(dict(vocab=vocab, removed_frequent=removed_frequent), ensure_ascii=False))
    print("обучающих документов:", d.groupby(["stratum", "corpus"]).size().to_dict())
    print(f"словарь: {len(vocab)} лемм; удалены как слишком частые: {removed_frequent[:30]}")


# ------------------------------------------------------------------ train
def cmd_train(cfg, sample: bool):
    import tomotopy as tp
    tc = cfg["train"]
    out = ROOT / cfg["out_dir"] / ("sample" if sample else "full")
    md = ROOT / cfg["models_dir"]
    md.mkdir(exist_ok=True)
    d = pd.read_parquet(out / "train_docs.parquet")
    vocab = set(json.loads((out / "vocab.json").read_text())["vocab"])
    Ks = tc["sample_K"] if sample else tc["K"]
    iters = tc["sample_iterations"] if sample else tc["iterations"]
    for K in Ks:
        f = md / f"lda_{'sample_' if sample else ''}K{K}.bin"
        tj = out / f"topics_K{K}.json"
        if f.exists() and tj.exists() and "coherence_c_npmi" in json.loads(tj.read_text()):
            print("модель и темы уже есть:", f)
            continue
        if f.exists():
            mdl = tp.LDAModel.load(str(f))
            topic_summary(mdl, d, vocab, out, K, cfg)
            continue
        mdl = tp.LDAModel(k=K, alpha=tc["alpha"], eta=tc["eta"], seed=cfg["seed"])
        for toks in d["tokens"]:
            w = [t for t in toks.split() if t in vocab]
            if w:
                mdl.add_doc(w)
        mdl.burn_in = tc["burn_in"]
        mdl.optim_interval = tc["optim_interval"]
        t0 = time.time()
        for i in range(0, iters, 50):
            mdl.train(50, workers=tc["workers"])
            print(f"K={K} iter {i + 50}: log-likelihood/слово {mdl.ll_per_word:.4f} ({time.time() - t0:.0f} с)", flush=True)
        mdl.save(str(f))
        topic_summary(mdl, d, vocab, out, K, cfg)
        print("сохранено:", f)


def topic_summary(mdl, d, vocab, out, K, cfg):
    """Топ-слова, распространённость по группам/корпусам, когерентность (c_v и c_npmi) на подвыборке 6 000 документов
    (полный c_v в tomotopy однопоточный и занимает часы)."""
    import tomotopy as tp
    tops = {k: [w for w, _ in mdl.get_topic_words(k, top_n=15)] for k in range(K)}
    theta = np.array([doc.get_topic_dist() for doc in mdl.docs])
    keep = [i for i, toks in enumerate(d["tokens"]) if any(t in vocab for t in toks.split())]
    meta = d.iloc[keep].reset_index(drop=True)
    prev = pd.DataFrame(theta).groupby(meta["stratum"].to_numpy()).mean()
    prev_corp = pd.DataFrame(theta).groupby(meta["corpus"].to_numpy()).mean()
    rng = np.random.default_rng(cfg["seed"])
    sub = rng.choice(len(meta), min(6000, len(meta)), replace=False)
    corp = tp.utils.Corpus()
    for i in sub:
        corp.add_doc([t for t in meta["tokens"].iat[i].split() if t in vocab])
    coh = {}
    for name in ["c_v", "c_npmi"]:
        targets = sorted({w for k in range(K) for w in tops[k][:10]})
        c = tp.coherence.Coherence(corp, coherence=name, targets=targets)
        coh[name] = {k: float(c.get_score(words=tops[k][:10])) for k in range(K)}
    (out / f"topics_K{K}.json").write_text(json.dumps(dict(
        top_words=tops, ll_per_word=mdl.ll_per_word, n_docs=len(mdl.docs), vocab=len(mdl.used_vocabs),
        prevalence_by_stratum=prev.round(5).to_dict(orient="index"),
        prevalence_by_corpus=prev_corp.round(5).to_dict(orient="index"),
        coherence_c_v=coh["c_v"], coherence_c_npmi=coh["c_npmi"], coherence_reference_docs=len(sub)),
        ensure_ascii=False, indent=1))
    print(f"K={K}: темы записаны; средняя c_v {np.mean(list(coh['c_v'].values())):.3f}, c_npmi {np.mean(list(coh['c_npmi'].values())):.3f}", flush=True)


# ------------------------------------------------------------------ infer
_MODEL = None


def _init_infer(cfg, model_path, vocab):
    global _MODEL, VOCAB
    import tomotopy as tp
    _setup(cfg)
    _MODEL = tp.LDAModel.load(model_path)
    VOCAB = vocab


def infer_shard(args):
    path, out_dir, ns, s01, ids = args
    name = Path(path).stem
    if all((out_dir / f"N{n}" / f"{name}.parquet").exists() for n in ns):
        return name
    s = pd.read_parquet(path, columns=["doc_id", "sent_idx", "lemmas", "pos"], filters=[("doc_id", "in", list(ids))])
    if not len(s):
        for n in ns:
            (out_dir / f"N{n}").mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(out_dir / f"N{n}" / f"{name}.parquet")
        return name
    s = s.sort_values(["doc_id", "sent_idx"])
    toks = {doc: [[t for t in content_tokens(l_, p_) if t in VOCAB] for l_, p_ in zip(g["lemmas"], g["pos"])]
            for doc, g in s.groupby("doc_id", sort=False)}
    docs = list(toks)
    for n in ns:
        seg = pd.read_parquet(s01 / f"segments_rel_N{n}.parquet", filters=[("doc_id", "in", docs)])
        rows, mdocs = [], []
        for r in seg.itertuples():
            words = [w for sent in toks[r.doc_id][r.sent_start:r.sent_end] for w in sent]
            rows.append((r.doc_id, r.seg, len(words)))
            mdocs.append(_MODEL.make_doc(words if words else ["__empty__"]))
        theta, _ = _MODEL.infer(mdocs, iterations=CFG["infer"]["iterations"], workers=1, together=False)
        th = np.asarray(theta, dtype=np.float32).reshape(len(mdocs), -1)
        df = pd.DataFrame(rows, columns=["doc_id", "seg", "n_content"])
        df["theta"] = list(th)
        (out_dir / f"N{n}").mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"N{n}" / f"{name}.parquet", index=False)
    return name


def cmd_infer(cfg, sample: bool, K: int):
    tc = cfg["train"]
    out = ROOT / cfg["out_dir"] / ("sample" if sample else "full")
    model_path = str(ROOT / cfg["models_dir"] / f"lda_{'sample_' if sample else ''}K{K}.bin")
    vocab = set(json.loads((out / "vocab.json").read_text())["vocab"])
    s01 = s01_dir(cfg, sample)
    a = pd.read_parquet(ROOT / cfg["analysis_set"])
    ids = set(a.loc[a["unit"].isin(cfg["infer"]["units"]) & (a["include"] | a["include_sens400"]), "doc_id"])
    od = out / f"theta_K{K}"
    shards = sorted(glob.glob(str(s01 / "sentences" / "*.parquet")))
    tasks = [(p, od, cfg["infer"]["N"], s01, ids) for p in shards]
    t0 = time.time()
    with Pool(cfg["infer"]["workers"], initializer=_init_infer, initargs=(cfg, model_path, vocab)) as pool:
        for i, _ in enumerate(pool.imap_unordered(infer_shard, tasks)):
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(tasks)}", flush=True)
    for n in cfg["infer"]["N"]:
        parts = [pd.read_parquet(p) for p in sorted((od / f"N{n}").glob("shard-*.parquet"))]
        pd.concat([p for p in parts if len(p)], ignore_index=True).to_parquet(od / f"theta_N{n}.parquet", index=False)
    print("готово:", od)


# ------------------------------------------------------------------ trajectory
def hellinger_matrix(X: np.ndarray) -> np.ndarray:
    R = np.sqrt(np.clip(X, 0, None))
    G = R @ R.T
    return np.sqrt(np.clip(1.0 - G, 0, None))


def shortest_tour(D: np.ndarray) -> float:
    """Кратчайший открытый путь через все точки: ближайший сосед из каждой стартовой точки + 2-opt."""
    n = len(D)
    best = np.inf
    for s in range(n):
        path = [s]
        left = set(range(n)) - {s}
        while left:
            j = min(left, key=lambda k: D[path[-1], k])
            path.append(j)
            left.remove(j)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for k in range(i + 1, n):
                    a, b = path[i - 1], path[i]
                    c = path[k]
                    d = path[k + 1] if k + 1 < n else None
                    old = D[a, b] + (D[c, d] if d is not None else 0)
                    new = D[a, c] + (D[b, d] if d is not None else 0)
                    if new < old - 1e-12:
                        path[i:k + 1] = path[i:k + 1][::-1]
                        improved = True
        best = min(best, sum(D[path[i], path[i + 1]] for i in range(n - 1)))
    return float(best)


def traj_metrics(X: np.ndarray) -> dict:
    n = len(X)
    D = hellinger_matrix(X)
    steps = np.array([D[i, i + 1] for i in range(n - 1)])
    path = steps.sum()
    cen = X.mean(0)
    rc = np.sqrt(np.clip(1 - np.sqrt(np.clip(X, 0, None)) @ np.sqrt(np.clip(cen, 0, None)), 0, None))
    tour = shortest_tour(D)
    mass = X.sum(0)
    w = mass / mass.sum()
    P = X / np.clip(mass, 1e-12, None)
    H = -(np.where(P > 0, P * np.log(P), 0)).sum(0) / np.log(n)
    return dict(speed=float(steps.mean()), volume=float(rc.mean()), circuitousness=float(path / tour) if tour > 0 else np.nan,
                pos_concentration=float((w * (1 - H)).sum()), path_length=float(path))


def cmd_traj(cfg, sample: bool, K: int):
    out = ROOT / cfg["out_dir"] / ("sample" if sample else "full")
    for n in cfg["infer"]["N"]:
        th = pd.read_parquet(out / f"theta_K{K}" / f"theta_N{n}.parquet").sort_values(["doc_id", "seg"])
        rows = []
        for doc, g in th.groupby("doc_id", sort=False):
            if len(g) != n:
                continue
            X = np.vstack(g["theta"].to_numpy()).astype(np.float64)
            m = traj_metrics(X)
            m.update(doc_id=doc, n_content_min=int(g["n_content"].min()), n_content_median=float(g["n_content"].median()))
            rows.append(m)
        df = pd.DataFrame(rows)
        df.to_parquet(out / f"topic_traj_N{n}_K{K}.parquet", index=False)
        print(f"N={n}: траекторий {len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("cmd", choices=["build", "train", "infer", "traj"])
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--K", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    _setup(cfg)
    K = args.K or (cfg["train"]["sample_K"][0] if args.sample else cfg["train"]["K"][0])
    {"build": lambda: cmd_build(cfg, args.sample), "train": lambda: cmd_train(cfg, args.sample),
     "infer": lambda: cmd_infer(cfg, args.sample, K), "traj": lambda: cmd_traj(cfg, args.sample, K)}[args.cmd]()


if __name__ == "__main__":
    main()
