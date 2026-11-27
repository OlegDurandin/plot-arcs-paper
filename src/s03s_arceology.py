"""Этап 3s. Plot Arceology на двух корпусах: субтитры сериалов (Ororo, русские субтитры) и проза + фанфики.

Отдельная LDA на каждом корпусе (K из конфига, в т. ч. 127 — как у Schmidt 2014), позиционные профили тем
(где в тексте тема концентрируется), классы формы профиля, нулевая модель — перестановка сегментов внутри текста,
сопоставление тем двух корпусов по словам.

Шаги (каждый кэшируется; --sample — выборочный прогон в отдельной папке):
  subs_ingest   русские .srt из архива (потоком, без распаковки) → чистка титров/тегов → леммы реплик (как этап 1)
  build         обучающие документы: субтитры — куски по 200 содержательных лемм (≤ 3 эпизода на сериал);
                проза + фанфики — куски этапа 3 для групп story/novella/povest/novel/fanfic; словари — отдельно
  train         LDA (tomotopy) на каждом корпусе для каждого K
  infer         θ сегментов: субтитры — N частей по словам (границы реплик) и N_time частей экранного времени;
                проза — N частей этапа 1 (по предложениям); ≤ 10 эпизодов на сериал, ≤ 1 текста на автора
  profiles      профили тем, нулевая модель, классы, сопоставление тем, рисунок, отчёт

Запуск: .venv/bin/python src/s03s_arceology.py configs/s03s_arceology.yaml STEP [--sample]
Выход:  reports/03s_arceology[_sample].md, reports/figures/03s_*.png, data/processed/s03s/{sample,full}/*
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
import s03_topics as T  # noqa: E402
from s01_segment import relative_segments  # noqa: E402

TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
TAG_RE = re.compile(r"<[^>]*>|\{[^}]*\}|\[[^\]]*\]")
SDH_RE = re.compile(r"\((?:[^()\s]+\s*){1,4}\)")
# v2 (2026-11-27): ремарки с пробелами у скобок «( смеётся )», в звёздочках «*вздыхает*» и заглавными «ОНА ВЗДЫХАЕТ»
SDH2_RE = re.compile(r"\(\s*(?:[^()\s]+\s*){1,6}\)|\*[^*\n]{1,60}\*")
CAPS_RE = re.compile(r"(?<![А-Яа-яЁёA-Za-z])[А-ЯЁ][А-ЯЁ-]+(?:[\s,.:!]+[А-ЯЁ][А-ЯЁ-]+)*")
LETTER = re.compile(r"[^\W\d_]")
CYR = re.compile(r"[А-Яа-яЁё]")
CFG: dict = {}
_MORPH = None
PROPER: set = set()


def out_dir(cfg, sample):
    return ROOT / cfg["out_dir"] / ("sample" if sample else "full")


# ------------------------------------------------------------------ субтитры: разбор
def _caps(m):
    """Заглавный отрезок — ремарка, если в нём ≥ 2 слов или слово ≥ 4 букв (аббревиатуры США, ФБР остаются)."""
    w = re.findall(r"[А-ЯЁ-]+", m.group(0))
    return " " if len(w) >= 2 or max(len(x) for x in w) >= 4 else m.group(0)


def parse_srt(raw: bytes, sdh_v2: bool = False):
    for enc in ("utf-8-sig", "cp1251", "utf-16"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return []
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", txt.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        m, k = None, None
        for i, ln in enumerate(lines[:3]):
            m = TIME_RE.search(ln)
            if m:
                k = i
                break
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        t0 = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        t1 = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        text = " ".join(TAG_RE.sub("", ln) for ln in lines[k + 1:])
        text = SDH_RE.sub(" ", text)                       # ремарки «(смеётся)», «(вздыхает)»: ≤ 4 слов в скобках
        if sdh_v2:
            text = CAPS_RE.sub(_caps, SDH2_RE.sub(" ", text))
        text = re.sub(r"(^|\s)[-–—]\s*", r"\1", text).strip()
        if text:
            cues.append((t0, t1, text))
    return cues


@lru_cache(maxsize=2_000_000)
def analyze(w: str):
    p = _MORPH.parse(w)[0]
    return p.normal_form.replace("ё", "е"), (p.tag.POS or "UNKN"), bool(PROPER & set(p.tag.grammemes))


def _init_ingest(cfg, tcfg):
    global CFG, _MORPH, PROPER
    import pymorphy3
    CFG = cfg
    _MORPH = pymorphy3.MorphAnalyzer()
    PROPER = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}
    T._setup(tcfg)


def ingest_shard(args):
    shard, items, zpath, outp = args
    if outp.exists():
        return shard, 0
    from razdel import tokenize
    sc = CFG["subs"]
    cre = re.compile(sc["credit_re"])
    rows, n_ok = [], 0
    with zipfile.ZipFile(zpath) as z:
        for ep, path in items:
            cues = parse_srt(z.read(path), sc.get("sdh_v2", False))
            cues = [c for c in cues if not cre.search(c[2])]
            dedup = []
            for c in cues:
                if not dedup or c[2] != dedup[-1][2]:
                    dedup.append(c)
            txt = " ".join(c[2] for c in dedup)
            letters = LETTER.findall(txt)
            if not letters or len(CYR.findall(txt)) / len(letters) < sc["min_cyrillic_share"]:
                continue
            words_total = 0
            ep_rows = []
            for ci, (t0, t1, text) in enumerate(dedup):
                toks = [t.text for t in tokenize(text) if LETTER.search(t.text)]
                lem, pos = [], []
                for tok in toks:
                    l_, p_, pr = analyze(tok.lower())
                    lem.append(l_)
                    pos.append("PROPN" if (pr and tok[:1].isupper()) else p_)
                content = T.content_tokens(" ".join(lem), " ".join(pos)) if toks else []
                words_total += len(toks)
                ep_rows.append((ep, ci, t0, t1, len(toks), " ".join(content)))
            if words_total < sc["min_words"]:
                continue
            rows.extend(ep_rows)
            n_ok += 1
    pd.DataFrame(rows, columns=["episode_id", "cue_idx", "t0", "t1", "n_words", "content"]).to_parquet(outp, index=False)
    return shard, n_ok


def cmd_subs_ingest(cfg, sample):
    import sqlite3
    sc = cfg["subs"]
    od = out_dir(cfg, sample) / "subs_cues"
    od.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ROOT / sc["sqlite"])
    ep = pd.read_sql("""select e.episode_id, e.show_id, e.season_number, e.episode_number, s.relative_path,
                               sh.name show_name, sh.release_year, sh.genres_json, sh.countries_json, sh.imdb_id
                        from subtitles s join episodes e using(episode_id) join shows sh using(show_id)
                        where lower(s.language) = lower(?) and s.status = 'downloaded'""", con, params=[sc["language"]])
    with zipfile.ZipFile(ROOT / sc["zip"]) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX") and n.endswith(f"_{sc['language']}.srt")]
    # путь в архиве: data_ororo/Serials/<show dir>/Season<s>/<name>_s<s>_e<e>_RU.srt; relative_path из базы — хвост
    by_tail = {}
    for n in names:
        by_tail[n.split("data_ororo/", 1)[-1]] = n
    def find(rp):
        if not isinstance(rp, str):
            return None
        rp = rp.replace("\\", "/")
        for cand in (rp, "Serials/" + rp, rp.split("data_ororo/", 1)[-1]):
            if cand in by_tail:
                return by_tail[cand]
        return None
    ep["zip_path"] = ep["relative_path"].map(find)
    miss = ep["zip_path"].isna()
    if miss.mean() > 0.5:   # сопоставление по хвосту пути не сработало — по имени файла
        base = {}
        for n in names:
            base.setdefault(Path(n).name, n)
        ep.loc[miss, "zip_path"] = ep.loc[miss, "relative_path"].map(lambda r: base.get(Path(str(r)).name))
    print(f"эпизодов с RU в базе: {len(ep)}, найдено в архиве: {ep['zip_path'].notna().sum()}, файлов RU в архиве: {len(names)}")
    ep = ep[ep["zip_path"].notna()].copy()
    rng = np.random.default_rng(cfg["seed"])
    if sample:
        shows = rng.choice(ep["show_id"].unique(), sc["sample_shows"], replace=False)
        ep = (ep[ep["show_id"].isin(shows)].sample(frac=1.0, random_state=cfg["seed"])
              .groupby("show_id").head(sc["sample_eps_per_show"]))
    ep.drop(columns=["relative_path"]).to_parquet(out_dir(cfg, sample) / "subs_episodes_all.parquet", index=False)
    items = list(zip(ep["episode_id"], ep["zip_path"]))
    nsh = 4 if sample else sc["shards"]
    tasks = [(i, items[i::nsh], ROOT / sc["zip"], od / f"shard-{i:03d}.parquet") for i in range(nsh)]
    tcfg = load_config(ROOT / cfg["topics_config"])
    t0 = time.time()
    with Pool(min(8, nsh), initializer=_init_ingest, initargs=(cfg, tcfg)) as pool:
        for sh, n in pool.imap_unordered(ingest_shard, tasks):
            print(f"[{time.time() - t0:6.0f}s] shard {sh}: {n} эпизодов", flush=True)


def load_cues(cfg, sample):
    return pd.concat([pd.read_parquet(p) for p in sorted((out_dir(cfg, sample) / "subs_cues").glob("shard-*.parquet"))],
                     ignore_index=True)


# ------------------------------------------------------------------ build
def chunks_of(tokens, size, cap, rng):
    ch = [tokens[i:i + size] for i in range(0, len(tokens) - size + 1, size)]
    if len(ch) > cap:
        idx = np.sort(rng.choice(len(ch), cap, replace=False))
        ch = [ch[i] for i in idx]
    return ch


def make_vocab(d, vc):
    df_doc, df_text = Counter(), {}
    for doc, toks in zip(d["doc_id"], d["tokens"]):
        u = set(toks.split())
        df_doc.update(u)
        for t in u:
            df_text.setdefault(t, set()).add(doc)
    vocab = sorted(t for t, c in df_doc.items() if len(df_text[t]) >= vc["min_texts"] and c / len(d) <= vc["max_doc_share"])
    freq = sorted((t for t, c in df_doc.items() if c / len(d) > vc["max_doc_share"]), key=lambda t: -df_doc[t])
    return vocab, freq


def cmd_build(cfg, sample):
    tc = cfg["train"]
    tcfg = load_config(ROOT / cfg["topics_config"])
    od = out_dir(cfg, sample)
    rng = np.random.default_rng(cfg["seed"])
    # субтитры
    ep = pd.read_parquet(od / "subs_episodes_all.parquet")
    cues = load_cues(cfg, sample)
    have = set(cues["episode_id"])
    ep = ep[ep["episode_id"].isin(have)]
    ep = ep.sample(frac=1.0, random_state=cfg["seed"]).groupby("show_id").head(tc["subs_eps_per_show"])
    target = tc["sample_chunks"] if sample else tc["subs_chunks"]
    toks = {e: [t for c in g["content"] for t in c.split()] for e, g in
            cues[cues["episode_id"].isin(set(ep["episode_id"]))].sort_values(["episode_id", "cue_idx"]).groupby("episode_id")}
    docs = []
    for r in ep.itertuples():
        for k, c in enumerate(chunks_of(toks.get(r.episode_id, []), tc["chunk_tokens"], tc["max_chunks_per_text"], rng)):
            docs.append(dict(stratum="subs", corpus="ororo", doc_id=f"ep:{r.episode_id}", unit=f"show:{r.show_id}",
                             chunk=k, tokens=" ".join(c)))
    d = pd.DataFrame(docs)
    if len(d) > target:
        d = d.sample(target, random_state=cfg["seed"]).reset_index(drop=True)
    vocab, freq = make_vocab(d, tcfg["vocab"])
    sv = cfg["subs"].get("vocab_extra")
    if sv:   # v2: имена персонажей и прочая лексика одного сериала — не тема корпуса
        n_occ, by_unit, units = Counter(), {}, {}
        for u, t in zip(d["unit"], d["tokens"]):
            for w, c in Counter(t.split()).items():
                n_occ[w] += c
                by_unit.setdefault(w, Counter())[u] += c
        keep = [w for w in vocab if len(by_unit[w]) >= sv["min_units"]
                and max(by_unit[w].values()) / n_occ[w] <= sv["max_unit_share"]]
        drop = sorted(set(vocab) - set(keep), key=lambda w: -n_occ[w])
        print(f"субтитры, доп. фильтр словаря: убрано {len(drop)} лемм; самые частые: {drop[:40]}")
        vocab = keep
    (od / "subs").mkdir(exist_ok=True)
    d.to_parquet(od / "subs" / "train_docs.parquet", index=False)
    (od / "subs" / "vocab.json").write_text(json.dumps(dict(vocab=vocab, removed_frequent=freq), ensure_ascii=False))
    print(f"субтитры: {len(d)} кусков из {d['doc_id'].nunique()} эпизодов, {d['unit'].nunique()} сериалов; словарь {len(vocab)}; "
          f"слишком частые: {freq[:25]}")
    # проза + фанфики: куски этапа 3 (сбалансированы по группам, ≤ 3 текста на автора)
    src = ROOT / tcfg["out_dir"] / ("sample" if sample else "full") / "train_docs.parquet"
    p = pd.read_parquet(src)
    p = p[p["stratum"].isin(cfg["prose"]["strata"])].reset_index(drop=True)
    vocab, freq = make_vocab(p, tcfg["vocab"])
    (od / "prose").mkdir(exist_ok=True)
    p.to_parquet(od / "prose" / "train_docs.parquet", index=False)
    (od / "prose" / "vocab.json").write_text(json.dumps(dict(vocab=vocab, removed_frequent=freq), ensure_ascii=False))
    print(f"проза + фанфики: {len(p)} кусков ({p.groupby('stratum').size().to_dict()}); словарь {len(vocab)}; "
          f"слишком частые: {freq[:25]}")


# ------------------------------------------------------------------ train
def cmd_train(cfg, sample):
    import tomotopy as tp
    tc = cfg["train"]
    od = out_dir(cfg, sample)
    md = ROOT / cfg["models_dir"] / ("sample" if sample else "full")
    md.mkdir(parents=True, exist_ok=True)
    Ks = tc["sample_K"] if sample else tc["K"]
    iters = tc["sample_iterations"] if sample else tc["iterations"]
    for corp in ["subs", "prose"]:
        d = pd.read_parquet(od / corp / "train_docs.parquet")
        vocab = set(json.loads((od / corp / "vocab.json").read_text())["vocab"])
        for K in Ks:
            f = md / f"lda_{corp}_K{K}.bin"
            tj = od / corp / f"topics_K{K}.json"
            if f.exists() and tj.exists():
                print("уже есть:", f)
                continue
            if f.exists():
                mdl = tp.LDAModel.load(str(f))
            else:
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
                    print(f"{corp} K={K} iter {i + 50}: ll/слово {mdl.ll_per_word:.4f} ({time.time() - t0:.0f} с)", flush=True)
                mdl.save(str(f))
            T.topic_summary(mdl, d, vocab, od / corp, K, cfg)


# ------------------------------------------------------------------ infer
_MODELS: dict = {}


def _init_infer(paths, iters):
    import tomotopy as tp
    global _MODELS, CFG
    _MODELS = {k: tp.LDAModel.load(p) for k, p in paths.items()}
    CFG = {"iters": iters}


def infer_batch(args):
    """args: (key, units) — units: список (doc_id, seg, words); → (key, rows, theta)."""
    key, units, outp = args
    if outp.exists():
        return key
    mdl = _MODELS[key]
    vocab = set(mdl.used_vocabs)
    mdocs = [mdl.make_doc([w for w in ws if w in vocab] or ["__empty__"]) for _, _, ws in units]
    theta, _ = mdl.infer(mdocs, iterations=CFG["iters"], workers=1, together=False)
    th = np.asarray(theta, dtype=np.float32).reshape(len(mdocs), -1)
    df = pd.DataFrame([(u[0], u[1], sum(w in vocab for w in u[2])) for u in units], columns=["doc_id", "seg", "n_content"])
    df["theta"] = list(th)
    df.to_parquet(outp, index=False)
    return key


def time_segments(t0, n):
    """Границы по экранному времени: реплика относится к части по своему началу."""
    t = np.asarray(t0, float)
    lo, hi = t.min(), t.max() + 1
    return np.minimum(((t - lo) / (hi - lo) * n).astype(int), n - 1)


def subs_units(cfg, sample):
    """Эпизоды для профилей (≤ eps_per_show_analysis на сериал) → сегменты по словам и по времени."""
    od = out_dir(cfg, sample)
    ep = pd.read_parquet(od / "subs_episodes_all.parquet")
    cues = load_cues(cfg, sample)
    ep = ep[ep["episode_id"].isin(set(cues["episode_id"]))]
    ep = ep.sample(frac=1.0, random_state=cfg["seed"] + 1).groupby("show_id").head(cfg["subs"]["eps_per_show_analysis"])
    cues = cues[cues["episode_id"].isin(set(ep["episode_id"]))].sort_values(["episode_id", "cue_idx"])
    N, Nt = cfg["infer"]["N"], cfg["infer"]["N_time"]
    words, timed = [], []
    for e, g in cues.groupby("episode_id", sort=False):
        cont = [c.split() for c in g["content"]]
        segs = relative_segments(np.cumsum(g["n_words"].to_numpy()), N)
        if segs is None:
            continue
        for k, (a, b) in enumerate(segs):
            words.append((f"ep:{e}", k, [w for c in cont[a:b] for w in c]))
        ts = time_segments(g["t0"].to_numpy(), Nt)
        for k in range(Nt):
            timed.append((f"ep:{e}", k, [w for c, s in zip(cont, ts) if s == k for w in c]))
    meta = ep.assign(doc_id="ep:" + ep["episode_id"].astype(str), unit="show:" + ep["show_id"].astype(str))
    return words, timed, meta


def prose_units(cfg, sample):
    tcfg = load_config(ROOT / cfg["topics_config"])
    T._setup(tcfg)
    pc = cfg["prose"]
    a = pd.read_parquet(ROOT / tcfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(["text", "linked_work"]) & a["form"].isin(pc["strata"])]
    s01 = T.s01_dir(tcfg, sample)
    have = set(pd.read_parquet(s01 / "docs.parquet", columns=["doc_id"])["doc_id"])
    a = a[a["doc_id"].isin(have)]
    n = pc["sample_texts_per_stratum"] if sample else pc["texts_per_stratum"]
    a = (a.sample(frac=1.0, random_state=cfg["seed"]).groupby(["form", "author_id"]).head(1)
         .groupby("form").head(n))
    content = T.read_content(s01, set(a["doc_id"]))
    seg = pd.read_parquet(s01 / f"segments_rel_N{cfg['infer']['N']}.parquet", filters=[("doc_id", "in", list(a["doc_id"]))])
    units = []
    for r in seg.itertuples():
        sents = content.get(r.doc_id)
        if sents is None:
            continue
        units.append((r.doc_id, r.seg, [w for s in sents[r.sent_start:r.sent_end] for w in s]))
    meta = a[["doc_id", "form", "author_id", "n_words"]].assign(unit="author:" + a["author_id"].astype(str))
    return units, meta


def cmd_infer(cfg, sample):
    od = out_dir(cfg, sample)
    md = ROOT / cfg["models_dir"] / ("sample" if sample else "full")
    Ks = cfg["train"]["sample_K"] if sample else cfg["train"]["K"]
    words, timed, smeta = subs_units(cfg, sample)
    smeta.to_parquet(od / "subs_meta.parquet", index=False)
    punits, pmeta = prose_units(cfg, sample)
    pmeta.to_parquet(od / "prose_meta.parquet", index=False)
    print(f"субтитры: {smeta['doc_id'].nunique()} эпизодов, {smeta['unit'].nunique()} сериалов; "
          f"проза: {pmeta.groupby('form').size().to_dict()}", flush=True)
    paths, tasks = {}, []
    B = 4000
    for K in Ks:
        for corp, sets in [("subs", {"words": words, "time": timed}), ("prose", {"words": punits})]:
            key = f"{corp}_K{K}"
            paths[key] = str(md / f"lda_{key}.bin")
            for kind, units in sets.items():
                d = od / "theta" / f"{key}_{kind}"
                d.mkdir(parents=True, exist_ok=True)
                for i in range(0, len(units), B):
                    tasks.append((key, units[i:i + B], d / f"b{i // B:04d}.parquet"))
    t0 = time.time()
    with Pool(cfg["infer"]["workers"], initializer=_init_infer, initargs=(paths, cfg["infer"]["iterations"])) as pool:
        for i, _ in enumerate(pool.imap_unordered(infer_batch, tasks)):
            if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                print(f"[{time.time() - t0:6.0f}s] {i + 1}/{len(tasks)}", flush=True)


def load_theta(od, key, kind):
    d = pd.concat([pd.read_parquet(p) for p in sorted((od / "theta" / f"{key}_{kind}").glob("b*.parquet"))], ignore_index=True)
    d = d.sort_values(["doc_id", "seg"])
    n = d.groupby("doc_id")["seg"].transform("size")
    d = d[n == n.max()]
    ids = d["doc_id"].drop_duplicates().to_numpy()
    X = np.stack(d["theta"].to_numpy()).reshape(len(ids), int(n.max()), -1)
    return ids, X


# ------------------------------------------------------------------ profiles
def unit_means(X, units):
    """Средние по единицам (сериал / автор), чтобы каждая единица весила одинаково: U × N × K, и индексы."""
    u, inv = np.unique(units, return_inverse=True)
    o = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inv[o])) + 1]
    S = np.add.reduceat(X[o], starts, axis=0)
    c = np.diff(np.r_[starts, len(o)])
    return S / c[:, None, None]


def shape_stats(P):
    """P: ... × N × K (профиль доли темы по позиции) → относительный профиль R и компоненты по ортогональным
    многочленам Лежандра позиции: b1 (линейная: > 0 — рост к концу), b2 (квадратичная: > 0 — края, < 0 — середина)."""
    N = P.shape[-2]
    x = np.linspace(-1, 1, N)
    Q, _ = np.linalg.qr(np.stack([np.ones(N), x, x ** 2], 1))   # дискретный ортонормированный базис (константа, x, x²)
    p1 = Q[:, 1] * np.sign(Q[-1, 1] - Q[0, 1])                 # > 0 — рост к концу
    p2 = Q[:, 2] * np.sign(Q[0, 2] - Q[N // 2, 2])             # > 0 — края выше середины
    R = P / P.mean(-2, keepdims=True)
    b1 = np.einsum("...nk,n->...k", R, p1)
    b2 = np.einsum("...nk,n->...k", R, p2)
    amp = np.sqrt(((R - 1) ** 2).sum(-2))
    return R, b1, b2, amp


def corpus_profiles(X, units, rng, cfg):
    pc = cfg["profiles"]
    U = unit_means(X, units)
    P = U.mean(0)
    R, b1, b2, amp = shape_stats(P)
    # бутстрэп по единицам (сериал / автор)
    nU = len(U)
    W = rng.poisson(1.0, (pc["bootstrap"], nU)).astype(np.float32)
    Pb = np.einsum("bu,unk->bnk", W, U) / W.sum(1)[:, None, None]
    Rb, b1b, b2b, _ = shape_stats(Pb)
    # нулевая модель: перестановка сегментов внутри текста (одна перестановка на текст на все темы)
    nb1, nb2, namp = [], [], []
    N = X.shape[1]
    for _ in range(pc["n_perm"]):
        perm = np.argsort(rng.random((X.shape[0], N)), axis=1)
        Xp = np.take_along_axis(X, perm[:, :, None], axis=1)
        _, a1, a2, am = shape_stats(unit_means(Xp, units).mean(0))
        nb1.append(a1), nb2.append(a2), namp.append(am)
    nb1, nb2, namp = map(np.array, (nb1, nb2, namp))
    z1 = (b1 - nb1.mean(0)) / nb1.std(0)
    z2 = (b2 - nb2.mean(0)) / nb2.std(0)
    p_amp = (1 + (namp >= amp).sum(0)) / (1 + pc["n_perm"])
    return dict(P=P, R=R, b1=b1, b2=b2, amp=amp, z1=z1, z2=z2, p_amp=p_amp,
                R_lo=np.percentile(Rb, 2.5, 0), R_hi=np.percentile(Rb, 97.5, 0),
                b1_ci=np.percentile(b1b, [2.5, 97.5], 0), b2_ci=np.percentile(b2b, [2.5, 97.5], 0), n_units=nU, n_texts=len(X))


def bh(p, q):
    p = np.asarray(p)
    o = np.argsort(p)
    th = q * np.arange(1, len(p) + 1) / len(p)
    ok = p[o] <= th
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(len(p), bool)
    sig[o[:k]] = True
    return sig


def classify(pr, q):
    """Класс формы: компонента с большим |z| (если тема значима по амплитуде после FDR): рост/спад/края/середина."""
    sig = bh(pr["p_amp"], q)
    cls = []
    for k in range(len(sig)):
        if not sig[k]:
            cls.append("плоский")
        elif abs(pr["z1"][k]) >= abs(pr["z2"][k]):
            cls.append("рост к концу" if pr["b1"][k] > 0 else "спад к концу")
        else:
            cls.append("края" if pr["b2"][k] > 0 else "середина")
    return np.array(cls), sig


def topic_table(pr, cls, tj, K, n_words=8):
    tops = tj["top_words"]
    coh = tj["coherence_c_npmi"]
    rows = []
    for k in range(K):
        R = pr["R"][:, k]
        rows.append({"тема": k, "слова": ", ".join(tops[str(k)][:n_words]), "доля, %": 100 * pr["P"][:, k].mean(),
                     "класс": cls[k], "R(1)": R[0], "R(N)": R[-1], "max R": R.max(), "пик": int(R.argmax()) + 1,
                     "z лин.": pr["z1"][k], "z кв.": pr["z2"][k], "p (амплитуда)": pr["p_amp"][k], "c_npmi": coh[str(k)]})
    return pd.DataFrame(rows)


def match_topics(mdl_a, mdl_b):
    """Сходство тем двух моделей: 1 − расстояние Йенсена–Шеннона (log2) между распределениями слов на общем словаре."""
    va, vb = list(mdl_a.used_vocabs), list(mdl_b.used_vocabs)
    common = sorted(set(va) & set(vb))
    ia = {w: i for i, w in enumerate(va)}
    ib = {w: i for i, w in enumerate(vb)}
    A = np.array([np.asarray(mdl_a.get_topic_word_dist(k))[[ia[w] for w in common]] for k in range(mdl_a.k)])
    Bm = np.array([np.asarray(mdl_b.get_topic_word_dist(k))[[ib[w] for w in common]] for k in range(mdl_b.k)])
    A /= A.sum(1, keepdims=True)
    Bm /= Bm.sum(1, keepdims=True)
    S = np.zeros((len(A), len(Bm)))
    for i in range(len(A)):
        M = 0.5 * (A[i] + Bm)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl1 = np.nansum(np.where(A[i] > 0, A[i] * np.log2(A[i] / M), 0), 1)
            kl2 = np.nansum(np.where(Bm > 0, Bm * np.log2(Bm / M), 0), 1)
        S[i] = 1 - np.sqrt(np.clip(0.5 * (kl1 + kl2), 0, 1))
    return S, len(common)


def cmd_profiles(cfg, sample):
    import tomotopy as tp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from s04_nulls import md as mdtab
    od = out_dir(cfg, sample)
    mdir = ROOT / cfg["models_dir"] / ("sample" if sample else "full")
    pc = cfg["profiles"]
    Ks = cfg["train"]["sample_K"] if sample else cfg["train"]["K"]
    figKs = [Ks[-1]] if sample else (pc["fig_K"] if isinstance(pc["fig_K"], list) else [pc["fig_K"]])
    def stream(K, corp, split):
        # независимый поток для каждой пары (K, корпус, разбиение): классы одного корпуса не зависят от другого
        # (v2: до 2026-11-27 был общий поток, и пересчёт субтитров сдвигал классы прозы)
        if not cfg.get("independent_streams", False):
            return shared
        return np.random.default_rng([cfg["seed"], K, ["subs", "prose"].index(corp), ["words", "time"].index(split)])
    shared = np.random.default_rng(cfg["seed"])
    smeta = pd.read_parquet(od / "subs_meta.parquet").set_index("doc_id")
    pmeta = pd.read_parquet(od / "prose_meta.parquet").set_index("doc_id")
    name = {"subs": "субтитры сериалов (Ororo, русские субтитры)", "prose": "проза + фанфики (Проза.ру, Ficbook)"}
    L = [f"# Этап 3s — Plot Arceology: темы субтитров и прозы + фанфиков{' (выборочный прогон)' if sample else ''}\n",
         "Отдельная LDA на каждом корпусе (правила словаря — этап 3). Позиционный профиль темы — средняя доля темы в "
         f"сегменте по позиции (N = {cfg['infer']['N']} равных по словам частей), сначала внутри единицы (сериал / автор), "
         "затем по единицам. R — профиль, делённый на среднюю долю темы (R = 1 — тема распределена ровно). Компоненты "
         "формы — проекции R на ортогональные многочлены Лежандра позиции: линейная (> 0 — рост к концу) и квадратичная "
         "(> 0 — края, < 0 — середина); z — относительно нулевой модели (перестановка сегментов внутри текста, "
         f"{pc['n_perm']} раз). Тема значима, если амплитуда профиля больше, чем в нулевой модели (FDR q = {pc['fdr_q']}); "
         "класс — компонента с большим |z|.\n"]
    res = {}
    for K in Ks:
        L.append(f"\n## K = {K}\n")
        for corp, meta in [("subs", smeta), ("prose", pmeta)]:
            key = f"{corp}_K{K}"
            ids, X = load_theta(od, key, "words")
            units = meta.loc[ids, "unit"].to_numpy()
            pr = corpus_profiles(X, units, stream(K, corp, "words"), cfg)
            cls, sig = classify(pr, pc["fdr_q"])
            tj = json.loads((od / corp / f"topics_K{K}.json").read_text())
            tab = topic_table(pr, cls, tj, K)
            tab.to_csv(od / f"topics_{key}.csv", index=False)
            res[key] = (pr, cls, tab, tj)
            cnt = pd.Series(cls).value_counts().to_dict()
            L.append(f"### {name[corp]}: {pr['n_texts']} текстов, {pr['n_units']} единиц; ll/слово {tj['ll_per_word']:.3f}, "
                     f"средняя c_npmi {np.mean(list(tj['coherence_c_npmi'].values())):.3f}\n")
            L.append("Классы формы профиля: " + ", ".join(f"{k} — {v}" for k, v in cnt.items()) + ".\n")
            if corp == "subs":
                ids_t, Xt = load_theta(od, key, "time")
                prt = corpus_profiles(Xt, smeta.loc[ids_t, "unit"].to_numpy(), stream(K, "subs", "time"), cfg)
                clt, _ = classify(prt, pc["fdr_q"])
                agree = np.mean(clt == cls)
                r1 = np.corrcoef(pr["z1"], prt["z1"])[0, 1]
                r2 = np.corrcoef(pr["z2"], prt["z2"])[0, 1]
                L.append(f"Проверка: {cfg['infer']['N_time']} частей экранного времени (как у Schmidt) вместо {cfg['infer']['N']} "
                         f"частей по словам — совпадение классов {agree:.0%}; r(z лин.) = {r1:.2f}, r(z кв.) = {r2:.2f}.\n")
            show = tab.copy()
            show["ord"] = np.where(show["класс"] == "плоский", 0, np.maximum(show["z лин."].abs(), show["z кв."].abs()))
            show = show.sort_values("ord", ascending=False).drop(columns="ord")
            for c in ["доля, %", "R(1)", "R(N)", "max R", "z лин.", "z кв.", "c_npmi"]:
                show[c] = show[c].map(lambda v: f"{v:.2f}")
            show["p (амплитуда)"] = show["p (амплитуда)"].map(lambda v: f"{v:.3f}")
            L.append(f"<details><summary>Все {K} тем (по убыванию силы позиционного профиля)</summary>\n\n" + mdtab(show) + "\n\n</details>\n")
        # сопоставление тем
        ma = tp.LDAModel.load(str(mdir / f"lda_subs_K{K}.bin"))
        mb = tp.LDAModel.load(str(mdir / f"lda_prose_K{K}.bin"))
        S, nc = match_topics(ma, mb)
        np.save(od / f"match_K{K}.npy", S)
        ts, tp_ = res[f"subs_K{K}"], res[f"prose_K{K}"]
        best = S.argmax(1)
        mutual = [i for i in range(len(S)) if S[:, best[i]].argmax() == i]
        rows = []
        for i in sorted(mutual, key=lambda i: -S[i, best[i]])[:pc["match_top"]]:
            j = best[i]
            rows.append({"субтитры: тема": f"{i}: " + ", ".join(ts[3]["top_words"][str(i)][:6]),
                         "проза: тема": f"{j}: " + ", ".join(tp_[3]["top_words"][str(j)][:6]),
                         "сходство (1 − JSD)": f"{S[i, j]:.2f}", "класс: субтитры": ts[1][i], "класс: проза": tp_[1][j],
                         "r профилей R": f"{np.corrcoef(ts[0]['R'][:, i], tp_[0]['R'][:, j])[0, 1]:+.2f}"})
        L.append(f"### Сопоставление тем двух корпусов (K = {K}; общий словарь {nc} лемм; взаимно ближайшие пары — "
                 f"{len(mutual)}; показаны {min(len(mutual), pc['match_top'])} самых похожих)\n")
        L.append(mdtab(pd.DataFrame(rows)) + "\n")
        same = [ts[1][i] == tp_[1][best[i]] for i in mutual if S[i, best[i]] >= 0.5]
        L.append(f"Пар со сходством ≥ 0,5: {len(same)}; из них с одинаковым классом формы профиля — {sum(same)}.\n")
    # рисунки
    for figK in figKs:
        classes = ["рост к концу", "спад к концу", "середина", "края"]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0"]
        x = np.arange(1, cfg["infer"]["N"] + 1)
        for r_, corp in enumerate(["subs", "prose"]):
            pr, cls, tab, tj = res[f"{corp}_K{figK}"]
            for c_, cl in enumerate(classes):
                ax = axes[r_, c_]
                t = tab[tab["класс"] == cl].copy()
                t["s"] = (t["z лин."] if cl in classes[:2] else t["z кв."]).abs()
                if not len(t):
                    ax.text(0.5, 0.5, "нет значимых тем этого класса", ha="center", va="center", transform=ax.transAxes,
                            color="#52514e", fontsize=9)
                for n_, k in enumerate(t.sort_values("s", ascending=False)["тема"].head(pc["fig_per_class"])):
                    ax.plot(x, pr["R"][:, k], color=cols[n_], lw=2, label=", ".join(tj["top_words"][str(k)][:3]))
                    ax.fill_between(x, pr["R_lo"][:, k], pr["R_hi"][:, k], color=cols[n_], alpha=0.15, lw=0)
                ax.axhline(1, color="#9a9994", lw=1, ls="--")
                ax.set_title(f"{'Субтитры' if corp == 'subs' else 'Проза + фанфики'}: {cl} ({(cls == cl).sum()} тем)", fontsize=10)
                ax.legend(fontsize=7.5, frameon=False, loc="best")
                ax.grid(alpha=0.3, lw=0.5)
                if c_ == 0:
                    ax.set_ylabel("доля темы / средняя доля")
                if r_ == 1:
                    ax.set_xlabel("позиция сегмента (N = 20)")
        fig.suptitle(f"Позиционные профили тем (LDA, K = {figK}; в каждом классе — до {pc['fig_per_class']} тем с наибольшим |z|; 95 % ДИ — бутстрэп по сериалам / авторам)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fp = ROOT / "reports" / "figures" / f"03s_profiles_K{figK}{cfg.get('report_suffix', '')}{'_sample' if sample else ''}.png"
        fig.savefig(fp, dpi=200)
        fig.savefig(fp.with_suffix(".pdf"))
        L.append(f"\n![профили](figures/{fp.name})\n")
    rep = ROOT / "reports" / f"03s_arceology{cfg.get('report_suffix', '')}{'_sample' if sample else ''}.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    print("отчёт:", rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("step", choices=["subs_ingest", "build", "train", "infer", "profiles"])
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    T._setup(load_config(ROOT / cfg["topics_config"]))
    {"subs_ingest": cmd_subs_ingest, "build": cmd_build, "train": cmd_train, "infer": cmd_infer,
     "profiles": cmd_profiles}[a.step](cfg, a.sample)


if __name__ == "__main__":
    main()
