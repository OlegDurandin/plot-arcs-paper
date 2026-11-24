"""Этап 1. Предобработка и сегментация.

Запуск:
    .venv/bin/python src/s01_segment.py configs/s01_segment.yaml           # выборка (sample_per_group из конфига)
    .venv/bin/python src/s01_segment.py configs/s01_segment.yaml --full    # все единицы

Результат — каталог data/processed/s01_<hash>[_sample]/:
    sentences/shard-XXXXX.parquet  — предложения: doc_id, part_idx, part_id, sent_idx, text, n_words, lemmas, pos
    docs.parquet                   — по документу: длина после очистки, число предложений, что удалено очисткой,
                                     проверка орфографии (дореформенные буквы)
    segments_rel_N{10,20,50}.parquet — сегменты как диапазоны предложений [sent_start, sent_end)
    segments_abs_W{...}.parquet    — окна фиксированной длины с шагом W/2, выровненные по предложениям
    meta.json                      — хеш, W0, сводка
Кэш: каталог определяется хешем (конфиг + код этапа); готовые шарды при повторном запуске не пересчитываются.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, config_hash, load_config  # noqa: E402

CFG: dict = {}
_LOADER = None
_MORPH = None

LETTER = re.compile(r"[^\W\d_]")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\S+\.(?:ru|com|net|org|me)/\S*", re.I)
INVISIBLE = dict.fromkeys(map(ord, "­​‌‍‎‏⁠﻿"), None)
PREREFORM = re.compile(r"[ѣѢіІѳѲѵѴ]")
NONRU = re.compile(r"[іїєґўІЇЄҐЎ]")
LATIN = re.compile(r"^[A-Za-z]+(?:[-'][A-Za-z]+)*$")
HARD_END = re.compile(r"[^\W\d_]{2,}ъ\b", re.I)
MAX_SENT_WORDS = 150          # «предложение» без знаков препинания длиннее — режется на куски


# ------------------------------------------------------------------ очистка
def strip_accents(s: str) -> str:
    # снимаем только комбинируемые диакритики, кроме бреве у «й» и диерезиса у «ё» (они в NFC — отдельные буквы)
    nfd = unicodedata.normalize("NFD", s)
    out = "".join(ch for ch in nfd if not (unicodedata.combining(ch) and ch not in "̆̈"))
    return unicodedata.normalize("NFC", out)


def clean_part(text: str, corpus: str, stats: dict) -> str:
    c = CFG["clean"]
    if c["remove_invisible"]:
        text = text.translate(INVISIBLE)
    if c["remove_combining_accents"]:
        n0 = len(text)
        text = strip_accents(text)
        stats["accents_removed"] += n0 - len(text)
    lines = text.replace("\r", "").split("\n")

    if corpus == "ficbook":
        fc = c["ficbook"]
        # заголовок главы: первая непустая строка, за которой пустая
        if fc["drop_first_header"] and len(lines) > 1 and lines[0].strip() and not lines[1].strip() \
                and len(lines[0].split()) <= fc["header_max_words"]:
            stats["header_dropped"] += 1
            lines = lines[2:]
        nre = re.compile(fc["notes_header_re"])
        idx = [i for i, ln in enumerate(lines) if nre.match(ln)]
        # примечания в начале: блок до ближайшей пустой строки
        if idx and idx[0] <= fc["notes_start_max_line"]:
            i = idx[0]
            j = i + 1
            while j < len(lines) and lines[j].strip():
                j += 1
            stats["notes_start_cut_lines"] += j - i
            lines = lines[:i] + lines[j:]
            idx = [k for k, ln in enumerate(lines) if nre.match(ln)]
        # примечания в конце: до конца файла
        tail = [i for i in idx if i >= c["notes_cut_min_position"] * len(lines)]
        if tail:
            stats["notes_end_cut_lines"] += len(lines) - tail[0]
            lines = lines[:tail[0]]
        dre = re.compile(fc["drop_line_re"])
        n0 = len(lines)
        lines = [ln for ln in lines if not dre.match(ln)]
        stats["service_lines_dropped"] += n0 - len(lines)

    if corpus == "proza":
        pc = c["proza"]
        dre = re.compile(pc["drop_line_re"])
        n0 = len(lines)
        lines = [ln for ln in lines if not dre.match(ln)]
        stats["service_lines_dropped"] += n0 - len(lines)
        nre = re.compile(pc["notes_header_re"])
        idx = [i for i, ln in enumerate(lines) if nre.match(ln)]
        if idx and idx[-1] >= pc["notes_cut_min_position"] * len(lines):
            stats["notes_end_cut_lines"] += len(lines) - idx[-1]
            lines = lines[:idx[-1]]
        fre = re.compile(pc["footnote_mark_re"])
        lines = [fre.sub("", ln) for ln in lines]

    if corpus == "fairytales":
        bre = re.compile(c["fairytales"]["drop_brace_notes_re"])
        lines = [bre.sub("", ln) for ln in lines]

    if c["remove_urls"]:
        new = [URL_RE.sub(" ", ln) for ln in lines]
        stats["urls_removed"] += sum(a != b for a, b in zip(lines, new))
        lines = new
    if c["drop_lines_without_letters"]:
        lines = [ln for ln in lines if LETTER.search(ln)]
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in lines)


# ------------------------------------------------------------------ морфология
@lru_cache(maxsize=2_000_000)
def analyze(word_lower: str) -> tuple[str, str, bool]:
    p = _MORPH.parse(word_lower)[0]
    proper = bool(set(CFG["morph"]["proper_grammemes"]) & set(p.tag.grammemes))
    return p.normal_form.replace("ё", "е"), (p.tag.POS or "UNKN"), proper


def process_doc(doc: dict) -> tuple[list[dict], dict]:
    from razdel import sentenize, tokenize
    stats = dict(accents_removed=0, header_dropped=0, notes_start_cut_lines=0, notes_end_cut_lines=0,
                 service_lines_dropped=0, urls_removed=0, forced_splits=0)
    parts = _LOADER.load(doc["doc_id"], doc["text_path"])
    raw_chars = sum(len(t) for _, t in parts)
    rows, sidx = [], 0
    pre_cnt = hard_cnt = nonru_cnt = latin_words = 0
    for pidx, (pid, text) in enumerate(parts):
        clean = clean_part(text, doc["corpus"], stats)
        pre_cnt += len(PREREFORM.findall(clean))
        nonru_cnt += len(NONRU.findall(clean))
        hard_cnt += len(HARD_END.findall(clean))
        for para in clean.split("\n"):
            for s in sentenize(para):
                toks = [t.text for t in tokenize(s.text) if LETTER.search(t.text)]
                if not toks:
                    continue
                chunks = [toks[i:i + MAX_SENT_WORDS] for i in range(0, len(toks), MAX_SENT_WORDS)]
                if len(chunks) > 1:
                    stats["forced_splits"] += 1
                for k, ch in enumerate(chunks):
                    latin_words += sum(1 for tok in ch if LATIN.match(tok))
                    lem, pos = [], []
                    for tok in ch:
                        l_, p_, proper = analyze(tok.lower())
                        lem.append(l_)
                        pos.append("PROPN" if (proper and tok[:1].isupper()) else p_)
                    rows.append(dict(doc_id=doc["doc_id"], part_idx=pidx, part_id=pid, sent_idx=sidx,
                                     text=s.text if len(chunks) == 1 else " ".join(ch), n_words=len(ch),
                                     lemmas=" ".join(lem), pos=" ".join(pos)))
                    sidx += 1
    nw = max(1, sum(r["n_words"] for r in rows))
    lc = CFG["lang_check"]
    stats["nonru_letters_per_word"] = round(nonru_cnt / nw, 4)
    stats["latin_words_share"] = round(latin_words / nw, 4)
    stats["non_russian"] = bool(stats["nonru_letters_per_word"] >= lc["non_russian_letters_per_word"]
                                or stats["latin_words_share"] >= lc["latin_words_share"])
    info = dict(doc_id=doc["doc_id"], n_parts=len(parts), raw_chars=raw_chars,
                n_sent=sidx, n_words_clean=int(sum(r["n_words"] for r in rows)),
                prereform_chars=pre_cnt, hard_sign_word_end=hard_cnt, **stats)
    return rows, info


def _init(cfg):
    global CFG, _LOADER, _MORPH
    import pymorphy3
    from textio import TextLoader
    CFG = cfg
    _LOADER = TextLoader()
    _MORPH = pymorphy3.MorphAnalyzer()


def run_shard(args) -> str:
    shard_id, docs, sent_dir, docs_dir = args
    out_s = sent_dir / f"shard-{shard_id:05d}.parquet"
    out_d = docs_dir / f"shard-{shard_id:05d}.parquet"
    if out_s.exists() and out_d.exists():
        return f"skip {shard_id}"
    t0 = time.time()
    rows, infos = [], []
    for doc in docs:
        r, info = process_doc(doc)
        rows += r
        infos.append(info)
    tmp = out_s.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False, compression="zstd")
    tmp.rename(out_s)
    pd.DataFrame(infos).to_parquet(out_d, index=False)
    nw = sum(i["n_words_clean"] for i in infos)
    return f"shard {shard_id}: {len(docs)} док., {nw} слов, {time.time() - t0:.0f} с"


# ------------------------------------------------------------------ сегментация
def relative_segments(cum: np.ndarray, n: int) -> list[tuple[int, int]] | None:
    """cum — накопленные слова на концах предложений (len = n_sent). Границы — ближайшие концы предложений
    к k*T/N; каждый сегмент — хотя бы одно предложение. None, если предложений меньше N."""
    ns = len(cum)
    if ns < n:
        return None
    total = cum[-1]
    b = [0]
    for k in range(1, n):
        target = k * total / n
        i = int(np.argmin(np.abs(cum - target))) + 1          # граница после предложения i-1
        i = max(i, b[-1] + 1)
        i = min(i, ns - (n - k))
        b.append(i)
    b.append(ns)
    return list(zip(b[:-1], b[1:]))


def absolute_windows(cum: np.ndarray, w: int, step: int) -> list[tuple[int, int]]:
    total = int(cum[-1]) if len(cum) else 0
    bounds = np.concatenate([[0], cum])                       # слова на границах предложений 0..ns
    out = []
    s = 0
    while s + w <= total:
        a = int(np.argmin(np.abs(bounds - s)))
        z = int(np.argmin(np.abs(bounds - (s + w))))
        if z > a:
            out.append((a, z))
        s += step
    return out


def build_segments(run_dir: Path, cfg: dict, w0: int):
    """Сегменты по шардам (документ целиком лежит в одном шарде) — без загрузки всех предложений в память."""
    sc = cfg["segment"]
    ws = [(w0 if w == "W0" else int(w)) for w in sc["absolute_W"]]
    rel = {n: [] for n in sc["relative_N"]}
    ab = {w: [] for w in ws}
    n_docs = 0
    for f in sorted((run_dir / "sentences").glob("*.parquet")):
        sent = pd.read_parquet(f, columns=["doc_id", "sent_idx", "n_words"]).sort_values(["doc_id", "sent_idx"])
        for doc, g in sent.groupby("doc_id", sort=False):
            n_docs += 1
            cum = g["n_words"].to_numpy().cumsum()
            starts = np.concatenate([[0], cum])
            for n in rel:
                segs = relative_segments(cum, n)
                if segs is not None:
                    rel[n] += [(doc, k, a, z, int(starts[z] - starts[a])) for k, (a, z) in enumerate(segs)]
            for w in ab:
                step = max(1, int(round(w * sc["absolute_step"])))
                ab[w] += [(doc, k, a, z, int(starts[z] - starts[a])) for k, (a, z) in enumerate(absolute_windows(cum, w, step))]
    cols = ["doc_id", "seg", "sent_start", "sent_end", "n_words"]
    summary = {}
    for n, rows in rel.items():
        df = pd.DataFrame(rows, columns=cols)
        df.to_parquet(run_dir / f"segments_rel_N{n}.parquet", index=False)
        summary[f"rel_N{n}"] = dict(docs=int(df["doc_id"].nunique()), segments=len(df),
                                    docs_too_few_sentences=n_docs - int(df["doc_id"].nunique()))
    for w, rows in ab.items():
        df = pd.DataFrame(rows, columns=cols)
        name = f"segments_abs_W{w}.parquet"
        df.to_parquet(run_dir / name, index=False)
        summary[f"abs_W{w}"] = dict(file=name, docs=int(df["doc_id"].nunique()), windows=len(df),
                                    step=max(1, int(round(w * sc["absolute_step"]))))
    return summary


# ------------------------------------------------------------------ main
def select_docs(cfg: dict, full: bool) -> pd.DataFrame:
    a = pd.read_parquet(ROOT / cfg["analysis_set"])
    sel = cfg["select"]
    m = a["include"].copy() if sel["include"] else pd.Series(False, index=a.index)
    for f in sel["extra_flags"]:
        m |= a[f].fillna(False).astype(bool)
    m &= a["unit"].isin(sel["units"])
    if sel["skip_ficbook_chapters"]:
        m &= ~((a["unit"] == "chapter") & (a["corpus"] == "ficbook"))
    d = a[m].copy()
    if not full and cfg["sample_per_group"]:
        n = cfg["sample_per_group"]
        # перемешать и взять первые n в каждой группе (groupby.apply в pandas 3 теряет ключевые колонки)
        d = d.sample(frac=1.0, random_state=cfg["seed"]).groupby(["unit", "corpus", "form"]).head(n)
        # базовые наборы Афанасьева — целиком (нужны для W0 и проверки)
        af = a[m & (a["corpus"] == "afanasyev")]
        d = pd.concat([d, af[~af["doc_id"].isin(d["doc_id"])]])
    d = d.sort_values("doc_id").reset_index(drop=True)
    assert d[["unit", "corpus", "form", "text_path"]].notna().all().all(), "пустые метаданные в отборе"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.full:
        cfg["sample_per_group"] = None
    code = "".join((Path(__file__).parent / f).read_text() for f in ("s01_segment.py", "textio.py", "common.py"))
    chash = config_hash({**cfg, "_code": hashlib.sha1(code.encode()).hexdigest()})
    run_dir = ROOT / cfg["out_dir"] / (f"s01_{chash}" + ("" if args.full else "_sample"))
    sent_dir, docs_dir = run_dir / "sentences", run_dir / "docs"
    sent_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    docs = select_docs(cfg, args.full)
    print(f"[{run_dir.name}] документов: {len(docs)}", flush=True)
    print(docs.groupby(["unit", "corpus", "form"]).size().to_string(), flush=True)
    recs = docs[["doc_id", "corpus", "text_path"]].to_dict("records")
    # крупные документы (Ficbook, серии) — в отдельные маленькие шарды, чтобы не было длинного хвоста
    k = cfg["shard_docs"]
    shards, cur, cur_w = [], [], 0
    for r, nw in zip(recs, docs["n_words"].fillna(0)):
        cur.append(r)
        cur_w += nw
        if len(cur) >= k or cur_w >= 3_000_000:
            shards.append(cur)
            cur, cur_w = [], 0
    if cur:
        shards.append(cur)
    tasks = [(i, s, sent_dir, docs_dir) for i, s in enumerate(shards)]
    with Pool(cfg["n_workers"], initializer=_init, initargs=(cfg,)) as pool:
        for j, msg in enumerate(pool.imap_unordered(run_shard, tasks)):
            print(f"[{time.time() - t0:7.0f}s] {j + 1}/{len(tasks)} {msg}", flush=True)

    dinfo = pd.concat([pd.read_parquet(f) for f in sorted(docs_dir.glob("*.parquet"))], ignore_index=True)
    dinfo = docs[["doc_id", "unit", "corpus", "form", "n_words", "base_magic600"]].merge(dinfo, on="doc_id")
    dinfo.to_parquet(run_dir / "docs.parquet", index=False)
    base = dinfo[dinfo["base_magic600"].fillna(False).astype(bool)]
    w0 = int(round(base["n_words_clean"].median())) if len(base) else None
    print(f"W0 = медиана base_magic600 после очистки: {w0} (n={len(base)})", flush=True)
    summary = build_segments(run_dir, cfg, w0)
    meta = dict(config_hash=chash, full=args.full, n_docs=len(dinfo), W0=w0, W0_n=len(base),
                words_total=int(dinfo["n_words_clean"].sum()), seconds=round(time.time() - t0),
                cache_info=None, segments=summary)
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
