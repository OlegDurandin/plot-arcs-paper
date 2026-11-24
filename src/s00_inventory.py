"""Этап 0. Инвентаризация корпусов: единый манифест, длины, дубли.

Запуск:
    .venv/bin/python src/s00_inventory.py configs/s00_inventory.yaml [--full]

--full переопределяет sample_frac = 1.0. Результат кэшируется по хешу конфига:
если манифест с тем же хешем уже есть, пересчёта нет.

Артефакты (в out_dir):
    manifest[_sample].parquet            — единая схема + служебные поля
    ficbook_chapters[_sample].parquet    — главы Ficbook (для решения о единице анализа)
    minhash[_sample].npy                 — сигнатуры MinHash (порядок = порядок манифеста)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xxhash
from datasketch import MinHash

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, config_hash, in_sample, load_config, normalize_for_hash, words  # noqa: E402

CFG: dict = {}


# ---------------------------------------------------------------- обработка одного текста
def _h32(b: bytes) -> int:
    return xxhash.xxh32_intdigest(b)


def text_stats(item: tuple[int, str]) -> tuple[int, dict, np.ndarray | None]:
    idx, text = item
    dd, vv = CFG["dedup"], CFG["verse"]
    toks = words(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n_short = sum(1 for ln in lines if len(words(ln)) <= vv["short_line_words"])
    short_share = n_short / len(lines) if lines else 0.0
    norm = normalize_for_hash(text)
    st = {
        "n_words": len(toks),
        "n_chars": len(text),
        "n_lines": len(lines),
        "short_line_share": round(short_share, 4),
        "verse_like": bool(len(lines) >= vv["min_lines"] and short_share >= vv["share"]),
        "sha1_norm": hashlib.sha1(norm.encode()).hexdigest(),
    }
    sig = None
    if len(toks) >= dd["min_words"]:
        lw = norm.split()
        k = dd["shingle_words"]
        sh = {" ".join(lw[i:i + k]).encode() for i in range(max(1, len(lw) - k + 1))}
        sh = list(sh)
        m = MinHash(num_perm=dd["num_perm"], hashfunc=_h32, seed=CFG["seed"])
        for j in range(0, len(sh), 20000):          # пакетами: ограничение памяти на длинных текстах
            m.update_batch(sh[j:j + 20000])
        sig = m.hashvalues.astype(np.uint32)
    return idx, st, sig


def _init(cfg):
    global CFG
    CFG = cfg


# ---------------------------------------------------------------- источники
def iter_afanasyev(cfg, frac):
    """Афанасьев, «Народные русские сказки» (Викитека): один вариант под номером = один текст."""
    src = cfg["sources"]["afanasyev"]["parquet"]
    df = pd.read_parquet(ROOT / src)
    df = df[df["parsed"]]
    for r in df.itertuples(index=False):
        doc_id = f"af:{int(r.af_no)}"
        if not in_sample(doc_id, frac):
            continue
        yield dict(
            doc_id=doc_id, corpus="afanasyev", form="tale",
            # собиратель один; единица вывода — сюжет (подстраница Викитеки), а не «автор»
            author_id=f"af:{r.af_page}", author_name="А. Н. Афанасьев (собиратель)", date=None, title=r.af_page,
            lang_orig=r.lang, is_translation=False, tags=r.atu or None, likes=None,
            text_path=f"parquet:{src}#af_no={int(r.af_no)}", n_words_src=None, source_db="wikisource",
            ft_kind="folk", af_no=int(r.af_no), af_note=r.note, af_lang=r.lang,
        ), r.text
    return


def iter_fairytales(cfg, frac):
    ft_cfg = cfg["sources"]["fairytales"]
    db = ROOT / ft_cfg["db"]
    con = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    regions = {}
    for tid, slug in con.execute("select tale_id, region_slug from tale_regions order by tale_id, region_slug"):
        regions.setdefault(tid, []).append(slug)
    cur = con.execute("select tale_id, url, url_en, title, category, author, author_slug, readability, "
                      "n_words, published, text from tales order by tale_id")
    for (tid, url, url_en, title, cat, author, aslug, read, nw_src, pub, text) in cur:
        doc_id = f"ft:{tid}"
        if not in_sample(doc_id, frac):
            continue
        meta = dict(
            doc_id=doc_id, corpus="fairytales", form="tale",
            author_id=f"ft:{aslug or author}", author_name=author, date=None, title=title,
            # Весь корпус — русская версия англоязычного fairytalez.com (у всех записей есть url_en).
            lang_orig="en", is_translation=True,
            tags=json.dumps(regions.get(tid, []), ensure_ascii=False), likes=None,
            text_path=f"sqlite:{cfg['sources']['fairytales']['db']}#tales/{tid}",
            n_words_src=nw_src, source_db=Path(db).name,
            ft_kind=("fable" if author in ft_cfg["fable_authors"] else
                     "literary" if author in ft_cfg["literary_authors"] else "folk"),
            ft_regions=",".join(regions.get(tid, [])), ft_category=cat, ft_readability=read,
            site_published=pub, url=url,
        )
        yield meta, text or ""
    con.close()


def iter_proza(cfg, frac):
    pz = cfg["sources"]["proza"]
    t2f = pz["topic_to_form"]
    for rel in pz["dbs"]:
        db = ROOT / rel
        con = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
        # bucket = crc32(story_id) % 10000 — штатное поле для воспроизводимой выборки
        q = ("select story_id, url, topic, title, author_login, author_name, published, day, "
             "copyright_year, n_words, text from stories where status=200 and bucket < ?")
        cur = con.execute(q, (int(frac * 10000) if frac < 1 else 10000,))
        while True:
            rows = cur.fetchmany(cfg["chunk_rows"])
            if not rows:
                break
            for (sid, url, topic, title, login, aname, pub, day, cyear, nw_src, text) in rows:
                yield dict(
                    doc_id=f"pz:{topic}:{sid}", corpus="proza", form=t2f.get(topic, f"topic{topic}"),
                    author_id=f"pz:{login}", author_name=aname, date=day, title=title,
                    lang_orig="ru", is_translation=False, tags=None, likes=None,
                    text_path=f"sqlite:{rel}#stories/{sid}",
                    n_words_src=nw_src, source_db=Path(rel).name, url=url,
                    pz_topic=topic, pz_published=pub, pz_copyright_year=cyear,
                ), text or ""
        con.close()


FB_STATUSES = {"Завершён", "В процессе", "Заморожен"}


def likes_from_badges(badges):
    """Лайки — первое число после статуса в плашках страницы (проверено: совпадает с works.likes
    у 19 199 из 19 199 произведений, где likes известны; NULL в works.likes соответствует «0» на странице)."""
    if not badges:
        return None
    pos = [k for k, x in enumerate(badges) if x in FB_STATUSES]
    if not pos:
        return None
    nums = [x for x in badges[pos[-1] + 1:] if str(x).isdigit()]
    return int(nums[0]) if nums else 0


def iter_ficbook(cfg, frac, chapter_rows: list):
    fb = cfg["sources"]["ficbook"]
    base = ROOT / fb["base"]
    con = sqlite3.connect(f"file:{ROOT / fb['db']}?mode=ro", uri=True)
    works = pd.read_sql_query(
        "select w.work_id, w.url, w.title, w.author, w.author_url, w.fandoms_json, w.directions_json, "
        "w.genres_json, w.tags_json, w.rating, w.status, w.chapters_count, w.likes, w.rewards, "
        "w.comments_count, w.published_at, w.updated_at, w.language, w.coauthors_json, w.metadata_json "
        "from works w order by w.work_id", con)
    ch = pd.read_sql_query(
        "select work_id, chapter_number, chapter_url, title, published_at, path, n_words, status, error "
        "from chapter_texts order by work_id, chapter_number", con)
    con.close()
    # «terminal» — страница благодарности автору после последней главы, не глава
    ch["is_thanks_page"] = ch["chapter_url"].str.endswith("thanks-author-page")
    ch_by_work = {k: g for k, g in ch.groupby("work_id", sort=False)}
    for w in works.itertuples(index=False):
        doc_id = f"fb:{w.work_id}"
        if not in_sample(doc_id, frac):
            continue
        g = ch_by_work.get(w.work_id)
        parts, n_ok, n_bad, first_date = [], 0, 0, None
        if g is not None:
            for c in g.itertuples(index=False):
                if c.is_thanks_page:
                    continue
                txt = None
                if c.status == "complete" and isinstance(c.path, str):
                    p = base / c.path
                    if p.exists():
                        txt = p.read_text(encoding="utf-8", errors="replace")
                ok = txt is not None
                n_ok += ok
                n_bad += (not ok)
                if ok:
                    parts.append(txt)
                    if isinstance(c.published_at, str) and (first_date is None or c.published_at < first_date):
                        first_date = c.published_at
                chapter_rows.append(dict(
                    doc_id=doc_id, chapter_number=c.chapter_number, ch_title=c.title,
                    ch_date=c.published_at, ch_status=c.status, ch_ok=ok,
                    n_words=len(words(txt)) if ok else None, n_words_src=c.n_words,
                    text_path=str(Path(fb["base"]) / c.path) if isinstance(c.path, str) else None,
                ))
        w_date = w.published_at if isinstance(w.published_at, str) else None
        likes, likes_src = (int(w.likes), "work") if pd.notna(w.likes) else \
            (likes_from_badges(json.loads(w.metadata_json).get("badges")), "badges")
        if likes is None:
            likes_src = None
        fandoms = json.loads(w.fandoms_json or "[]")
        tags = json.loads(w.tags_json or "[]")
        yield dict(
            doc_id=doc_id, corpus="ficbook", form="fanfic",
            author_id=f"fb:{w.author_url or w.author}", author_name=w.author,
            date=w_date or first_date, title=w.title,
            lang_orig="ru", is_translation=False,
            tags=json.dumps(tags, ensure_ascii=False), likes=likes,
            text_path=str(Path(fb["base"]) / "data_ficbook/texts" / "*" / "*" / w.work_id),
            n_words_src=None, source_db=Path(fb["db"]).name, url=w.url,
            fb_date_source="work" if w_date else ("first_chapter" if first_date else None),
            fb_is_original=fandoms == ["Ориджиналы"], fb_fandoms=w.fandoms_json,
            fb_directions=w.directions_json, fb_genres=w.genres_json, fb_rating=w.rating,
            fb_status=w.status, fb_chapters_declared=w.chapters_count,
            fb_chapters_ok=n_ok, fb_chapters_missing=n_bad,
            fb_likes_source=likes_src,
            fb_rewards=w.rewards, fb_comments=w.comments_count, fb_updated=w.updated_at,
            fb_n_coauthors=len(json.loads(w.coauthors_json or "[]")),
            fb_tag_poetry="Стихи" in tags,
        ), "\n\n".join(parts)


# ---------------------------------------------------------------- дедупликация
def near_dup_clusters(sigs: np.ndarray, has_sig: np.ndarray, dd: dict, log) -> tuple[np.ndarray, dict]:
    """LSH-бандинг по сигнатурам, проверка кандидатов по оценке Жаккара, union-find."""
    n = len(sigs)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    idx = np.where(has_sig)[0]
    b = dd["lsh_bands"]
    r = dd["num_perm"] // b
    checked, confirmed, skipped_buckets = set(), 0, 0
    for band in range(b):
        sl = np.ascontiguousarray(sigs[idx, band * r:(band + 1) * r])
        keys = np.array([xxhash.xxh64_intdigest(row.tobytes()) for row in sl], dtype=np.uint64)
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        for grp in np.split(order, bounds):
            if len(grp) < 2:
                continue
            if len(grp) > dd["max_bucket"]:
                skipped_buckets += 1
                continue
            members = idx[grp]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, c = int(members[i]), int(members[j])
                    if (a, c) in checked:
                        continue
                    checked.add((a, c))
                    jac = float(np.mean(sigs[a] == sigs[c]))
                    if jac >= dd["jaccard_threshold"]:
                        confirmed += 1
                        ra, rc = find(a), find(c)
                        if ra != rc:
                            parent[max(ra, rc)] = min(ra, rc)
        log(f"  LSH band {band + 1}/{b}: candidates checked={len(checked)}, confirmed={confirmed}")
    roots = np.array([find(i) for i in range(n)])
    return roots, dict(pairs_checked=len(checked), pairs_confirmed=confirmed, buckets_skipped=skipped_buckets)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--full", action="store_true", help="полный корпус (sample_frac=1.0)")
    ap.add_argument("--force", action="store_true", help="пересчитать, игнорируя кэш")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.full:
        cfg["sample_frac"] = 1.0
    frac = cfg["sample_frac"]
    # хешируем только разделы, влияющие на манифест (остальные разделы конфига — этапы 0b–0d)
    chash = config_hash({k: cfg[k] for k in ("seed", "sample_frac", "sources", "dedup", "verse")})
    suffix = "" if frac >= 1 else f"_sample{frac:g}"
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    f_man = out / f"manifest{suffix}.parquet"
    f_ch = out / f"ficbook_chapters{suffix}.parquet"
    f_sig = out / f"minhash{suffix}.npy"
    f_meta = out / f"manifest{suffix}.meta.json"
    if not args.force and f_meta.exists() and json.loads(f_meta.read_text()).get("config_hash") == chash:
        print(f"Кэш актуален ({chash}): {f_man}")
        return

    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    chapter_rows: list = []
    metas: list[dict] = []

    def gen():
        for it in (iter_afanasyev(cfg, frac), iter_fairytales(cfg, frac), iter_proza(cfg, frac), iter_ficbook(cfg, frac, chapter_rows)):
            for meta, text in it:
                metas.append(meta)
                if len(metas) % 20000 == 0:
                    log(f"  прочитано {len(metas)} текстов ({meta['corpus']})")
                yield len(metas) - 1, text

    stats: dict[int, dict] = {}
    sigs_d: dict[int, np.ndarray] = {}
    # Пакетами по batch_texts: Pool.imap выбирает генератор жадно и поднял бы весь корпус в память.
    batch_n = 4000
    with Pool(cfg["n_workers"], initializer=_init, initargs=(cfg,)) as pool:
        it = gen()
        while True:
            batch = [x for _, x in zip(range(batch_n), it)]
            if not batch:
                break
            for idx, st, sig in pool.imap_unordered(text_stats, batch, chunksize=8):
                stats[idx] = st
                if sig is not None:
                    sigs_d[idx] = sig
    log(f"обработано текстов: {len(metas)}")

    df = pd.DataFrame(metas)
    st_df = pd.DataFrame.from_dict(stats, orient="index").sort_index()
    df = pd.concat([df, st_df], axis=1)
    n = len(df)
    sigs = np.zeros((n, cfg["dedup"]["num_perm"]), dtype=np.uint32)
    has_sig = np.zeros(n, dtype=bool)
    for i, s in sigs_d.items():
        sigs[i] = s
        has_sig[i] = True

    # одна и та же запись, выгруженная в две базы (напр. 2018 и 2022-2023): оставляем первую
    keep = ~df["doc_id"].duplicated()
    n_same_id = int((~keep).sum())
    log(f"повторы doc_id между базами (удалены): {n_same_id}")
    df, sigs, has_sig = df[keep.values].reset_index(drop=True), sigs[keep.values], has_sig[keep.values]
    n = len(df)

    # точные дубли: одинаковый нормализованный текст; «канонический» — первый в порядке манифеста
    nonempty = df["n_words"] > 0
    df["exact_dup_group"] = df.groupby("sha1_norm").ngroup().where(nonempty)
    first = df[nonempty].groupby("sha1_norm").cumcount()
    df["is_exact_dup"] = False
    df.loc[first.index, "is_exact_dup"] = first.values > 0
    log(f"точных дублей (кроме первого экземпляра): {int(df['is_exact_dup'].sum())}")

    roots, lsh_info = near_dup_clusters(sigs, has_sig, cfg["dedup"], log)
    df["near_dup_cluster"] = roots
    csize = pd.Series(roots).map(pd.Series(roots).value_counts())
    df["near_dup_cluster_size"] = csize.values
    # в кластере оставляем самый длинный текст
    rank = df.assign(_neg=-df["n_words"]).sort_values(["near_dup_cluster", "_neg", "doc_id"]) \
             .groupby("near_dup_cluster").cumcount()
    df["is_near_dup"] = (rank.reindex(df.index) > 0) & (df["near_dup_cluster_size"] > 1)
    log(f"почти-дублей (кроме представителя кластера): {int(df['is_near_dup'].sum())}")

    df.to_parquet(f_man, index=False)
    pd.DataFrame(chapter_rows).to_parquet(f_ch, index=False)
    np.save(f_sig, sigs)
    f_meta.write_text(json.dumps(dict(config_hash=chash, sample_frac=frac, n_docs=n, same_id_dropped=n_same_id, lsh=lsh_info,
                                      seconds=round(time.time() - t0, 1)), ensure_ascii=False, indent=2))
    log(f"записано: {f_man} ({n} строк)")


if __name__ == "__main__":
    main()
