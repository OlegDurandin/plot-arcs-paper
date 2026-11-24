"""Этап 0b. Склейка глав Проза.ру (рубрики «роман», «повесть») в произведения.

Связи «глава → произведение» в данных нет; восстанавливаем эвристически:
  1) серия = автор + основа заголовка (первые k слов после удаления маркеров глав и номеров);
  2) из заголовка извлекаются том / часть / глава (последний уровень — номер главы);
  3) порядок — по (том, часть, глава), при отсутствии номеров — по времени публикации;
  4) флаги полноты: есть начало, номера без пропусков, есть маркер конца, номера согласуются со временем.

Запуск:
    .venv/bin/python src/s00b_link_proza.py configs/s00_inventory.yaml
Артефакты:
    data/interim/proza_works.parquet          — серия = кандидат в произведение (+ флаги)
    data/interim/proza_work_members.parquet   — записи серии в порядке чтения
    reports/00b_proza_linking_review.md       — случайные серии для ручной проверки
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

ORD = {"первая": 1, "первой": 1, "вторая": 2, "второй": 2, "третья": 3, "третьей": 3, "четвертая": 4,
       "четвёртая": 4, "пятая": 5, "шестая": 6, "седьмая": 7, "восьмая": 8, "девятая": 9, "десятая": 10,
       "первый": 1, "второй": 2, "третий": 3}
ROMAN = re.compile(r"^[ivxlc]+$")


def to_num(tok: str, cap: int) -> int | None:
    tok = tok.lower().strip(".-)")
    if tok.isdigit():
        v = int(tok)
    elif tok in ORD:
        v = ORD[tok]
    elif ROMAN.match(tok):
        vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
        v, prev = 0, 0
        for ch in reversed(tok):
            x = vals[ch]
            v = v - x if x < prev else v + x
            prev = max(prev, x)
    else:
        return None
    return v if 0 <= v <= cap else None


NUMTOK = r"(\d+|[ivxlc]+\b|первая|первой|вторая|второй|третья|третьей|четв[её]ртая|пятая|шестая|седьмая|восьмая|девятая|десятая|первый|третий)"
LEVELS = {
    "vol": re.compile(r"(?<!\w)(?:том|книга|кн\.)\s*" + NUMTOK, re.I),
    "part": re.compile(r"(?<!\w)(?:часть|части|ч\.)\s*" + NUMTOK, re.I),
    "ch": re.compile(r"(?<!\w)(?:глава|главы|гл\.?|продолжение|эпизод|серия|выпуск)\s*" + NUMTOK, re.I),
}
LEAD = re.compile(r"^\s*(\d+)(?:[\s.)-]|$)")
TRAIL = re.compile(r"(?:^|\s)(\d+)\s*$")
START_RE = re.compile(r"пролог|предисловие|начало\b", re.I)
END_RE = re.compile(r"эпилог|окончание|конец\b|финал|заключени|заключительн|послесловие", re.I)
STRIP = re.compile(r"(?<!\w)(?:том|книга|кн\.|часть|части|ч\.|глава|главы|гл\.?|продолжение|окончание|начало|эпизод|"
                   r"серия|выпуск|пролог|эпилог|послесловие|предисловие|финал)\s*" + NUMTOK + "?", re.I)


def parse_title(t: str, cap: int) -> dict:
    t = t or ""
    out = {}
    for lvl, rx in LEVELS.items():
        ms = rx.findall(t)
        out[lvl] = to_num(ms[-1], cap) if ms else None
    if out["ch"] is None and out["part"] is None:
        m = LEAD.match(t) or TRAIL.search(t)
        out["ch"] = to_num(m.group(1), cap) if m else None
    out["is_start"] = bool(START_RE.search(t))
    out["is_end"] = bool(END_RE.search(t))
    return out


def stem(t: str, k: int) -> str:
    t = STRIP.sub(" ", (t or "").lower().replace("ё", "е"))
    return " ".join(re.sub(r"[\W\d_]+", " ", t).split()[:k])


def contiguous(levels: pd.DataFrame) -> tuple[bool, bool]:
    """(номера без пропусков, есть начало) с учётом уровней «часть → глава»."""
    lv = levels.dropna(subset=["ch"]) if levels["ch"].notna().any() else levels.dropna(subset=["part"]).assign(ch=lambda x: x["part"], part=np.nan)
    if len(lv) < 2:
        return False, False
    if lv["part"].notna().any() and lv["part"].nunique() > 1:
        ok, start = True, False
        parts = sorted(lv["part"].dropna().unique())
        ok &= parts[-1] - parts[0] + 1 == len(parts)
        for i, (p, g) in enumerate(lv.groupby("part")):
            nums = sorted(set(g["ch"].astype(int)))
            if nums[-1] - nums[0] + 1 != len(nums):
                ok = False
            if i == 0 and nums[0] <= 1 and p <= 1:
                start = True
        # главы могут идти сквозной нумерацией через части — это тоже «без пропусков»
        allnums = sorted(set(lv["ch"].astype(int)))
        if not ok and allnums[-1] - allnums[0] + 1 == len(allnums):
            ok = True
            start = start or allnums[0] <= 1
        return ok, start
    nums = sorted(set(lv["ch"].astype(int)))
    return nums[-1] - nums[0] + 1 == len(nums), nums[0] <= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    lc = cfg["proza_linking"]
    cap = lc["max_chapter_number"]
    d = pd.read_parquet(ROOT / cfg["out_dir"] / "manifest.parquet",
                        columns=["doc_id", "corpus", "form", "author_id", "title", "n_words", "pz_published",
                                 "is_exact_dup", "verse_like", "tail_continuation", "tail_end_marker", "tail_partial_end"])
    d = d[(d["corpus"] == "proza") & d["form"].isin(lc["include_forms"]) & ~d["is_exact_dup"]]
    authors = set(d.loc[d["form"].isin(lc["forms"]), "author_id"])
    d = d[d["author_id"].isin(authors)].copy()
    d["stem"] = [stem(t, lc["stem_words"]) for t in d["title"]]
    d = d[d["stem"].str.len() >= lc["min_stem_chars"]]
    d["key"] = d["author_id"] + "|" + d["stem"]
    keys = set(d.loc[d["form"].isin(lc["forms"]), "key"])
    d = d[d["key"].isin(keys)]
    # записи тех же серий, недоступные на сайте (автор закрыл страницу / удалено / не найдено / не загружено):
    # их нет в манифесте, но заголовок и автор известны из списков за день -> пропущенные части серии
    gone = []
    for rel in cfg["sources"]["proza"]["dbs"]:
        con = sqlite3.connect(f"file:{ROOT / rel}?immutable=1", uri=True)
        gone += con.execute("select 'pz:' || author_login, title, coalesce(error, 'не загружено') from stories "
                            "where status is null or status != 200").fetchall()
        con.close()
    gone = pd.DataFrame(gone, columns=["author_id", "title", "error"])
    gone = gone[gone["author_id"].isin(authors)]
    gone["key"] = gone["author_id"] + "|" + [stem(t, lc["stem_words"]) for t in gone["title"]]
    site_missing = gone[gone["key"].isin(keys)].groupby("key").size()
    print(f"недоступных на сайте записей в сериях-кандидатах: {int(site_missing.sum())} в {len(site_missing)} сериях")

    parsed = pd.DataFrame([parse_title(t, cap) for t in d["title"]], index=d.index)
    d = pd.concat([d, parsed], axis=1)

    works, members = [], []
    for key, g in d.groupby("key", sort=False):
        if len(g) < lc["min_records"]:
            continue
        g = g.sort_values("pz_published")
        g["t_rank"] = np.arange(len(g))
        # повторно опубликованная глава (тот же номер): оставляем последнюю публикацию
        num_cols = ["vol", "part", "ch"]
        has_num = g[num_cols].notna().any(axis=1)
        dup = g[has_num].duplicated(subset=num_cols, keep="last")
        g["dup_chapter"] = False
        g.loc[dup[dup].index, "dup_chapter"] = True
        gg = g[~g["dup_chapter"]]
        contig, start_num = contiguous(gg[num_cols])
        # начало второй книги / третьей части — не начало произведения
        if (gg["vol"].min() or 1) > 1 or (gg["part"].notna().any() and gg["part"].min() > 1):
            start_num = False
        # согласие номеров с порядком публикации
        nn = gg[gg[num_cols].notna().any(axis=1)]
        if len(nn) >= 3:
            seq = nn[num_cols].fillna(0).apply(tuple, axis=1).rank(method="dense")
            corr = float(pd.Series(seq.values).corr(pd.Series(nn["t_rank"].values), method="spearman"))
        else:
            corr = np.nan
        order_by_num = len(nn) == len(gg) and not np.isnan(corr)
        gg = gg.sort_values(num_cols + ["pz_published"], na_position="last") if order_by_num else gg
        lf = g[g["form"].isin(lc["forms"])]
        wid = "pzw:" + key
        last = gg.iloc[-1]
        if last["tail_continuation"]:
            complete, csrc = "false", "continuation_marker"
        elif last["tail_partial_end"]:
            complete, csrc = "false", "partial_end_marker"
        elif gg["is_end"].any() or last["tail_end_marker"]:
            complete, csrc = "explicit", "end_marker"
        else:
            complete, csrc = "unknown", "no_marker"
        total = int(gg["n_words"].sum())
        thr = cfg["selection"]["long_form_min_words"]
        form_len = "novel" if total >= thr["novel"] else "povest" if total >= thr["povest"] else "short_series"
        works.append(dict(
            work_id=wid, author_id=gg["author_id"].iat[0], stem=gg["stem"].iat[0],
            form=form_len, form_rubric=lf["form"].mode().iat[0], forms=",".join(sorted(set(gg["form"]))),
            is_complete=complete, is_complete_source=csrc,
            n_records=len(gg), n_dup_chapters=int(g["dup_chapter"].sum()),
            n_numbered=len(nn), n_words=int(gg["n_words"].sum()),
            first=gg["pz_published"].min(), last=gg["pz_published"].max(),
            has_start=bool(start_num or gg["is_start"].any()), contiguous=bool(contig),
            has_end=bool(gg["is_end"].any()), order_corr=corr,
            order_consistent=bool(not np.isnan(corr) and corr >= lc["order_rank_corr_min"]),
            ordered_by="number" if order_by_num else "time",
            verse_share=float(gg["verse_like"].mean()),
            n_site_missing=int(site_missing.get(key, 0)),
        ))
        for i, r in enumerate(gg.itertuples()):
            members.append(dict(work_id=wid, pos=i, doc_id=r.doc_id, title=r.title, form=r.form,
                                vol=r.vol, part=r.part, ch=r.ch, published=r.pz_published, n_words=r.n_words))
    w = pd.DataFrame(works)
    m = pd.DataFrame(members)
    # пропуски: дыры в нумерации или части, недоступные на сайте
    w["has_gaps"] = ~w["contiguous"] | (w["n_site_missing"] > 0)
    w["linked_ok"] = w["has_start"] & w["contiguous"] & w["order_consistent"]
    w["complete_marked"] = w["linked_ok"] & w["has_end"]
    out = ROOT / cfg["out_dir"]
    w.to_parquet(out / "proza_works.parquet", index=False)
    m.to_parquet(out / "proza_work_members.parquet", index=False)

    # отчёт + выборка для ручной проверки
    rep = ROOT / cfg["report_dir"] / "00b_proza_linking_review.md"
    L = ["# Этап 0b. Склейка глав Проза.ру — сводка и выборка для ручной проверки\n"]
    lf_total = int(d["form"].isin(lc["forms"]).sum())
    L.append(f"Записей рубрик {lc['forms']} у авторов-кандидатов: {lf_total}. Серий (≥ {lc['min_records']} записей): {len(w)}.\n")
    rows = []
    for name, mask in [("все серии", w.index == w.index), ("есть начало", w.has_start), ("номера без пропусков", w.contiguous),
                       ("порядок согласован с датами", w.order_consistent), ("**linked_ok** (все три)", w.linked_ok),
                       ("linked_ok + маркер конца", w.complete_marked)]:
        s = w[mask]
        rows.append(f"| {name} | {len(s)} | {s['author_id'].nunique()} | {int(s['n_records'].sum())} | "
                    f"{int(s['n_words'].median()) if len(s) else 0} |")
    L.append("| фильтр | серий | авторов | записей | медиана слов |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    ok = w[w["linked_ok"]]
    L.append("Форма серии — по суммарной длине (повесть ≥ {povest}, роман ≥ {novel} слов); рубрика — в `form_rubric`.\n"
             .format(**cfg["selection"]["long_form_min_words"]))
    L.append("| форма по длине | explicit | false | unknown | с пропусками (has_gaps) |\n|---|---|---|---|---|")
    for f, g in ok.groupby("form"):
        vc = g["is_complete"].value_counts()
        L.append(f"| {f} | {vc.get('explicit', 0)} | {vc.get('false', 0)} | {vc.get('unknown', 0)} | {int(g['has_gaps'].sum())} |")
    L.append("")
    for f in ["povest", "novel"]:
        s = w[w["linked_ok"] & (w["form"] == f)]
        L.append(f"- linked_ok, форма {f}: {len(s)} серий, {s['author_id'].nunique()} авторов, медиана {int(s['n_words'].median()) if len(s) else 0} слов, "
                 f"с маркером конца {int(s['has_end'].sum())}.")
    L.append("")
    samp = w[w["linked_ok"]].sample(min(lc["review_n"], int(w["linked_ok"].sum())), random_state=cfg["seed"])
    L.append(f"## Случайные {len(samp)} серий из linked_ok (seed {cfg['seed']})\n")
    L.append("Проверка: все ли записи — главы одного произведения, верен ли порядок, нет ли посторонних записей.\n")
    for i, r in enumerate(samp.itertuples(), 1):
        mm = m[m["work_id"] == r.work_id]
        L.append(f"### {i}. `{r.work_id}` — {r.form}, {r.n_records} зап., {r.n_words} сл., конец: {'да' if r.has_end else 'нет'}\n")
        L.append("\n".join(f"{x.pos + 1}. [{x.form}] {x.published[:10]} — {x.title} ({x.n_words})" for x in mm.itertuples()) + "\n")
    rep.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print("записано:", rep)


if __name__ == "__main__":
    main()
