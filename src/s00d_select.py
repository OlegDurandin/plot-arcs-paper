"""Этап 0d. Аналитическая выборка по решениям Ворот 0.

Единицы анализа (колонка unit):
  text          — самостоятельный текст одной записью (сказка, рассказ, новелла, повесть/роман целиком, фанфик целиком);
  linked_work   — повесть/роман Проза.ру, склеенные из глав (src/s00b_link_proza.py);
  chapter       — глава (Проза.ру: фрагменты и записи «романов/повестей» короче порога; Ficbook: главы) — для RQ3.

Правила (configs/s00_inventory.yaml, раздел selection):
  * ≥ min_words слов, без стихоподобных, без дублей; Ficbook — ориджиналы, завершённые, без пропущенных глав;
  * повесть/роман одной записью — только от long_form_min_words; короче или с «продолжение следует» -> глава;
  * форма склеенной серии — по суммарной длине теми же порогами (короче порога повести -> short_series, вне анализа);
  * is_complete (true/false/unknown) для каждой единицы; strict_long_form — строгие пороги для проверки устойчивости;
  * Афанасьев: af_type по номеру; include_sens400 — набор для проверки чувствительности (порог 400).

Запуск:
    .venv/bin/python src/s00d_select.py configs/s00_inventory.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402


def af_type_map(types: dict) -> dict[int, str]:
    out = {}
    for t, ranges in types.items():
        for a, b in ranges:
            for n in range(a, b + 1):
                out[n] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc = cfg["selection"]
    thr, strict = sc["long_form_min_words"], sc["long_form_strict_min_words"]
    inter = ROOT / cfg["out_dir"]
    d = pd.read_parquet(inter / "manifest.parquet")
    frag_re = re.compile(sc["fragment_title_re"])
    d["fragment_title"] = d["title"].fillna("").map(lambda t: bool(frag_re.search(t)))
    w = pd.read_parquet(inter / "proza_works.parquet")
    wm = pd.read_parquet(inter / "proza_work_members.parquet")
    wsel = w[w[sc["linked_works"]]].copy()
    linked_docs = set(wm["doc_id"])            # все записи, попавшие в любую серию (≥ 3 записей)
    afmap = af_type_map(cfg["afanasyev_types"])
    d["af_type"] = d["af_no"].map(afmap) if "af_no" in d else None

    reasons = pd.Series("", index=d.index)

    def excl(mask, why):
        m = mask & (reasons == "")
        reasons[m] = why

    excl(d["n_words"] == 0, "пустой")
    excl(d["is_exact_dup"], "точный дубль")
    excl(d["is_near_dup"], "почти-дубль")
    if sc["exclude_verse_like"]:
        excl(d["verse_like"], "стихоподобный")
    excl((d["corpus"] == "afanasyev") & ~d["lang_orig"].isin(sc["afanasyev_langs"]), "Афанасьев: не русский")
    excl((d["corpus"] == "fairytales") & ~d["ft_kind"].isin(sc["fairytales_kinds"]), "fairytalez: басня/литературная")
    fbc = sc["ficbook"]
    fb = d["corpus"] == "ficbook"
    if fbc["originals_only"]:
        excl(fb & (d["fb_is_original"] != True), "Ficbook: не ориджинал")  # noqa: E712
    excl(fb & (d["fb_status"] != fbc["status"]), "Ficbook: не завершён")
    if fbc["no_missing_chapters"]:
        excl(fb & (d["fb_chapters_missing"] > 0), "Ficbook: пропущены главы")
    pz = d["corpus"] == "proza"
    pz_long = pz & d["form"].isin(["novel", "povest"])
    long_thr = d["form"].map(thr)
    # записи «романов/повестей», которые не являются целым произведением -> unit=chapter
    sel_members = set(wm.loc[wm["work_id"].isin(wsel["work_id"]), "doc_id"])
    # запись любой рубрики, вошедшая в отобранную серию, — глава этой серии;
    # запись «романа/повести» из любой серии (в т.ч. не прошедшей linked_ok) — тоже глава
    ch_linked = (pz_long & d["doc_id"].isin(linked_docs)) | d["doc_id"].isin(sel_members)
    ch_title = pz_long & d["fragment_title"]
    ch_cont = pz_long & d["tail_continuation"].fillna(False)
    ch_short = pz_long & (d["n_words"] < long_thr)
    excl(ch_linked, "глава: входит в серию (unit=linked_work/chapter)")
    excl(ch_title, "глава: заголовок-фрагмент")
    excl(ch_cont, "глава: «продолжение следует»")
    excl(ch_short, "глава: короче порога повести/романа")
    is_chapter = ch_linked | ch_title | ch_cont | ch_short
    excl(pz & ~pz_long & d["fragment_title"], "фрагмент по заголовку")
    below = d["n_words"] < sc["min_words"]
    excl(below, f"короче {sc['min_words']} слов")
    d["exclude_reason"] = reasons
    d["include"] = reasons == ""
    d["unit"] = "text"
    d["strict_long_form"] = pz_long & (d["n_words"] >= d["form"].map(strict))
    # Афанасьев, проверка чувствительности: всё то же, но порог 400
    sens = sc["min_words_sensitivity"]["afanasyev"]
    only_len = (reasons == f"короче {sc['min_words']} слов")
    d["include_sens400"] = (d["corpus"] == "afanasyev") & (d["include"] | (only_len & (d["n_words"] >= sens)))

    keep_cols = ["doc_id", "unit", "corpus", "form", "author_id", "title", "date", "n_words", "likes", "tags",
                 "lang_orig", "is_translation", "ft_kind", "af_no", "af_type", "fb_chapters_ok", "fb_date_source",
                 "fb_likes_source", "is_complete", "is_complete_source", "strict_long_form", "include_sens400",
                 "text_path", "include", "exclude_reason"]
    texts = d[keep_cols].copy()

    # склеенные произведения Проза.ру
    vv = d.set_index("doc_id")[["verse_like", "date"]]
    mm = wm.merge(vv, left_on="doc_id", right_index=True, how="left")
    agg = mm.groupby("work_id").agg(verse_share=("verse_like", "mean"), date=("date", "min"))
    wsel = wsel.drop(columns=["verse_share"]).merge(agg, left_on="work_id", right_index=True, how="left")
    lw = pd.DataFrame(dict(
        doc_id=wsel["work_id"].values, unit="linked_work", corpus="proza", form=wsel["form"].values,
        form_rubric=wsel["form_rubric"].values, author_id=wsel["author_id"].values, title=wsel["stem"].values,
        date=wsel["date"].values, n_words=wsel["n_words"].values, lang_orig="ru", is_translation=False,
        fb_chapters_ok=wsel["n_records"].values, is_complete=wsel["is_complete"].values,
        is_complete_source=wsel["is_complete_source"].values,
        strict_long_form=(wsel["n_words"] >= wsel["form"].map(strict)).fillna(False).values,
        include_sens400=False, text_path="members:data/interim/proza_work_members.parquet"))
    lr = pd.Series("", index=lw.index)
    lr[(wsel["verse_share"] > 0.5).values] = "стихоподобный"
    lr[(lr == "") & (lw["form"] == "short_series")] = "серия короче порога повести"
    lw["exclude_reason"] = lr.values
    lw["include"] = lw["exclude_reason"] == ""

    # главы для RQ3
    ch_pz = d[is_chapter & (d["n_words"] > 0) & ~d["is_exact_dup"]][["doc_id", "form", "author_id", "title", "date",
                                                                      "n_words", "text_path", "verse_like"]]
    ch_pz = ch_pz.assign(unit="chapter", corpus="proza", parent=ch_pz["doc_id"].map(wm.set_index("doc_id")["work_id"]))
    fbch = pd.read_parquet(inter / "ficbook_chapters.parquet")
    ok_works = d.loc[fb & d["include"], ["doc_id", "author_id", "date"]]
    fbch = fbch[fbch["ch_ok"]].merge(ok_works, on="doc_id")
    ch_fb = pd.DataFrame(dict(doc_id=fbch["doc_id"] + "#" + fbch["chapter_number"].astype(str), unit="chapter",
                              corpus="ficbook", form="fanfic", author_id=fbch["author_id"], title=fbch["ch_title"],
                              date=fbch["ch_date"], n_words=fbch["n_words"], text_path=fbch["text_path"],
                              verse_like=False, parent=fbch["doc_id"]))
    ch = pd.concat([ch_pz, ch_fb], ignore_index=True)
    cr = pd.Series("", index=ch.index)
    cr[ch["verse_like"].fillna(False).astype(bool)] = "стихоподобный"
    cr[(cr == "") & (ch["n_words"] < sc["min_words"])] = f"короче {sc['min_words']} слов"
    ch["exclude_reason"] = cr
    ch["include"] = cr == ""
    ch["is_complete"] = "n/a"
    ch = ch.drop(columns=["verse_like"])

    out = pd.concat([texts, lw, ch], ignore_index=True)

    # проверка: глава склеенной серии не попадает в выборку одиночных текстов
    inc_text = set(out.loc[(out["unit"] == "text") & out["include"], "doc_id"])
    inc_works = set(lw.loc[lw["include"], "doc_id"])
    members_inc = set(wm.loc[wm["work_id"].isin(inc_works), "doc_id"])
    overlap = inc_text & members_inc
    assert not overlap, f"{len(overlap)} глав склеенных серий попали в одиночные тексты"
    long_inc = set(out.loc[(out["unit"] == "text") & out["include"] & out["form"].isin(["povest", "novel"]), "doc_id"])
    assert not (long_inc & linked_docs), "повесть/роман из серии попали в одиночные тексты"
    print(f"записей рубрик «рассказ/новелла» из серий, не прошедших linked_ok, оставлены текстами: {len(inc_text & linked_docs)}")
    print(f"проверка пересечения: 0 из {len(members_inc)} глав склеенных серий среди {len(inc_text)} одиночных текстов")

    # воспроизводимый ранг внутри (автор, форма, единица) — для лимита max_per_author_form
    rng = np.random.default_rng(cfg["seed"])
    out["_r"] = rng.random(len(out))
    out["author_rank"] = (out[out["include"]].sort_values("_r").groupby(["author_id", "form", "unit"]).cumcount())
    out = out.drop(columns="_r")
    out["within_author_cap"] = out["include"] & (out["author_rank"] < sc["max_per_author_form"])
    for col in ["strict_long_form", "include_sens400"]:
        out[col] = out[col].fillna(False).astype(bool)
    out.to_parquet(ROOT / sc["out"], index=False)

    inc = out[out["include"]]
    summ = inc.groupby(["unit", "corpus", "form"]).agg(
        единиц=("doc_id", "size"), авторов=("author_id", "nunique"), медиана_слов=("n_words", "median"),
        в_лимите_20=("within_author_cap", "sum"))
    print(summ.to_string())
    print("\nis_complete (include, без глав):")
    print(pd.crosstab([inc.loc[inc.unit != "chapter", "unit"], inc.loc[inc.unit != "chapter", "form"]],
                      inc.loc[inc.unit != "chapter", "is_complete"]).to_string())
    print("\nпричины исключения (unit=text, Проза.ру):")
    print(out[(out["unit"] == "text") & (out["corpus"] == "proza")].groupby("exclude_reason").size().to_string())
    print("записано:", ROOT / sc["out"])


if __name__ == "__main__":
    main()
