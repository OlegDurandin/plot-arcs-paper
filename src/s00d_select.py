"""Этап 0d. Аналитическая выборка по решениям Ворот 0.

Единицы анализа (колонка unit):
  text          — самостоятельный текст одной записью (сказка, рассказ, новелла, повесть/роман целиком, фанфик целиком);
  linked_work   — повесть/роман Проза.ру, склеенные из глав (src/s00b_link_proza.py);
  chapter       — глава (Проза.ру: записи-фрагменты рубрик «роман/повесть»; Ficbook: главы) — для RQ3.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc = cfg["selection"]
    inter = ROOT / cfg["out_dir"]
    d = pd.read_parquet(inter / "manifest.parquet")
    frag_re = re.compile(sc["fragment_title_re"])
    d["fragment_title"] = d["title"].fillna("").map(lambda t: bool(frag_re.search(t)))
    w = pd.read_parquet(inter / "proza_works.parquet")
    wm = pd.read_parquet(inter / "proza_work_members.parquet")
    linked_docs = set(wm["doc_id"])

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
    pz_long = (d["corpus"] == "proza") & d["form"].isin(["novel", "povest"])
    # записи-главы романов/повестей — не самостоятельные тексты; они идут в unit=chapter
    is_chapter = pz_long & (d["fragment_title"] | d["doc_id"].isin(linked_docs))
    excl(is_chapter, "глава (см. unit=chapter / linked_work)")
    # фрагменты в рубриках рассказ/новелла тоже не самостоятельные тексты
    excl((d["corpus"] == "proza") & ~pz_long & d["fragment_title"], "фрагмент по заголовку")
    excl(d["n_words"] < sc["min_words"], f"короче {sc['min_words']} слов")
    d["exclude_reason"] = reasons
    d["include"] = reasons == ""
    d["unit"] = "text"

    keep_cols = ["doc_id", "unit", "corpus", "form", "author_id", "title", "date", "n_words", "likes", "tags",
                 "lang_orig", "is_translation", "ft_kind", "fb_chapters_ok", "fb_date_source", "fb_likes_source",
                 "text_path", "include", "exclude_reason"]
    texts = d[keep_cols]

    # склеенные произведения Проза.ру
    wsel = w[w[sc["linked_works"]]].copy()
    vv = d.set_index("doc_id")[["verse_like", "is_near_dup", "date"]]
    mm = wm.merge(vv, left_on="doc_id", right_index=True, how="left")
    agg = mm.groupby("work_id").agg(verse_share=("verse_like", "mean"), date=("date", "min"))
    wsel = wsel.drop(columns=["verse_share"]).merge(agg, left_on="work_id", right_index=True, how="left")
    lw = pd.DataFrame(dict(
        doc_id=wsel["work_id"], unit="linked_work", corpus="proza", form=wsel["form"], author_id=wsel["author_id"],
        title=wsel["stem"], date=wsel["date"], n_words=wsel["n_words"], likes=None, tags=None, lang_orig="ru",
        is_translation=False, ft_kind=None, fb_chapters_ok=wsel["n_records"], fb_date_source=None, fb_likes_source=None,
        text_path="members:data/interim/proza_work_members.parquet", complete_marked=wsel["complete_marked"].values))
    lr = pd.Series("", index=lw.index)
    lr[(wsel["verse_share"] > 0.5).values] = "стихоподобный"
    lr[(lr == "") & (lw["n_words"] < sc["min_words"])] = f"короче {sc['min_words']} слов"
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
    ch = ch.drop(columns=["verse_like"])

    out = pd.concat([texts, lw, ch], ignore_index=True)
    # воспроизводимый ранг внутри (автор, форма, единица) — для лимита max_per_author_form
    rng = np.random.default_rng(cfg["seed"])
    out["_r"] = rng.random(len(out))
    out["author_rank"] = (out[out["include"]].sort_values("_r")
                          .groupby(["author_id", "form", "unit"]).cumcount())
    out = out.drop(columns="_r")
    out["within_author_cap"] = out["include"] & (out["author_rank"] < sc["max_per_author_form"])
    out.to_parquet(ROOT / sc["out"], index=False)

    inc = out[out["include"]]
    summ = inc.groupby(["unit", "corpus", "form"]).agg(
        единиц=("doc_id", "size"), авторов=("author_id", "nunique"), медиана_слов=("n_words", "median"),
        в_лимите_20=("within_author_cap", "sum"))
    print(summ.to_string())
    print("\nПричины исключения (unit=text):")
    print(out[out["unit"] == "text"].groupby(["corpus", "exclude_reason"]).size().to_string())
    print("записано:", ROOT / sc["out"])


if __name__ == "__main__":
    main()
