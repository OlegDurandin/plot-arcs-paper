"""Этап 0. Отчёт по инвентаризации: таблицы, графики, reports/00_inventory.md.

Запуск:
    .venv/bin/python src/s00_report.py configs/s00_inventory.yaml [--full]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

FORMS = ["tale_ru", "tale", "story", "novella", "povest", "novel", "fanfic"]
FORM_RU = {"tale_ru": "сказка (Афанасьев)", "tale": "сказка (fairytalez, пер.)", "story": "рассказ", "novella": "новелла", "povest": "повесть",
           "novel": "роман", "fanfic": "фанфик (Ficbook)"}
PZ = ["story", "novella", "povest", "novel"]
# слоты 1–4 эталонной палитры (dataviz/references/palette.md), фиксированный порядок
SERIES = {"story": "#2a78d6", "novella": "#eb6834", "povest": "#1baf7a", "novel": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Признаки «это часть / глава», а не самостоятельное произведение (по заголовку)
CHAPTER_RE = re.compile(  # группы без захвата
    r"(?:глава|главы|част[ьи]|продолжени|окончани|начало|эпизод|серия|том\s*\d|книга\s*\d|"
    r"\bгл\.|\bч\.\s*\d|пролог|эпилог|^\s*\d+[\s.)-]|\s\d+\s*$|\b[ivxlc]+\s*$)",
    re.IGNORECASE)


def q(s, ps=(5, 25, 50, 75, 95)):
    return {f"p{p}": int(np.percentile(s, p)) for p in ps} if len(s) else {f"p{p}": None for p in ps}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (f"{v:,}".replace(",", " ") if isinstance(v, (int, np.integer)) else
                                                          (f"{v:.3g}" if isinstance(v, float) else str(v))) for v in r) + " |")
    return "\n".join(out)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def ovl(a, b, bins):
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    w = np.diff(bins)
    return float(np.sum(np.minimum(ha, hb) * w))


def best_threshold_acc(a, b):
    """Лучшая точность разделения двух форм одним порогом по log-длине (баланс классов)."""
    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    best = 0.5
    for t in np.quantile(x, np.linspace(0.01, 0.99, 197)):
        tpr = np.mean(b > t)
        tnr = np.mean(a <= t)
        best = max(best, (tpr + tnr) / 2, ((1 - tpr) + (1 - tnr)) / 2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    frac = 1.0 if args.full else cfg["sample_frac"]
    suffix = "" if frac >= 1 else f"_sample{frac:g}"
    inter = ROOT / cfg["out_dir"]
    rep = ROOT / cfg["report_dir"]
    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inter / f"manifest{suffix}.meta.json").read_text())
    d = pd.read_parquet(inter / f"manifest{suffix}.parquet")
    ch = pd.read_parquet(inter / f"ficbook_chapters{suffix}.parquet")
    d.loc[d["corpus"] == "afanasyev", "form"] = "tale_ru"
    d["form"] = pd.Categorical(d["form"], FORMS, ordered=True)
    d["is_dup"] = d["is_exact_dup"] | d["is_near_dup"]
    d["empty"] = d["n_words"] == 0
    clean = d[~d["is_dup"] & ~d["empty"]].copy()
    clean["lw"] = np.log10(clean["n_words"])
    L = []  # строки отчёта
    P = L.append
    title_sfx = "" if frac >= 1 else f" (ВЫБОРКА {frac:.0%})"

    P(f"# Этап 0. Инвентаризация корпусов{title_sfx}\n")
    P(f"Манифест: `{cfg['out_dir']}/manifest{suffix}.parquet`, хеш конфига `{meta['config_hash']}`, "
      f"строк: {meta['n_docs']}. Длина в словах — единый регулярный токенизатор (`src/common.py`), "
      "одинаковый для всех корпусов.\n")

    # ---------------- 1. Сводка
    P("## 1. Сводка по корпусам и формам\n")
    P("Исходные записи → после удаления пустых текстов, точных и почти-дублей (Жаккар ≥ 0,8 по 5-словным шинглам).\n")
    rows = []
    for (c, f), g in d.groupby(["corpus", "form"], observed=True):
        gc = g[~g["is_dup"] & ~g["empty"]]
        rows.append(dict(корпус=c, форма=FORM_RU[f], записей=len(g), пустых=int(g["empty"].sum()),
                         точн_дубли=int(g["is_exact_dup"].sum()), почти_дубли=int((g["is_near_dup"] & ~g["is_exact_dup"]).sum()),
                         чистых=len(gc), авторов=gc["author_id"].nunique(),
                         слов_млн=round(gc["n_words"].sum() / 1e6, 2),
                         доля_стихоподобных=round(float(gc["verse_like"].mean()), 3)))
    P(md_table(pd.DataFrame(rows)) + "\n")
    P("«Стихоподобный» — ≥ 8 непустых строк, из них ≥ 70 % короче 9 слов (эвристика из конфига; "
      "ловит и стихи, и диалоговые/«лесенкой» тексты, поэтому это верхняя оценка).\n")

    # ---------------- 2. Длины
    P("## 2. Распределение длин (слов), чистые тексты\n")
    rows = []
    for f, g in clean.groupby("form", observed=True):
        rows.append(dict(форма=FORM_RU[f], n=len(g), **q(g["n_words"]), среднее=int(g["n_words"].mean()),
                         макс=int(g["n_words"].max()),
                         **{"<400": round(float((g["n_words"] < 400).mean()), 3),
                            "<1000": round(float((g["n_words"] < 1000).mean()), 3)}))
    P(md_table(pd.DataFrame(rows)) + "\n")
    P("`<400` — доля текстов короче 400 слов: при N = 20 это меньше 20 слов на сегмент.\n")

    fig, axes = plt.subplots(len(FORMS), 1, figsize=(7, 9), sharex=True)
    bins = np.linspace(1, 6, 81)
    for ax, f in zip(axes, FORMS):
        x = clean.loc[clean["form"] == f, "lw"]
        ax.hist(x, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.4)
        if len(x):
            ax.axvline(np.median(x), color=INK, lw=1, ls="--")
            ax.text(np.median(x), ax.get_ylim()[1] * 0.85, f"  медиана {int(10 ** np.median(x))}",
                    fontsize=8, color=INK)
        ax.set_ylabel(FORM_RU[f], rotation=0, ha="right", va="center", fontsize=9, color=INK)
        style(ax)
    axes[-1].set_xticks(range(1, 7), ["10", "100", "1 000", "10 000", "100 000", "1 000 000"])
    axes[-1].set_xlabel("длина текста, слов (лог. шкала)", color=INK2, fontsize=9)
    fig.suptitle("Длины текстов по формам (без дублей)", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / f"00_length_hist{suffix}.png", dpi=150)
    plt.close(fig)
    P(f"![Длины по формам](figures/00_length_hist{suffix}.png)\n")

    # ---------------- 3. Проза.ру: форма vs длина
    P("## 3. Проза.ру: согласуется ли метка формы с длиной\n")
    P("Метка формы — это **рубрика сайта**, выбранная автором при публикации (`topic`: 02 рассказ, "
      "21 новелла, 01 повесть, 04 роман), а не отдельное поле «форма».\n")
    pz = clean[clean["corpus"] == "proza"]
    bins = np.linspace(1, 6, 101)
    rows = []
    for i, a in enumerate(PZ):
        for b in PZ[i + 1:]:
            xa = pz.loc[pz["form"] == a, "lw"].values
            xb = pz.loc[pz["form"] == b, "lw"].values
            if len(xa) and len(xb):
                rows.append(dict(пара=f"{FORM_RU[a]} – {FORM_RU[b]}", OVL=round(ovl(xa, xb, bins), 3),
                                 точность_порога=round(best_threshold_acc(xa, xb), 3)))
    P(md_table(pd.DataFrame(rows)) + "\n")
    P("OVL — площадь пересечения плотностей log-длины (1 = распределения совпадают). "
      "«Точность порога» — сбалансированная точность лучшего одного порога длины (0,5 = длина не различает формы).\n")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for f in PZ:
        x = np.sort(pz.loc[pz["form"] == f, "lw"].values)
        if not len(x):
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=SERIES[f], lw=2, label=FORM_RU[f])
    ax.set_xticks(range(1, 7), ["10", "100", "1 000", "10 000", "100 000", "1 000 000"])
    ax.set_xlabel("длина записи, слов (лог. шкала)", color=INK2, fontsize=9)
    ax.set_ylabel("доля записей ≤ x", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.suptitle("Проза.ру: накопленное распределение длины по рубрикам", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / f"00_proza_ecdf{suffix}.png", dpi=150)
    plt.close(fig)
    P(f"![ECDF Проза.ру](figures/00_proza_ecdf{suffix}.png)\n")

    # главы
    P("### 3.1. Записи-главы\n")
    pz = pz.assign(chapter_title=pz["title"].fillna("").str.contains(CHAPTER_RE))
    stem = (pz["title"].fillna("").str.lower()
            .str.replace(CHAPTER_RE, " ", regex=True)
            .str.replace(r"[\W\d_]+", " ", regex=True).str.strip().str[:25])
    pz["series_key"] = pz["author_id"] + "|" + stem
    ssize = pz.groupby("series_key")["doc_id"].transform("size")
    pz["in_series"] = (ssize >= 3) & (stem.str.len() >= 3)
    rows = []
    for f, g in pz.groupby("form", observed=True):
        rows.append(dict(форма=FORM_RU[f], записей=len(g),
                         заголовок_как_у_главы=round(float(g["chapter_title"].mean()), 3),
                         в_серии_от_3=round(float(g["in_series"].mean()), 3),
                         медиана_слов=int(g["n_words"].median()),
                         текстов_на_автора_медиана=float(g.groupby("author_id").size().median())))
    P(md_table(pd.DataFrame(rows)) + "\n")
    P("«Заголовок как у главы» — заголовок содержит «глава/часть/продолжение/…» или номер. "
      "«В серии» — у того же автора ≥ 3 записей с совпадающей основой заголовка (после удаления номеров и слов-маркеров). "
      "Связи «глава → произведение» в данных нет, поэтому обе оценки эвристические.\n")
    wf = inter / "proza_works.parquet"
    if frac >= 1 and wf.exists():
        w = pd.read_parquet(wf)
        P("**Склейка глав (этап 0b, `src/s00b_link_proza.py`).** Серия = автор + основа заголовка; "
          f"серий из ≥ 3 записей: {len(w)}; надёжно сшитых (`linked_ok`: есть начало, номера без пропусков, "
          f"порядок номеров согласован с датами): {int(w['linked_ok'].sum())} ({w.loc[w['linked_ok'], 'author_id'].nunique()} авторов, "
          f"медиана {int(w.loc[w['linked_ok'], 'n_words'].median())} слов); из них с маркером конца (эпилог/окончание/…): "
          f"{int(w['complete_marked'].sum())}. Ручная проверка 50 серий — `reports/00b_proza_linking_review.md`.\n")
    ex = pz[(pz["form"] == "novel")].sample(min(12, (pz["form"] == "novel").sum()), random_state=cfg["seed"])
    P("Случайные заголовки из рубрики «роман»:\n")
    P("\n".join(f"- {t} — {n} сл." for t, n in zip(ex["title"], ex["n_words"])) + "\n")

    # ---------------- 4. Сказки
    P("## 4. Сказки\n")
    af = d[d["corpus"] == "afanasyev"]
    afc = clean[clean["corpus"] == "afanasyev"]
    P("### 4.1. Афанасьев, «Народные русские сказки» (основная фольклорная база)\n")
    P(f"- Источник: Викитека, подстраницы «Народные русские сказки (Афанасьев)/…» (общественное достояние); "
      f"сырьё — `data/external/afanasyev/pages.jsonl`, разбор — `src/s00c_afanasyev.py`.")
    P(f"- Вариантов под номерами Афанасьева: {len(af)} (№ 1–579, без пропусков и повторов); пустых в Викитеке: {int(af['empty'].sum())}.")
    P(f"- Язык: {af['lang_orig'].value_counts().to_dict()} (украинские и белорусские варианты в сборнике есть; в базу идут только русские).")
    P(f"- Длина (чистые русские): {q(afc.loc[afc['lang_orig'] == 'ru', 'n_words'])}; ≥ 600 слов: "
      f"{int(((afc['lang_orig'] == 'ru') & (afc['n_words'] >= 600) & ~afc['verse_like']).sum())}.")
    P("- `author_id` = сюжет (подстраница Викитеки): варианты одного сюжета зависимы, кластеризуем по нему.\n")
    P("### 4.2. fairytalez.sqlite (мировой фольклор в машинном переводе; проверка)\n")
    ft = d[d["corpus"] == "fairytales"]
    ftc = clean[clean["corpus"] == "fairytales"]
    P(f"- Записей: {len(ft)}; у **всех** есть `url_en` — это русская версия англоязычного сайта fairytalez.com. "
      "Признак перевода восстанавливается тривиально: переведены все тексты, и переводились они **с английского** "
      "(в т.ч. «русские» сказки — например, сборник «Русская гирлянда», англ. Russian Garland, 1916).")
    P("- Качество перевода по ручному просмотру — машинное, с ошибками: «Кузнечик-гей» (англ. *the gay grasshopper*), "
      "автор «фонтан» (Лафонтен), «избушка-царевич» вместо имени героя.")
    P("- Народ восстанавливается из `tale_regions` (63 региона; у 27 % сказок больше одного региона, напр. nordic + scandinavian + danish).")
    P(f"- Стихоподобных среди чистых: {ftc['verse_like'].mean():.1%} (басни Лафонтена и пр.).")
    kinds = ftc["ft_kind"].value_counts().to_dict()
    P(f"- Тип текста (по списку авторов в конфиге): {kinds}. В базу идут только folk; басни (Эзоп, Лафонтен) и "
      "литературные сказки (Андерсен, Гауф, Уайльд, Перро, Пушкин…) исключаются.\n")
    reg = ftc.assign(r=ftc["ft_regions"].str.split(",")).explode("r")
    top = reg["r"].value_counts().head(15)
    P("Топ-15 регионов (сказка может входить в несколько):\n")
    P(md_table(top.rename_axis("регион").reset_index(name="сказок")) + "\n")
    coll = ftc["author_name"].value_counts().head(12)
    P("Крупнейшие «авторы» (на деле — сборники/собиратели; это поле пойдёт в `author_id`):\n")
    P(md_table(coll.rename_axis("сборник").reset_index(name="сказок")) + "\n")
    russian = ftc["ft_regions"].fillna("").str.contains(r"\b(?:russian|ukrainian|slavic)\b")
    P(f"Регионы russian / ukrainian / slavic: {int(russian.sum())} сказок, медиана {int(ftc.loc[russian, 'n_words'].median())} слов. "
      "Но и они — обратный перевод с английского.\n")

    # ---------------- 5. Ficbook
    P("## 5. Ficbook\n")
    fb = d[d["corpus"] == "ficbook"]
    fbc = clean[clean["corpus"] == "ficbook"]
    P(f"- Произведений в `ficbook_originals_2026.sqlite`: {len(fb)}; с хотя бы одной загруженной главой: {int((~fb['empty']).sum())}.")
    P(f"- Ориджиналы (фандом ровно «Ориджиналы»): {fb['fb_is_original'].mean():.1%}; остальное — фанфики по фандомам (BTS, Гарри Поттер…).")
    P(f"- Статус: {fb['fb_status'].value_counts(dropna=False).to_dict()}.")
    P(f"- Дата публикации: из карточки произведения {(fb['fb_date_source'] == 'work').mean():.1%}, по первой главе "
      f"{(fb['fb_date_source'] == 'first_chapter').mean():.1%}, нет {fb['fb_date_source'].isna().mean():.1%} (у пустых — без загруженных глав — даты нет).")
    P(f"- Лайки: из `works.likes` {(fb['fb_likes_source'] == 'work').mean():.1%}, из плашек страницы (`metadata_json.badges`) "
      f"{(fb['fb_likes_source'] == 'badges').mean():.1%}, нет {fb['likes'].isna().mean():.1%}. Правило «первое число после статуса» "
      "совпадает с `works.likes` у 19 199 из 19 199 произведений; NULL в `works.likes` — это «0» на странице.")
    P(f"- С пропущенными главами (ошибка загрузки): {(fb['fb_chapters_missing'] > 0).mean():.1%}.")
    P(f"- Тег «Стихи»: {fb['fb_tag_poetry'].mean():.1%}; стихоподобных по эвристике: {fbc['verse_like'].mean():.1%}.")
    yrs = pd.to_datetime(fb["date"], errors="coerce").dt.year.value_counts().sort_index()
    P(f"- Годы публикации: {yrs.to_dict()}.\n")
    P("**Устройство глав.** Одно произведение = каталог `texts/YYYY/MM/<work_id>/` с файлами `part-NNN.txt`, "
      "по файлу на главу; в БД — таблица `chapter_texts` (номер, дата, путь). Последняя «глава» с адресом "
      "`thanks-author-page` — служебная страница, в тексты не включена.\n")
    chok = ch[ch["ch_ok"]]
    nch = fbc["fb_chapters_ok"]
    rows = [dict(показатель="глав на произведение", **q(nch), доля_одноглавых=round(float((nch == 1).mean()), 3)),
            dict(показатель="слов в главе", **q(chok["n_words"].dropna()), доля_одноглавых=None),
            dict(показатель="слов в произведении", **q(fbc["n_words"]), доля_одноглавых=None)]
    P(md_table(pd.DataFrame(rows)) + "\n")
    fbo = fbc[fbc["fb_is_original"]]
    P(f"Лайки (ориджиналы, без NULL): {q(fbo['likes'].dropna())}.\n")
    old = ROOT / cfg["sources"]["ficbook"]["old_db"]
    if old.exists():
        con = sqlite3.connect(f"file:{old}?mode=ro", uri=True)
        con.execute(f"attach 'file:{ROOT / cfg['sources']['ficbook']['db']}?mode=ro' as n")
        n_old, n_both = con.execute("select count(*), sum(work_id in (select work_id from n.works)) from works").fetchone()
        has_text = con.execute("select count(*) from sqlite_master where name='chapter_texts'").fetchone()[0]
        con.close()
        P(f"Старая база `ficbook.sqlite`: {n_old} произведений, {n_both} из них есть в новой; таблицы текстов "
          f"{'есть' if has_text else 'нет'} — в манифест не включена.\n")

    # ---------------- 6. Дубли
    P("## 6. Дедупликация\n")
    P(f"- Повторы одной записи в двух базах Проза.ру (одинаковый `story_id`, удалены до дедупликации): {meta.get('same_id_dropped')}.")
    P(f"- Точные дубли (нормализованный текст совпадает), кроме первого экземпляра: {int(d['is_exact_dup'].sum())}.")
    P(f"- Почти-дубли (MinHash 128, LSH 16×8, Жаккар ≥ 0,8), кроме самого длинного в кластере: "
      f"{int((d['is_near_dup'] & ~d['is_exact_dup']).sum())}. Проверено пар-кандидатов: {meta['lsh']['pairs_checked']}, "
      f"пропущено переполненных бакетов: {meta['lsh']['buckets_skipped']}.")
    cl = d[d["near_dup_cluster_size"] > 1]
    cross = cl.groupby("near_dup_cluster")["corpus"].nunique()
    cross_form = cl.groupby("near_dup_cluster")["form"].nunique()
    P(f"- Кластеров почти-дублей: {cl['near_dup_cluster'].nunique()}; из них межкорпусных: {int((cross > 1).sum())}, "
      f"с разными формами: {int((cross_form > 1).sum())}.")
    multi_author = cl.groupby("near_dup_cluster")["author_id"].nunique()
    P(f"- Кластеров с разными авторами: {int((multi_author > 1).sum())} (возможные заимствования/перепосты).\n")
    big = cl.groupby("near_dup_cluster").agg(n=("doc_id", "size"), author=("author_id", "first"),
                                             title=("title", "first"), words=("n_words", "median"),
                                             forms=("form", lambda s: ",".join(sorted(set(map(str, s)))))) \
            .sort_values("n", ascending=False).head(10)
    P("Крупнейшие кластеры:\n")
    P(md_table(big.reset_index(drop=True)) + "\n")

    # ---------------- 7. Авторы
    P("## 7. Авторы и тексты на автора (чистые тексты)\n")
    rows = []
    for f, g in clean.groupby("form", observed=True):
        per = g.groupby("author_id").size().sort_values(ascending=False)
        k1 = max(1, int(np.ceil(len(per) * 0.01)))
        rows.append(dict(форма=FORM_RU[f], авторов=len(per), текстов=int(per.sum()),
                         **{f"на_автора_{k}": v for k, v in q(per.values, (50, 90, 99)).items()},
                         макс=int(per.max()), доля_топ1проц_авторов=round(float(per.head(k1).sum() / per.sum()), 3),
                         доля_авторов_с_1_текстом=round(float((per == 1).mean()), 3)))
    P(md_table(pd.DataFrame(rows)) + "\n")
    multi = clean[clean["corpus"] == "proza"].groupby("author_id")["form"].nunique()
    P(f"Авторов Проза.ру, пишущих в ≥ 2 рубриках: {int((multi > 1).sum())} из {len(multi)} "
      "(для смешанных моделей это хорошо: эффект формы оценивается и внутри автора).\n")

    # ---------------- 7a. Пороги длины
    P("## 7a. Сколько остаётся при разных минимальных длинах\n")
    P("Чистые тексты без стихоподобных; Ficbook — только ориджиналы. В ячейке: доля оставшихся текстов / число авторов.\n")
    base = clean[~clean["verse_like"] & ((clean["corpus"] != "ficbook") | clean["fb_is_original"].fillna(False).astype(bool))]
    thr = [200, 400, 600, 1000, 2000, 5000]
    rows = []
    for f, g in base.groupby("form", observed=True):
        r = dict(форма=FORM_RU[f], n=len(g))
        for t in thr:
            k = g[g["n_words"] >= t]
            r[f"≥{t}"] = f"{len(k) / len(g):.2f} / {k['author_id'].nunique()}"
        rows.append(r)
    P(md_table(pd.DataFrame(rows)) + "\n")
    chb = ch[ch["ch_ok"] & ch["doc_id"].isin(base.loc[base["corpus"] == "ficbook", "doc_id"])]
    P(f"Ficbook, если единица — глава: глав {len(chb)}, из них ≥ 1000 слов — {(chb['n_words'] >= 1000).mean():.2f}.\n")

    # ---------------- 7b. Аналитическая выборка (этап 0d)
    asf = ROOT / cfg["selection"]["out"]
    if frac >= 1 and asf.exists():
        a = pd.read_parquet(asf)
        inc = a[a["include"]]
        P("## 7b. Аналитическая выборка по решениям Ворот 0 (`data/interim/analysis_set.parquet`)\n")
        thr, strict = cfg["selection"]["long_form_min_words"], cfg["selection"]["long_form_strict_min_words"]
        P(f"Правила: ≥ {cfg['selection']['min_words']} слов, без стихоподобных и дублей; повесть одной записью ≥ {thr['povest']}, "
          f"роман ≥ {thr['novel']} слов (короче или с «продолжение следует» — глава); форма склеенной серии — по суммарной "
          "длине теми же порогами; Ficbook — завершённые ориджиналы без пропусков глав.\n")
        summ = inc.groupby(["unit", "corpus", "form"]).agg(
            единиц=("doc_id", "size"), авторов=("author_id", "nunique"), медиана_слов=("n_words", "median"),
            в_лимите_20=("within_author_cap", "sum")).reset_index()
        summ["медиана_слов"] = summ["медиана_слов"].astype(int)
        P(md_table(summ) + "\n")
        P("### Завершённость (`is_complete`)\n")
        nc = inc[inc["unit"] != "chapter"]
        ct = pd.crosstab([nc["unit"], nc["corpus"], nc["form"]], nc["is_complete"]).reset_index()
        P(md_table(ct) + "\n")
        P("Источник: Ficbook — статус произведения; Проза.ру — «продолжение следует» (false) / «конец», «закончено» (true) "
          "в последних 400 символах, иначе одиночная запись — true; склеенная серия — маркер конца в заголовках или в хвосте "
          "последней главы (true), «продолжение следует» в последней главе (false), иначе unknown; сказки — true.\n")
        P("### Строгие пороги (проверка устойчивости RQ1, RQ2)\n")
        rows = []
        for (u, f), g in inc[inc["form"].isin(["povest", "novel"]) & inc["unit"].isin(["text", "linked_work"])].groupby(["unit", "form"]):
            s = g[g["strict_long_form"]]
            sc_ = s[s["is_complete"] == "true"]
            rows.append(dict(единица=u, форма=f, основной_порог=len(g), строгий_порог=len(s), авторов_строгий=s["author_id"].nunique(),
                             строгий_и_завершённые=len(sc_)))
        P(md_table(pd.DataFrame(rows)) + "\n")
        P(f"Строгие пороги: повесть ≥ {strict['povest']}, роман ≥ {strict['novel']} слов.\n")
        P("### Афанасьев: типы сказок при порогах 600 и 400\n")
        af_ = a[(a["corpus"] == "afanasyev")]
        base = af_[(af_["lang_orig"] == "ru") & (af_["n_words"] > 0)]
        rows = []
        for t, g in base.groupby("af_type"):
            rows.append(dict(тип=t, всего_рус=len(g), при_600=int(g["include"].sum()), при_400=int(g["include_sens400"].sum())))
        tt = pd.DataFrame(rows)
        tot = dict(тип="итого", всего_рус=int(tt["всего_рус"].sum()), при_600=int(tt["при_600"].sum()), при_400=int(tt["при_400"].sum()))
        tt = pd.concat([tt, pd.DataFrame([tot])], ignore_index=True)
        for c_ in ["при_600", "при_400"]:
            tt[c_ + "_доля"] = (tt[c_] / tot[c_]).round(2)
        P(md_table(tt) + "\n")
        P("Типы — по номеру Афанасьева (жанровые разделы издания; разметка в `configs/s00_inventory.yaml`, `afanasyev_types`). "
          "animal — о животных, magic — волшебные (вкл. богатырские), novella — бытовые/новеллистические, "
          "anecdote — анекдоты, other — былички, легенды, небылицы, докучные сказки, прибаутки.\n")

    # ---------------- 8. Что вне манифеста
    P("## 8. Что не вошло в манифест\n")
    excl = []
    if frac >= 1:
        for rel in cfg["sources"]["proza"]["dbs"]:
            con = sqlite3.connect(f"file:{ROOT / rel}?immutable=1", uri=True)
            for st, err, n in con.execute("select status, coalesce(error,''), count(*) from stories "
                                          "where status is null or status!=200 group by 1,2"):
                excl.append(dict(база=Path(rel).name, статус=st, причина=err[:50], записей=n))
            con.close()
        P(md_table(pd.DataFrame(excl).groupby(["статус", "причина"], dropna=False)["записей"].sum().reset_index()) + "\n")
    else:
        P("(считается только при полном прогоне)\n")
    P("- Новостной корпус — ещё не получен (пользователь обогащает его, будет позже).")
    P("- Ficbook: главы со статусом error/empty — см. раздел 5.\n")

    out = rep / f"00_inventory{suffix}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("записано:", out)


if __name__ == "__main__":
    main()
