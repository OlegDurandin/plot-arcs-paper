"""Этап 1. Отчёт для Ворот 1: сводка очистки и сегментации + 20 случайных документов каждой группы.

Запуск:
    .venv/bin/python src/s01_review.py configs/s01_segment.yaml <run_dir>
Пишет reports/01_segmentation.md (сводка) и reports/01_segmentation_review.md (документы для ручной проверки).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from textio import TextLoader  # noqa: E402

PER_GROUP = 20


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else (f"{v:,}".replace(",", " ") if isinstance(v, (int, np.integer))
                                                          else f"{v:.3g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def snippet(s: str, n: int = 220, tail: bool = False) -> str:
    s = " ".join(s.split())
    return ("…" + s[-n:]) if tail else (s[:n] + ("…" if len(s) > n else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("run_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = ROOT / run
    meta = json.loads((run / "meta.json").read_text())
    d = pd.read_parquet(run / "docs.parquet")
    d["group"] = d["unit"] + " / " + d["corpus"] + " / " + d["form"]
    rep = ROOT / "reports"

    # ---------------- сводка
    L = [f"# Этап 1. Предобработка и сегментация — сводка\n",
         f"Каталог: `{run.relative_to(ROOT)}`, хеш `{meta['config_hash']}`, документов {meta['n_docs']}, "
         f"слов после очистки {meta['words_total']:,}".replace(",", " ") + f", время {meta['seconds']} с.\n",
         f"W0 = медиана длины основной базы (`base_magic600`, n = {meta['W0_n']}) после очистки = **{meta['W0']}** слов.\n",
         "Слово = токен `razdel`, содержащий букву (числа и пунктуация не считаются; на этапе 0 числа считались словами, "
         "поэтому `n_words_clean` до ~1–2 % меньше у обычных текстов и сильнее — у текстов с цифрами и ссылками).\n",
         "## Очистка по группам\n"]
    d["ratio"] = d["n_words_clean"] / d["n_words"]
    g = d.groupby("group").agg(
        док=("doc_id", "size"), слов_млн=("n_words_clean", lambda x: round(x.sum() / 1e6, 2)),
        доля_слов_после_очистки_медиана=("ratio", "median"), p1=("ratio", lambda x: x.quantile(.01)),
        заголовков_глав=("header_dropped", "sum"), строк_примечаний=("notes_end_cut_lines", "sum"),
        строк_примеч_в_начале=("notes_start_cut_lines", "sum"), служебных_строк=("service_lines_dropped", "sum"),
        строк_со_ссылками=("urls_removed", "sum"), знаков_ударения=("accents_removed", "sum"),
        длинных_предл_разрезано=("forced_splits", "sum"), не_русский=("non_russian", "sum")).reset_index()
    L.append(md_table(g.round(3)) + "\n")
    L.append("`не_русский` — доля букв і/ї/є/ґ/ў ≥ 0,05 на слово (украинский/белорусский) или ≥ 30 % слов латиницей; "
             "такие документы надо исключить на следующих этапах (флаг `non_russian` в `docs.parquet`).\n")
    pre = d.groupby("corpus").agg(дореформ_букв=("prereform_chars", "sum"), ъ_на_конце=("hard_sign_word_end", "sum"),
                                  док_с_дореформ=("prereform_chars", lambda x: int((x > 0).sum()))).reset_index()
    L.append("## Орфография\n")
    L.append(md_table(pre) + "\n")
    L.append("Буквы ѣ, і, ѳ, ѵ: у Афанасьева 0. В Проза.ру буква «і» почти целиком приходится на украинские тексты "
             "(помечены `non_russian`).\n")
    L.append("## Сегменты\n")
    rows = []
    for k, v in meta["segments"].items():
        rows.append(dict(схема=k, документов=v["docs"], сегментов=v.get("segments", v.get("windows")),
                         без_сегментов=v.get("docs_too_few_sentences", ""), шаг=v.get("step", "")))
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("Относительные сегменты: границы — концы предложений, ближайшие к k·T/N; в каждом сегменте ≥ 1 предложение; "
             "если предложений меньше N, документ для этого N не сегментируется. Окна: длина W, шаг W/2, концы выровнены "
             "по предложениям; документы короче W окон не получают.\n")
    rows = []
    for n in [10, 20, 50]:
        s = pd.read_parquet(run / f"segments_rel_N{n}.parquet").merge(d[["doc_id", "group"]], on="doc_id")
        for grp, gg in s.groupby("group"):
            rows.append(dict(N=n, группа=grp, медиана_слов=int(gg["n_words"].median()),
                             p5=int(gg["n_words"].quantile(.05)), доля_сегм_1_предл=round(float(((gg["sent_end"] - gg["sent_start"]) == 1).mean()), 3)))
    L.append("### Слов в сегменте\n")
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    (rep / "01_segmentation.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------- ручная проверка
    a = pd.read_parquet(ROOT / cfg["analysis_set"]).set_index(["doc_id", "unit"])  # doc_id не уникален между unit
    sent_files = sorted((run / "sentences").glob("*.parquet"))
    samp = d.sample(frac=1.0, random_state=cfg["seed"]).groupby("group").head(PER_GROUP).sort_values(["group", "doc_id"])
    want = set(samp["doc_id"])
    sents = []
    for f in sent_files:
        x = pd.read_parquet(f, columns=["doc_id", "sent_idx", "part_idx", "text", "n_words"], filters=[("doc_id", "in", list(want))])
        if len(x):
            sents.append(x)
    sents = pd.concat(sents).sort_values(["doc_id", "sent_idx"])
    seg20 = pd.read_parquet(run / "segments_rel_N20.parquet", filters=[("doc_id", "in", list(want))])
    loader = TextLoader()
    R = [f"# Этап 1. Ручная проверка очистки и сегментации ({PER_GROUP} случайных документов на группу, seed {cfg['seed']})\n",
         "Для каждого документа: начало и конец исходного текста и после очистки; что удалила очистка; "
         "длины 20 сегментов (N = 20) и три случайных стыка сегментов «…конец | начало…».\n",
         "Что проверить: (1) очистка не съела авторский текст и убрала служебное; (2) стыки сегментов проходят по концам "
         "предложений; (3) длины сегментов ровные.\n"]
    rng = np.random.default_rng(cfg["seed"])
    for grp, gs in samp.groupby("group"):
        R.append(f"\n## {grp}\n")
        for r in gs.itertuples():
            parts = loader.load(r.doc_id, a.loc[(r.doc_id, r.unit), "text_path"])
            raw = "\n".join(t for _, t in parts)
            ss = sents[sents["doc_id"] == r.doc_id]
            clean = " ".join(ss["text"])
            acts = {k: int(getattr(r, k)) for k in ["header_dropped", "notes_start_cut_lines", "notes_end_cut_lines",
                                                    "service_lines_dropped", "urls_removed", "accents_removed", "forced_splits"]
                    if getattr(r, k)}
            R.append(f"### `{r.doc_id}` — {int(r.n_words)} → {int(r.n_words_clean)} слов, {r.n_sent} предл., частей {r.n_parts}"
                     + (" — **НЕ РУССКИЙ**" if r.non_russian else ""))
            R.append(f"- очистка: {acts if acts else 'ничего не удалено'}")
            R.append(f"- начало, исходный: {snippet(raw)}")
            R.append(f"- начало, очищенный: {snippet(clean)}")
            R.append(f"- конец, исходный: {snippet(raw, tail=True)}")
            R.append(f"- конец, очищенный: {snippet(clean, tail=True)}")
            sg = seg20[seg20["doc_id"] == r.doc_id].sort_values("seg")
            if len(sg):
                R.append(f"- N = 20, слов в сегментах: {' '.join(map(str, sg['n_words']))}")
                tx = ss.set_index("sent_idx")["text"]
                for k in sorted(rng.choice(np.arange(1, len(sg)), size=min(3, len(sg) - 1), replace=False)):
                    b = int(sg.iloc[k]["sent_start"])
                    left = " ".join(str(tx.get(b - 1, "")).split()[-8:])
                    right = " ".join(str(tx.get(b, "")).split()[:8])
                    R.append(f"  - стык {k}: …{left} **|** {right}…")
            else:
                R.append("- N = 20: не сегментирован (предложений меньше 20)")
            R.append("")
    (rep / "01_segmentation_review.md").write_text("\n".join(R), encoding="utf-8")
    print("записано: reports/01_segmentation.md, reports/01_segmentation_review.md")


if __name__ == "__main__":
    main()
