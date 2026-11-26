"""[УСТАРЕЛО — batch0, НЕ ОТДАВАТЬ; заменено s02_students_batch.py по решению Ворот 5, п. 3, 2026-11-26]
Партия 1 для студентов (решение Ворот 2/5, 2026-11-25): человек проверяет РАЗНИЦУ «рассказ − сказка» на концовках
(30 + 30, последние 2 сегмента N20) и началах (15 + 15, первые 2 сегмента), вперемешку, вслепую, со скрытыми повторами.

Тексты — из выборки LLM-дуг (`data/processed/s02/llm_arcs/arcs_key.parquet`): сказки базы Афанасьева и рассказы в
диапазоне длины сказок. Каждый текст даёт не больше одного фрагмента; тексты для концовок и начал не пересекаются.
LLM-оценки этих фрагментов (среднее тональностей двух сегментов) сохраняются в ключе — для сравнения после разметки.

Критерий (до разметки): вывод о концовке засчитывается, если разница «рассказ − сказка» по человеческой разметке
того же знака, что у LLM на тех же фрагментах, и её 95 % ДИ не включает 0. Для начал — то же для разницы
«(начало − середина своего текста)»: к каждому тексту с началом добавлен фрагмент из середины.

Запуск:  .venv/bin/python src/s02_students_batch0.py configs/s02_students_batch0.yaml
Выход:   data/processed/s02/students/batch0_key.parquet, reports/02_students_batch0.html
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
import s02_validate as V  # noqa: E402
from s02_gold import TEMPLATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    ecfg = load_config(ROOT / cfg["emotion_config"])
    acfg = load_config(ROOT / cfg["arcs_config"])
    bc = cfg["batch"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    kf = out / "batch0_key.parquet"
    if kf.exists():
        key = pd.read_parquet(kf)
        print("ключ уже есть — страница пересобирается по нему:", kf)
    else:
        ak = pd.read_parquet(ROOT / acfg["out_dir"] / "arcs_key.parquet")
        ann = V.read_ann(ROOT / acfg["out_dir"] / f"arcs__{acfg['models']['main']}.csv")
        ak = ak.merge(ann[["uid", "tone", "skip"]], on="uid")
        ok = ak.groupby("doc_id")["skip"].transform("sum") == 0
        ak = ak[ok & ak["stratum"].isin(["afanasyev", "story"])]
        rows = []
        used = set()
        for zone, segs, spec in [("конец", [18, 19], bc["endings"]), ("начало", [0, 1], bc["beginnings"])]:
            for st, n in spec.items():
                docs = [d for d in ak.loc[ak["stratum"] == st, "doc_id"].unique() if d not in used]
                pick = rng.choice(docs, n, replace=False)
                used.update(pick)
                for d in pick:
                    zs = [(zone, segs)]
                    if zone == "начало" and bc.get("beginnings_with_middle"):
                        zs.append(("середина", bc["middle_segs"]))
                    for zn, sg in zs:
                        t = ak[(ak["doc_id"] == d) & ak["seg"].isin(sg)].sort_values("seg")
                        rows.append(dict(doc_id=d, stratum=st, zone=zn, author_id=t["author_id"].iloc[0],
                                         segs=",".join(map(str, t["seg"])), n_words=int(t["shown_n_words"].sum()),
                                         llm_tone=float(t["tone"].mean()), text=" ".join(t["text"])))
        key = pd.DataFrame(rows).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
        key.insert(0, "uid", [f"b{i + 1:03d}" for i in range(len(key))])
        # скрытые повторы
        base = list(key["uid"])
        src = np.sort(rng.choice(min(bc["repeat_source_max"], len(base)), bc["n_repeats"], replace=False))
        reps = []
        for j, p in enumerate(src):
            r = key.iloc[p].copy()
            r["repeat_of"] = r["uid"]
            r["uid"] = f"b{len(base) + j + 1:03d}"
            reps.append(r)
        key["repeat_of"] = None
        key = pd.concat([key, pd.DataFrame(reps)], ignore_index=True)
        key.to_parquet(kf, index=False)
    base = list(key.loc[key["repeat_of"].isna(), "uid"])
    reps = key[key["repeat_of"].notna()]
    pos = {u: i for i, u in enumerate(base)}
    order = list(base)
    rr = np.random.default_rng(cfg["seed"] + 1)
    for _, r in reps.sort_values("uid", ascending=False).iterrows():
        lo = pos[r["repeat_of"]] + bc["repeat_min_gap"]
        order.insert(int(rr.integers(lo, len(base) + 1)), r["uid"])
    ix = {u: i for i, u in enumerate(order)}
    assert all(ix[r.uid] - ix[r.repeat_of] >= bc["repeat_min_gap"] for r in reps.itertuples()), "разрыв < gap"
    items = dict(zip(key["uid"], key["text"]))
    page = (TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__FRESH__", json.dumps(order)).replace("__APPEND__", json.dumps(order))
            .replace("__EMOTIONS__", json.dumps(ecfg["gold"]["emotions"], ensure_ascii=False))
            .replace('const KEY = "gold_annotation_v1";', f'const KEY = "{cfg["storage_key"]}";')
            .replace('"gold_annotation_" + new Date()', '"students_batch1_" + new Date()')
            .replace("<title>Золотая разметка — тональность и эмоции</title>", "<title>batch0 — устарело, не отдавать</title>"))
    assert cfg["storage_key"] in page and "students_batch1_" in page
    (ROOT / cfg["page"]).write_text(page, encoding="utf-8")
    o = key[key["repeat_of"].isna()]
    print(o.groupby(["zone", "stratum"]).agg(фрагментов=("uid", "size"), слов_медиана=("n_words", "median"),
                                             LLM_среднее=("llm_tone", "mean")).round(2).to_string())
    print(f"страница: {len(o)} фрагментов + {len(reps)} повторов, слов к прочтению {int(key['n_words'].sum())} → {cfg['page']}")
    for z in ["конец", "начало", "середина"]:
        x = o[o["zone"] == z]
        if not len(x):
            continue
        d = x.loc[x["stratum"] == "story", "llm_tone"].mean() - x.loc[x["stratum"] == "afanasyev", "llm_tone"].mean()
        print(f"LLM на этих фрагментах, {z}: рассказ − сказка = {d:+.2f} балла")
    b = o[o["zone"].isin(["начало", "середина"])].pivot_table(index=["doc_id", "stratum"], columns="zone", values="llm_tone").reset_index()
    if "середина" in b:
        b["нач_мин_сер"] = b["начало"] - b["середина"]
        g = b.groupby("stratum")["нач_мин_сер"].mean()
        print(f"LLM, начало − середина своего текста: сказка {g['afanasyev']:+.2f}, рассказ {g['story']:+.2f}, "
              f"разница рассказ − сказка {g['story'] - g['afanasyev']:+.2f}")


if __name__ == "__main__":
    main()
