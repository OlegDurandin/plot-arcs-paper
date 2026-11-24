"""Этап 2. Лексиконные признаки: по предложениям -> по сегментам N = 10, 20; покрытие лексиконами.

Запуск:
    .venv/bin/python src/s02_lexicon.py configs/s02_emotion.yaml            # на выборочном прогоне этапа 1
    .venv/bin/python src/s02_lexicon.py configs/s02_emotion.yaml --full     # на полном прогоне этапа 1

Лексиконы приводятся к леммам pymorphy3 (как леммы корпуса, ё -> е). Словосочетания RuSentiLex в однословном
сопоставлении не участвуют (их число пишется в отчёт).
Признаки предложения: n_words; RuSentiLex — pos, neg, hits (вкл. нейтральные/неоднозначные); LinisCrowd — hits, сумма
средних оценок, hits_polar (|оценка| ≥ 0,5); NRC — счётчики 10 категорий и hits (≥ 1 категория).
Сегмент: суммы по предложениям [sent_start, sent_end).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

NRC_CATS = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]
LEX: dict = {}


def lemma_variants(morph, w: str) -> set[str]:
    w = w.strip().lower().replace("ё", "е")
    out = {w}
    if " " not in w and w:
        out.add(morph.parse(w)[0].normal_form.replace("ё", "е"))
    return out


def build_lexicons(cfg: dict) -> dict:
    import pymorphy3
    morph = pymorphy3.MorphAnalyzer()
    base = ROOT / cfg["external_dir"]
    info = {}
    # RuSentiLex: «слово, POS, лемма, тональность, источник[, ...]»
    rsl = {}
    n_phr = 0
    for ln in (base / "rusentilex/rusentilex_2017.txt").read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("!"):
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue
        lem, pol = parts[2], parts[3]
        if " " in lem:
            n_phr += 1
            continue
        val = {"positive": 1, "negative": -1}.get(pol, 0)
        for v in lemma_variants(morph, lem):
            # если у слова несколько значений с разной тональностью — берём 0 (контекстно-зависимое)
            rsl[v] = val if v not in rsl or rsl[v] == val else 0
    info["rusentilex"] = dict(entries_single=len(rsl), phrases_skipped=n_phr,
                              pos=sum(v > 0 for v in rsl.values()), neg=sum(v < 0 for v in rsl.values()))
    # LinisCrowd: «Words;mean;dispersion;average rate» (cp1251, десятичная запятая)
    lin = pd.read_csv(base / "linis_crowd/unzipped/dictionary_documents/words_all_full_rating.csv",
                      sep=";", encoding="cp1251", decimal=",", usecols=[0, 1])
    lin.columns = ["w", "mean"]
    lin = lin.dropna()
    linis = {}
    for w, m in zip(lin["w"].astype(str), lin["mean"].astype(float)):
        for v in lemma_variants(morph, w):
            linis.setdefault(v, m)
    info["linis_crowd"] = dict(entries=len(linis), polar=sum(abs(v) >= 0.5 for v in linis.values()))
    # NRC EmoLex, русская версия (машинный перевод)
    nrc_df = pd.read_csv(base / "nrc_emolex/unzipped/NRC-Emotion-Lexicon/OneFilePerLanguage/Russian-NRC-EmoLex.txt", sep="\t")
    nrc = {}
    n_multi = 0
    for r in nrc_df.itertuples(index=False):
        ru = str(getattr(r, "_11") if hasattr(r, "_11") else r[-1])
        vec = np.array([getattr(r, c) for c in NRC_CATS], dtype=np.int8)
        if vec.sum() == 0:
            continue
        if " " in ru.strip():
            n_multi += 1
            continue
        for v in lemma_variants(morph, ru):
            nrc[v] = np.maximum(nrc.get(v, np.zeros(10, np.int8)), vec)
    info["nrc_emolex_ru"] = dict(entries_with_category=len(nrc), multiword_skipped=n_multi)
    return dict(rsl=rsl, linis=linis, nrc=nrc, info=info)


def _init(lex):
    global LEX
    LEX = lex


def shard_features(path: str) -> pd.DataFrame:
    rsl, linis, nrc = LEX["rsl"], LEX["linis"], LEX["nrc"]
    s = pd.read_parquet(path, columns=["doc_id", "sent_idx", "n_words", "lemmas"])
    n = len(s)
    out = {k: np.zeros(n, np.int32) for k in ["rsl_pos", "rsl_neg", "rsl_hits", "lin_hits", "lin_polar", "nrc_hits"]}
    lin_sum = np.zeros(n, np.float32)
    nrc_cnt = np.zeros((n, 10), np.int32)
    for i, lem in enumerate(s["lemmas"].to_numpy()):
        for w in lem.split():
            v = rsl.get(w)
            if v is not None:
                out["rsl_hits"][i] += 1
                if v > 0:
                    out["rsl_pos"][i] += 1
                elif v < 0:
                    out["rsl_neg"][i] += 1
            m = linis.get(w)
            if m is not None:
                out["lin_hits"][i] += 1
                lin_sum[i] += m
                if abs(m) >= 0.5:
                    out["lin_polar"][i] += 1
            e = nrc.get(w)
            if e is not None:
                out["nrc_hits"][i] += 1
                nrc_cnt[i] += e
    df = s[["doc_id", "sent_idx", "n_words"]].copy()
    for k, v in out.items():
        df[k] = v
    df["lin_sum"] = lin_sum
    for j, c in enumerate(NRC_CATS):
        df[f"nrc_{c}"] = nrc_cnt[:, j]
    return df


FEATS = ["n_words", "rsl_pos", "rsl_neg", "rsl_hits", "lin_hits", "lin_polar", "lin_sum", "nrc_hits"] + [f"nrc_{c}" for c in NRC_CATS]


def aggregate(sf: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    """Суммы признаков по сегментам [sent_start, sent_end) через накопленные суммы внутри документа."""
    sf = sf.sort_values(["doc_id", "sent_idx"])
    rows = []
    segg = dict(tuple(seg.groupby("doc_id")))
    for doc, g in sf.groupby("doc_id", sort=False):
        if doc not in segg:
            continue
        m = g[FEATS].to_numpy(dtype=np.float64)
        cs = np.vstack([np.zeros((1, m.shape[1])), np.cumsum(m, axis=0)])
        sg = segg[doc]
        vals = cs[sg["sent_end"].to_numpy()] - cs[sg["sent_start"].to_numpy()]
        part = pd.DataFrame(vals, columns=FEATS)
        part.insert(0, "seg", sg["seg"].to_numpy())
        part.insert(0, "doc_id", doc)
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def work_shard(args):
    path, out_dir, ns, s01 = args
    name = Path(path).stem
    done = all((out_dir / f"N{n}" / f"{name}.parquet").exists() for n in ns)
    if done:
        return name
    sf = shard_features(path)
    docs = sf["doc_id"].unique()
    for n in ns:
        seg = pd.read_parquet(s01 / f"segments_rel_N{n}.parquet", filters=[("doc_id", "in", list(docs))])
        agg = aggregate(sf, seg)
        (out_dir / f"N{n}").mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_dir / f"N{n}" / f"{name}.parquet", index=False)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s01 = ROOT / cfg["s01_run"]
    if not args.full:
        cands = sorted(glob.glob(str(ROOT / "data/processed/s01_*_sample")), key=lambda p: Path(p).stat().st_mtime)
        s01 = Path(cands[-1])
    out = ROOT / cfg["out_dir"] / ("lex" if args.full else "lex_sample")
    out.mkdir(parents=True, exist_ok=True)
    lexf = ROOT / cfg["out_dir"] / "lexicons.json"
    lex = build_lexicons(cfg)
    lexf.write_text(json.dumps(lex["info"], ensure_ascii=False, indent=2))
    print("лексиконы:", json.dumps(lex["info"], ensure_ascii=False))
    shards = sorted(glob.glob(str(s01 / "sentences" / "*.parquet")))
    tasks = [(p, out, cfg["segments"], s01) for p in shards]
    with Pool(8, initializer=_init, initargs=(lex,)) as pool:
        for i, name in enumerate(pool.imap_unordered(work_shard, tasks)):
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                print(f"{i + 1}/{len(tasks)}", flush=True)
    for n in cfg["segments"]:
        parts = sorted((out / f"N{n}").glob("shard-*.parquet"))
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(out / f"lex_N{n}.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(dict(s01=str(s01.relative_to(ROOT)), segments=cfg["segments"],
                                                   lexicons=lex["info"]), ensure_ascii=False, indent=2))
    print("готово:", out)


if __name__ == "__main__":
    main()
