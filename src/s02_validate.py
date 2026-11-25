"""Этап 2. Валидация сигналов на золотой выборке.

Запуск:
    .venv/bin/python src/s02_validate.py configs/s02_emotion.yaml --gold <csv пользователя> [--llm data/processed/s02/gold_llm.csv]
    .venv/bin/python src/s02_validate.py configs/s02_emotion.yaml --llm-only     # ПРЕДВАРИТЕЛЬНО: против LLM-разметки

Сигналы считаются ровно на показанном фрагменте (для повестей/романов — отрывок ≈ 350 слов из сегмента).
Тональность: ρ Спирмена сигнала с оценкой −2…+2, по группам; 95 % ДИ — бутстрэп по авторам (1000 повторов).
Эмоция: доля совпадений и macro-F1 доминирующей эмоции (CEDR не различает «отвращение»).
Потолок: согласие разметчика с собой по скрытым повторам (ρ тональности, κ Коэна эмоции).
Пишет reports/02_validation[_llm_preliminary].md.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

EMO_CEDR = {"no_emotion": "нейтрально", "joy": "радость", "sadness": "грусть", "surprise": "удивление",
            "fear": "страх", "anger": "гнев"}
EMO_NRC = {"joy": "радость", "sadness": "грусть", "surprise": "удивление", "fear": "страх", "anger": "гнев",
           "disgust": "отвращение"}


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (not isinstance(v, str) and pd.isna(v)) else
                                     (str(v) if isinstance(v, (int, np.integer, str)) else f"{v:.2f}") for v in r) + " |")
    return "\n".join(out)


def gold_signals(cfg, key: pd.DataFrame) -> pd.DataFrame:
    """Лексиконы и классификаторы на показанных фрагментах."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s02_classify import run_models
    from s02_lexicon import build_lexicons, shard_features, _init
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    s01 = ROOT / cfg["s01_run"]
    lex = build_lexicons(cfg)
    _init(lex)
    ids = set(key["doc_id"])
    feats = []
    for f in sorted(glob.glob(str(s01 / "sentences" / "*.parquet"))):
        s = pd.read_parquet(f, columns=["doc_id"], filters=[("doc_id", "in", list(ids))])
        if len(s):
            sf = shard_features(f)
            feats.append(sf[sf["doc_id"].isin(ids)])
    sf = pd.concat(feats).set_index(["doc_id", "sent_idx"]).sort_index()
    rows = []
    for r in key.itertuples():
        x = sf.loc[r.doc_id].loc[r.shown_sent_start:r.shown_sent_end - 1].sum()
        rows.append(x)
    L = pd.DataFrame(rows).reset_index(drop=True)
    out = key[["uid", "doc_id", "stratum", "author_id", "shown_n_words", "is_excerpt", "text"]].reset_index(drop=True).copy()
    nw = L["n_words"].clip(lower=1)
    out["RSL"] = (L["rsl_pos"] - L["rsl_neg"]) / nw
    out["RSL_доля_полярных"] = (L["rsl_pos"] - L["rsl_neg"]) / (L["rsl_pos"] + L["rsl_neg"]).replace(0, np.nan)
    out["LinisCrowd"] = L["lin_sum"] / nw
    out["LinisCrowd_среднее_по_словам"] = L["lin_sum"] / L["lin_hits"].replace(0, np.nan)
    out["NRC_полярность"] = (L["nrc_positive"] - L["nrc_negative"]) / nw
    for c in EMO_NRC:
        out[f"nrc_{c}"] = L[f"nrc_{c}"] / nw
    mc = cfg["models"]
    dev = mc["device"] if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mc["sentiment"]["hf"], revision=mc["sentiment"].get("revision"))
    models = {"sent": AutoModelForSequenceClassification.from_pretrained(mc["sentiment"]["hf"]).to(dev).eval(),
              "emo": AutoModelForSequenceClassification.from_pretrained(mc["emotion"]["hf"]).to(dev).eval()}
    # как в основном расчёте: куски ≤ 510 токенов по предложениям, среднее по словам
    sent = {}
    for f in sorted(glob.glob(str(s01 / "sentences" / "*.parquet"))):
        s = pd.read_parquet(f, columns=["doc_id", "sent_idx", "text", "n_words"], filters=[("doc_id", "in", list(ids))])
        for d, g in s.groupby("doc_id"):
            sent[d] = g.set_index("sent_idx").sort_index()
    chunks, owner, weight = [], [], []
    for i, r in enumerate(key.itertuples()):
        g = sent[r.doc_id].loc[r.shown_sent_start:r.shown_sent_end - 1]
        tl = [len(t) for t in tok(list(g["text"]), add_special_tokens=False)["input_ids"]]
        buf, bt, bw = [], 0, 0
        for txt, t_, w_ in zip(g["text"], tl, g["n_words"]):
            if buf and bt + t_ > mc["max_tokens"] - 2:
                chunks.append(" ".join(buf)); owner.append(i); weight.append(bw)
                buf, bt, bw = [], 0, 0
            buf.append(txt); bt += t_; bw += w_
        if buf:
            chunks.append(" ".join(buf)); owner.append(i); weight.append(bw)
    res = run_models(chunks, tok, models, dev, mc["max_tokens"])
    owner, weight = np.array(owner), np.array(weight, float)
    labs = {k: [m.config.id2label[j] for j in range(m.config.num_labels)] for k, m in models.items()}
    for k in res:
        for j, lb in enumerate(labs[k]):
            v = np.bincount(owner, weights=res[k][:, j] * weight, minlength=len(key)) / np.bincount(owner, weights=weight, minlength=len(key))
            out[f"{k}_{lb}"] = v
    out["Классификатор_тональности"] = out["sent_positive"] - out["sent_negative"]
    return out


def spearman_ci(x, y, groups, rng, B=1000):
    m = ~(pd.isna(x) | pd.isna(y))
    x, y, groups = np.asarray(x)[m], np.asarray(y)[m], np.asarray(groups)[m]
    if len(x) < 8:
        return np.nan, np.nan, np.nan, len(x)
    rho = stats.spearmanr(x, y).statistic
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    bs = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if np.std(x[take]) > 0 and np.std(y[take]) > 0:
            bs.append(stats.spearmanr(x[take], y[take]).statistic)
    return rho, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(x)


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def macro_f1(true, pred):
    cats = sorted(set(true))
    f = []
    for c in cats:
        tp = np.sum((pred == c) & (true == c)); fp = np.sum((pred == c) & (true != c)); fn = np.sum((pred != c) & (true == c))
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


SIGNALS = ["Классификатор_тональности", "RSL", "RSL_доля_полярных", "LinisCrowd", "LinisCrowd_среднее_по_словам", "NRC_полярность"]
STRATA = ["afanasyev", "fanfic", "story", "novella", "povest", "novel"]
ST_RU = {"afanasyev": "волшебная сказка (Афанасьев)", "fanfic": "фанфик", "story": "рассказ", "novella": "новелла",
         "povest": "повесть (отрывок)", "novel": "роман (отрывок)"}


def validate(cfg, sig: pd.DataFrame, ann: pd.DataFrame, label: str, rng) -> list[str]:
    d = sig.merge(ann, on="uid")
    d = d[d["skip"] == 0]
    L = [f"## Тональность: ρ Спирмена сигнала с разметкой ({label})\n",
         "В ячейке: ρ [95 % ДИ, бутстрэп по авторам], n. Для повестей и романов — на отрывке ≈ 350 слов; дуги строятся по целым "
         "сегментам, где сигнал усреднён по большему тексту и стабильнее, поэтому ρ на отрывках — **нижняя оценка** качества "
         "для длинных форм.\n"]
    rows = []
    for s_ in SIGNALS:
        r = {"сигнал": s_}
        for st in ["все"] + STRATA + ["художественные (без сказок)"]:
            x = d if st == "все" else d[~d["stratum"].isin(["afanasyev", "fairytales"])] if st.startswith("художеств") else d[d["stratum"] == st]
            rho, lo, hi, n = spearman_ci(x[s_], x["tone"], x["author_id"], rng)
            r[ST_RU.get(st, st)] = f"{rho:.2f} [{lo:.2f}; {hi:.2f}], {n}" if not np.isnan(rho) else "—"
        rows.append(r)
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    # короткие сегменты сказок
    tale = d[d["stratum"] == "afanasyev"].copy()
    tale["размер"] = pd.cut(tale["shown_n_words"], [0, 40, 70, 10_000], labels=["≤40 слов", "41–70", ">70"])
    rows = []
    for s_ in SIGNALS:
        r = {"сигнал": s_}
        for b, x in tale.groupby("размер", observed=True):
            rho, lo, hi, n = spearman_ci(x[s_], x["tone"], x["author_id"], rng)
            r[str(b)] = f"{rho:.2f} [{lo:.2f}; {hi:.2f}], {n}" if not np.isnan(rho) else "—"
        rows.append(r)
    L.append("### Сказки Афанасьева по длине сегмента\n")
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    # эмоции
    cedr = d[[f"emo_{k}" for k in EMO_CEDR]].to_numpy().argmax(1)
    d["pred_cedr"] = [EMO_CEDR[list(EMO_CEDR)[i]] for i in cedr]
    nrcm = d[[f"nrc_{k}" for k in EMO_NRC]].to_numpy()
    d["pred_nrc"] = np.where(nrcm.max(1) > 0, [EMO_NRC[list(EMO_NRC)[i]] for i in nrcm.argmax(1)], "нейтрально")
    rows = []
    for name, col in [("CEDR (классификатор)", "pred_cedr"), ("NRC (лексикон)", "pred_nrc")]:
        r = {"сигнал": name}
        for st in ["все"] + STRATA:
            x = d if st == "все" else d[d["stratum"] == st]
            x = x[x["emotion"].astype(str) != ""]
            if len(x):
                r[ST_RU.get(st, st)] = f"{np.mean(x[col] == x['emotion']):.2f} / F1 {macro_f1(x['emotion'].to_numpy(), x[col].to_numpy()):.2f}"
        rows.append(r)
    L.append(f"## Доминирующая эмоция ({label}): доля совпадений / macro-F1\n")
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("Распределение эмоций в разметке: " + json.dumps(d["emotion"].value_counts().to_dict(), ensure_ascii=False) + "\n")
    return L


def read_ann(path) -> pd.DataFrame:
    a = pd.read_csv(path)
    a["skip"] = a["skip"].fillna(0).astype(int)
    a["tone"] = pd.to_numeric(a["tone"], errors="coerce")
    a["emotion"] = a["emotion"].fillna("").astype(str)
    a.loc[a["tone"].isna() & (a["skip"] == 0), "skip"] = 1          # без тональности — не используется
    return a[["uid", "tone", "emotion", "skip"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gold")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--llm-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / cfg["out_dir"]
    key = pd.read_parquet(out / "gold_key.parquet")
    orig = key[key["repeat_of"].isna()]
    sf = out / "gold_signals.parquet"
    if sf.exists():
        sig = pd.read_parquet(sf)
    else:
        sig = gold_signals(cfg, orig)
        sig.to_parquet(sf, index=False)
    R = []
    if args.llm_only:
        llm = read_ann(args.llm or out / "gold_llm.csv")
        R = ["# Этап 2. ПРЕДВАРИТЕЛЬНАЯ валидация сигналов против разметки локальной LLM\n",
             "**Предварительно.** Разметка — локальная LLM (MLX, температура 0), не человек. Используется только для отладки "
             "и оценки согласия; выбор главного сигнала — **только по разметке пользователя**.\n"]
        R += validate(cfg, sig, llm, "LLM, предварительно", rng)
        (ROOT / "reports" / "02_validation_llm_preliminary.md").write_text("\n".join(R), encoding="utf-8")
        print("\n".join(R))
        return
    gold = read_ann(args.gold)
    # согласие разметчика с собой по повторам
    rep = key[key["repeat_of"].notna()][["uid", "repeat_of"]]
    g1 = gold.set_index("uid")
    pairs = rep.join(g1, on="uid").join(g1, on="repeat_of", rsuffix="_first")
    pairs = pairs[(pairs["skip"] == 0) & (pairs["skip_first"] == 0)]
    self_rho = stats.spearmanr(pairs["tone"], pairs["tone_first"]).statistic if len(pairs) > 5 else np.nan
    self_k = cohen_kappa(pairs["emotion"], pairs["emotion_first"]) if len(pairs) > 5 else np.nan
    gold_o = gold[gold["uid"].isin(orig["uid"])]
    R = ["# Этап 2. Валидация сигналов на золотой разметке\n",
         f"Размечено: {int((gold_o['skip'] == 0).sum())} из {len(orig)} (пропущено {int((gold_o['skip'] == 1).sum())}).\n",
         f"**Потолок — согласие разметчика с собой** (скрытые повторы, n = {len(pairs)}): ρ тональности {self_rho:.2f}, "
         f"κ эмоции {self_k:.2f}.\n"]
    R += validate(cfg, sig, gold_o, "пользователь", rng)
    if args.llm and Path(args.llm).exists():
        llm = read_ann(args.llm)
        m = gold_o.merge(llm, on="uid", suffixes=("", "_llm"))
        m = m[(m["skip"] == 0) & (m["skip_llm"] == 0)]
        R.append(f"## Согласие LLM с пользователем\nρ тональности {stats.spearmanr(m['tone'], m['tone_llm']).statistic:.2f}, "
                 f"κ эмоции {cohen_kappa(m['emotion'], m['emotion_llm']):.2f}, n = {len(m)}.\n")
    (ROOT / "reports" / "02_validation.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
