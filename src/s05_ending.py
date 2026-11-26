"""Ворота 5, п. 2: подъём концовки волшебной сказки — развязка или финальная формула.

Шаги (configs/s05_ending.yaml):
  contrast  (2.1) контраст концовки без последнего сегмента: z(18–19) − z(3–17), а также z(20) − z(3–17), z(19–20) − z(3–17),
            для всех форм и сигналов (clf, rsl — полный корпус N20; LLM-дуги); ДИ — пуассоновский бутстрэп по авторам;
            разница «форма − рассказ»: полный корпус — пары по log(n_words), LLM — независимые группы.
  sample    (2.2) ключ для разметки по предложениям: последние 2 сегмента всех сказок базы и 100 рассказов той же длины
            (пары по log(n_words) из выборки LLM-дуг); каждое предложение — с 2 предыдущими как контекстом.
            Плюс (а) части смешанных предложений «развязка + формула» до формулы, (б) последний сегмент сказок без
            формульного блока — как сегмент, исходной инструкцией. Печатает оценку стоимости, запросов не шлёт.
  analyze   (2.3–2.4) формулы, тональность формулы и развязки, сравнение с последними предложениями рассказов,
            дуга сказки с последним сегментом без формулы.

Разметка между sample и analyze:  .venv/bin/python src/s02_llm_api.py configs/s05_ending.yaml --key sent
Запуск:   .venv/bin/python src/s05_ending.py configs/s05_ending.yaml --step {contrast,sample,analyze}
Выход:    reports/05_ending.md, reports/figures/05_ending_*.png, data/processed/s05/ending/*.parquet
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s03_report import FORM_RU  # noqa: E402
from s04_nulls import ORDER, md, zscore  # noqa: E402
from s05_mean_arcs import LLM_RU, author_sums, match_pairs  # noqa: E402
import s02_validate as V  # noqa: E402

REPORT = ROOT / "reports" / "05_ending.md"


def idx(pos):
    """Позиции 1-based: [a, b] — диапазон включительно, [a] — одна позиция; → индексы 0-based."""
    a, b = (pos[0], pos[-1])
    return np.arange(a - 1, b)


def ci_auth(c, authors, rng, B):
    """Среднее по текстам с пуассоновским бутстрэпом по авторам."""
    S, n = author_sums(np.asarray(c, float)[:, None], authors)
    m = S.sum() / n.sum()
    W = rng.poisson(1.0, (B, len(n))).astype(float)
    bs = (W @ S[:, 0]) / (W @ n)
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def diff_ci(ca, aa, cb, ab, rng, B):
    """Разница средних двух независимых групп; бутстрэп по авторам в каждой группе."""
    Sa, na = author_sums(np.asarray(ca, float)[:, None], aa)
    Sb, nb = author_sums(np.asarray(cb, float)[:, None], ab)
    Wa = rng.poisson(1.0, (B, len(na))).astype(float)
    Wb = rng.poisson(1.0, (B, len(nb))).astype(float)
    bs = (Wa @ Sa[:, 0]) / (Wa @ na) - (Wb @ Sb[:, 0]) / (Wb @ nb)
    return Sa.sum() / na.sum() - Sb.sum() / nb.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}; {hi:+.3f}]"


def llm_arcs(acfg):
    ao = ROOT / acfg["out_dir"]
    k = pd.read_parquet(ao / "arcs_key.parquet", columns=["uid", "doc_id", "stratum", "author_id", "seg", "doc_n_words"])
    an = V.read_ann(ao / f"arcs__{acfg['models']['main']}.csv")
    x = k.merge(an, on="uid")
    x = x[x.groupby("doc_id")["skip"].transform("sum") == 0].sort_values(["doc_id", "seg"])
    docs = x.groupby("doc_id").agg(stratum=("stratum", "first"), author_id=("author_id", "first"),
                                   n_words=("doc_n_words", "first")).reset_index()
    T = x["tone"].to_numpy(float).reshape(-1, 20)
    return docs, T


# ---------------------------------------------------------------- 2.1
def step_contrast(cfg):
    cc = cfg["contrast"]
    ncfg = load_config(ROOT / cfg["nulls_config"])
    acfg = load_config(ROOT / cfg["arcs_config"])
    rng = np.random.default_rng(cfg["seed"])
    B = cc["bootstrap"]
    mid = idx(cc["middle"])
    zones = {"18–19 (без последнего)": idx(cc["end_wo_last"]), "20 (последний)": idx(cc["end_last"]),
             "19–20": idx(cc["end_both"])}
    L = [f"## 2.1. Контраст концовки без последнего сегмента (N = 20; середина — позиции {cc['middle'][0]}–{cc['middle'][1]})\n",
         "Контраст текста = среднее z(зона) − среднее z(середина); z — внутри текста, без сглаживания. ДИ 95 % — пуассоновский "
         f"бутстрэп по авторам (B = {B}). Колонка «− рассказ»: разница контрастов «форма − рассказ»; полный корпус — пары, "
         "сопоставленные по log(n_words) (|Δ| ≤ 0,05), LLM — рассказы той же длины (независимые группы).\n"]
    store = []
    # полный корпус: clf, rsl
    arcs = pd.read_parquet(ROOT / ncfg["out_dir"] / "full" / "arcs_N20.parquet")
    arcs = arcs[arcs["within_author_cap"]].reset_index(drop=True)
    for s_ in ["clf", "rsl"]:
        Z = zscore(np.vstack(arcs[s_].to_numpy()).astype(np.float64))
        ok = ~np.isnan(Z).any(1)
        d, Z = arcs[ok].reset_index(drop=True), Z[ok]
        lw = np.log(d["n_words"].to_numpy())
        C = {zn: Z[:, ii].mean(1) - Z[:, mid].mean(1) for zn, ii in zones.items()}
        rows = []
        st = np.where((d["fk"] == "story").to_numpy())[0]
        for f in [f for f in ORDER if (d["fk"] == f).any()]:
            ia = np.where((d["fk"] == f).to_numpy())[0]
            row = {"форма": FORM_RU[f], "текстов": len(ia)}
            if f != "story":
                pa, pb = match_pairs(lw[ia], lw[st], 0.05, rng)
            for zn, c in C.items():
                m, lo, hi = ci_auth(c[ia], d["author_id"].astype(str).to_numpy()[ia], rng, B)
                row[zn] = fmt(m, lo, hi)
                store.append(dict(signal=s_, group=f, zone=zn, kind="contrast", mean=m, lo=lo, hi=hi, n=len(ia)))
            for zn, c in C.items():
                if f != "story" and not zn.startswith("19"):
                    dd = c[ia[pa]] - c[st[pb]]
                    bi = rng.integers(0, len(dd), (B, len(dd)))
                    bs = dd[bi].mean(1)
                    row[f"{zn.split(' ')[0]}: − рассказ"] = fmt(dd.mean(), *np.percentile(bs, [2.5, 97.5]))
                    row["пар с рассказом"] = len(dd)
                    store.append(dict(signal=s_, group=f, zone=zn, kind="minus_story", mean=dd.mean(),
                                      lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n=len(dd)))
            rows.append(row)
        L.append(f"### Сигнал {s_}, полный корпус ({len(d)} дуг)\n")
        L.append(md(pd.DataFrame(rows).fillna("")) + "\n")
    # LLM
    docs, T = llm_arcs(acfg)
    Z = zscore(T)
    ok = ~np.isnan(Z).any(1)
    docs, Z = docs[ok].reset_index(drop=True), Z[ok]
    C = {zn: Z[:, ii].mean(1) - Z[:, mid].mean(1) for zn, ii in zones.items()}
    rows = []
    sm = (docs["stratum"] == "story").to_numpy()
    au = docs["author_id"].astype(str).to_numpy()
    for g in ["afanasyev", "fairytales", "story", "novella", "fanfic", "povest", "novel"]:
        m_ = (docs["stratum"] == g).to_numpy()
        row = {"группа": LLM_RU[g], "текстов": int(m_.sum())}
        for zn, c in C.items():
            m, lo, hi = ci_auth(c[m_], au[m_], rng, B)
            row[zn] = fmt(m, lo, hi)
            store.append(dict(signal="llm", group=g, zone=zn, kind="contrast", mean=m, lo=lo, hi=hi, n=int(m_.sum())))
        for zn, c in C.items():
            if g != "story" and not zn.startswith("19"):
                m2, lo2, hi2 = diff_ci(c[m_], au[m_], c[sm], au[sm], rng, B)
                row[f"{zn.split(' ')[0]}: − рассказ"] = fmt(m2, lo2, hi2)
                store.append(dict(signal="llm", group=g, zone=zn, kind="minus_story", mean=m2, lo=lo2, hi=hi2, n=int(m_.sum())))
        rows.append(row)
    L.append(f"### LLM-дуги ({len(docs)} текстов; рассказы, новеллы, фанфики — той же длины, что сказки базы)\n")
    L.append(md(pd.DataFrame(rows).fillna("")) + "\n")
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(store).to_parquet(out / "contrast.parquet", index=False)
    (out / "contrast.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# ---------------------------------------------------------------- 2.2 / 2.3 helpers
def compile_formulas(fc):
    strict = re.compile("|".join(f"(?:{p})" for p in fc["strict"]), re.I)
    ext = re.compile("|".join(f"(?:{p})" for p in fc["strict"] + fc["extended"]), re.I)
    edit = re.compile("|".join(f"(?:{p})" for p in fc["editorial"]))
    return strict, ext, edit


def formula_block(texts, R, edit, tail):
    """Индекс начала формульного блока: первое предложение с формулой среди последних tail содержательных предложений."""
    cand = [i for i in range(len(texts)) if not edit.search(texts[i])][-tail:]
    for i in cand:
        if R.search(texts[i]):
            return i
    return None


def prefix_before(text, R, min_words):
    """Часть смешанного предложения до формульной клаузы (обрезка по последней границе ; , — : « и » « а » « да » перед формулой)."""
    m = R.search(text)
    if not m:
        return None
    pre = text[:m.start()]
    cut = max(pre.rfind(c) for c in [";", ",", "—", ":", " и ", " а ", " да "])
    if cut < 0:
        return None
    pre = re.sub(r"[\s,;:—-]+$", "", pre[:cut]).strip()
    pre = re.sub(r"\s+(и|а|да)$", "", pre)
    if len(V_words(pre)) < min_words:
        return None
    return pre + "."


def V_words(t):
    return re.findall(r"\w+", t)


def load_sentences(cfg, docs):
    out = ROOT / cfg["out_dir"]
    f = out / "sentences.parquet"
    if f.exists():
        return pd.read_parquet(f)
    ecfg = load_config(ROOT / cfg["emotion_config"])
    s01 = ROOT / ecfg["s01_run"]
    want = list(docs)
    parts = [pd.read_parquet(p, columns=["doc_id", "sent_idx", "text", "n_words"], filters=[("doc_id", "in", want)])
             for p in sorted(glob.glob(str(s01 / "sentences" / "*.parquet")))]
    s = pd.concat([p for p in parts if len(p)]).sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    s.to_parquet(f, index=False)
    return s


# ---------------------------------------------------------------- 2.2
def step_sample(cfg):
    sc = cfg["sample"]
    acfg = load_config(ROOT / cfg["arcs_config"])
    out = ROOT / cfg["out_dir"]
    kf = out / "sent_key.parquet"
    rng = np.random.default_rng(cfg["seed"])
    strict, ext, edit = compile_formulas(cfg["formulas"])
    ak = pd.read_parquet(ROOT / acfg["out_dir"] / "arcs_key.parquet")
    docs, _ = llm_arcs(acfg)                                   # только тексты без skip
    if kf.exists():
        key = pd.read_parquet(kf)
        print("ключ уже есть:", kf)
    else:
        tales = docs[docs["stratum"] == "afanasyev"].reset_index(drop=True)
        st = docs[docs["stratum"] == "story"].reset_index(drop=True)
        pa, pb = match_pairs(np.log(tales["n_words"].to_numpy()), np.log(st["n_words"].to_numpy()), sc["caliper_log"], rng)
        print(f"пар «сказка — рассказ» по длине: {len(pa)} (из {len(tales)} сказок, {len(st)} рассказов)")
        keep = rng.choice(len(pa), min(sc["n_story"], len(pa)), replace=False)
        pairs = pd.DataFrame({"tale": tales["doc_id"].to_numpy()[pa], "story": st["doc_id"].to_numpy()[pb]})
        pairs["story_in_sample"] = False
        pairs.loc[keep, "story_in_sample"] = True
        out.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(out / "pairs.parquet", index=False)
        sel = pd.concat([tales.assign(group="tale"), st[st["doc_id"].isin(pairs.loc[pairs["story_in_sample"], "story"])].assign(group="story")])
        sents = load_sentences(cfg, sel["doc_id"])
        segk = ak[ak["doc_id"].isin(set(sel["doc_id"]))][["doc_id", "seg", "seg_sent_start", "seg_sent_end", "text"]]
        rows, uid = [], 0
        last0 = [s - 1 for s in sc["last_segs"]]
        for r in sel.itertuples():
            sk = segk[segk["doc_id"] == r.doc_id].set_index("seg")
            lo, hi = int(sk.loc[last0[0], "seg_sent_start"]), int(sk.loc[last0[-1], "seg_sent_end"])
            s = sents[sents["doc_id"] == r.doc_id].set_index("sent_idx").sort_index()
            zone = s.loc[lo:hi - 1]
            texts = list(zone["text"])
            fb_s = formula_block(texts, strict, edit, cfg["formulas"]["tail_sents"]) if r.group == "tale" else None
            fb_e = formula_block(texts, ext, edit, cfg["formulas"]["tail_sents"]) if r.group == "tale" else None
            for j, (si, t) in enumerate(zip(zone.index, texts)):
                ctx_idx = [k for k in range(si - sc["context_sents"], si) if k in s.index]
                ctx = " ".join(s.loc[ctx_idx, "text"]) if ctx_idx else "(начало текста)"
                seg = int(last0[0] if si < int(sk.loc[last0[-1], "seg_sent_start"]) else last0[-1])
                base = dict(doc_id=r.doc_id, group=r.group, author_id=r.author_id, doc_n_words=r.n_words, seg=seg, sent_idx=int(si),
                            pos_in_zone=j, n_in_zone=len(texts), is_editorial=bool(edit.search(t)),
                            strict_hit=bool(strict.search(t)), ext_hit=bool(ext.search(t)),
                            in_block_strict=fb_s is not None and j >= fb_s, in_block_ext=fb_e is not None and j >= fb_e)
                uid += 1
                rows.append(dict(uid=f"s{uid:05d}", kind="sentence", text=t, context=ctx, **base))
                # часть смешанного предложения до формулы (для обоих уровней списка — по расширенному)
                if r.group == "tale" and base["ext_hit"] and (base["in_block_ext"] or base["in_block_strict"]):
                    pre = prefix_before(t, ext, sc["min_prefix_words"])
                    if pre:
                        uid += 1
                        rows.append(dict(uid=f"s{uid:05d}", kind="prefix", text=pre, context=ctx, **base))
            # последний сегмент без формульного блока — как сегмент (исходная инструкция, без контекста)
            if r.group == "tale" and sc["rerate_last_segment"]:
                for lvl, fb, R in [("strict", fb_s, strict), ("ext", fb_e, ext)]:
                    if fb is None:
                        continue
                    ls = int(sk.loc[last0[-1], "seg_sent_start"]) - lo     # индекс первого предложения последнего сегмента в зоне
                    if fb < ls:
                        part = []                                            # блок начинается в предпоследнем сегменте
                    else:
                        part = [t for t in texts[ls:fb] if not edit.search(t)]
                    pre = prefix_before(texts[fb], R, sc["min_prefix_words"])
                    if pre:
                        part.append(pre)
                    uid += 1
                    rows.append(dict(uid=f"s{uid:05d}", kind=f"lastseg_wo_{lvl}", text=" ".join(part), context=None,
                                     doc_id=r.doc_id, group="tale", author_id=r.author_id, doc_n_words=r.n_words, seg=last0[-1],
                                     sent_idx=-1, pos_in_zone=-1, n_in_zone=len(texts), is_editorial=False, strict_hit=False,
                                     ext_hit=False, in_block_strict=False, in_block_ext=False))
        key = pd.DataFrame(rows)
        empty = key["text"].str.strip().eq("")
        print(f"последний сегмент целиком формульный (нечего переоценивать): {int(empty.sum())}")
        key = key[~empty].reset_index(drop=True)
        key.to_parquet(kf, index=False)
    cm = cfg["cost_model"]
    nw = key["text"].map(lambda t: len(V_words(t))) + key["context"].fillna("").map(lambda t: len(V_words(t)))
    key["est_rub"] = cm["a"] + cm["b"] * nw
    print(key.groupby(["group", "kind"]).agg(запросов=("uid", "size"), текстов=("doc_id", "nunique"),
                                             оценка_руб=("est_rub", "sum")).round(2).to_string())
    print(f"ИТОГО: {len(key)} запросов, ≈ {key['est_rub'].sum():.1f} ₽ (стоп {cfg['stop_at_rub']} ₽)")


# ---------------------------------------------------------------- 2.3–2.4
def step_analyze(cfg):
    acfg = load_config(ROOT / cfg["arcs_config"])
    out = ROOT / cfg["out_dir"]
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["contrast"]["bootstrap"]
    key = pd.read_parquet(out / "sent_key.parquet")
    an = V.read_ann(out / f"sent__{cfg['models']['main']}.csv")
    x = key.merge(an, on="uid", how="left")
    n_miss, n_skip = int(x["tone"].isna().sum()), int((x["skip"] == 1).sum())
    x = x[x["skip"] == 0]
    docs, T = llm_arcs(acfg)
    mid = idx(cfg["contrast"]["middle"])
    dl = docs.set_index("doc_id")
    seg_mid_raw = pd.Series(T[:, mid].mean(1), index=docs["doc_id"])      # сырая LLM-оценка середины (сегменты)
    L = ["## 2.2–2.4. Разметка концовок по предложениям: развязка или финальная формула\n",
         f"Модель {cfg['models']['main']}, та же инструкция; каждое предложение последних 2 сегментов — с 2 предыдущими "
         "предложениями как контекстом (не оцениваются). Сказки базы — все; рассказы — 100 из пар «сказка — рассказ», "
         f"сопоставленных по log(n_words). Ответов без оценки: {n_miss} без ответа, {n_skip} skip — исключены.\n"]
    s = x[x["kind"].isin(["sentence", "prefix"])]
    sent = s[s["kind"] == "sentence"]
    pref = s[s["kind"] == "prefix"].set_index(["doc_id", "sent_idx"])["tone"]
    tales = sent[sent["group"] == "tale"]
    ntale = tales["doc_id"].nunique()
    # 2.3 — формулы
    L.append("### 2.3. Финальные формулы\n")
    rows = []
    per = {}
    for lvl in ["strict", "ext"]:
        blk = tales[f"in_block_{lvl}"]
        has = tales.groupby("doc_id")[f"in_block_{lvl}"].any()
        per[lvl] = has
        bl = tales[blk & ~tales["is_editorial"]]
        rows.append({"список": "строгий (канонические формулы)" if lvl == "strict" else "расширенный (+ клише счастливого конца)",
                     "сказок с формулой": f"{int(has.sum())} из {ntale} ({has.mean():.0%})",
                     "предложений в блоке, медиана": f"{bl.groupby('doc_id').size().median():.0f}",
                     "из них без шаблона (внутри блока)": int((~bl[f"{'strict' if lvl == 'strict' else 'ext'}_hit"]).sum()),
                     "смешанных «развязка + формула» с отдельно оценённой частью до формулы":
                         int(bl.set_index(["doc_id", "sent_idx"]).index.isin(pref.index).sum())})
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append(f"Редакторских пометок (исключены): {int(tales['is_editorial'].sum())} предложений. Шаблоны — "
             "configs/s05_ending.yaml; список проверен глазами по последним 4 предложениям всех сказок базы. "
             "Формульный блок — от первого предложения с шаблоном среди последних 4 содержательных предложений до конца "
             "текста.\n")
    # тональность по частям (на текст): развязка, формула, часть смешанного до формулы
    res = []
    for lvl in ["strict", "ext"]:
        for d, g in tales[~tales["is_editorial"]].groupby("doc_id"):
            blk = g[f"in_block_{lvl}"]
            r = dict(doc_id=d, level=lvl, has_formula=bool(blk.any()))
            den = g.loc[~blk, "tone"]
            # «развязка+»: развязка плюс части смешанных предложений до формулы
            pre = [pref.get((d, si)) for si in g.loc[blk, "sent_idx"] if (d, si) in pref.index]
            r["denouement"] = den.mean() if len(den) else np.nan
            r["denouement_plus"] = np.mean(list(den) + pre) if len(den) + len(pre) else np.nan
            r["formula"] = g.loc[blk, "tone"].mean() if blk.any() else np.nan
            r["all"] = g["tone"].mean()
            r["last3_wo"] = (list(den) + pre)[-3:] and np.mean((list(den) + pre)[-3:])
            r["last3"] = g["tone"].iloc[-3:].mean()
            res.append(r)
    R_ = pd.DataFrame(res)
    st = sent[(sent["group"] == "story") & ~sent["is_editorial"]]
    so = st.groupby("doc_id").agg(all=("tone", "mean"), last3=("tone", lambda v: v.iloc[-3:].mean())).reset_index()
    au_of = lambda ids: dl.loc[ids, "author_id"].astype(str).to_numpy()  # noqa: E731
    L.append("### Тональность частей концовки (сырые оценки −2…+2, среднее по текстам; ДИ — бутстрэп по авторам)\n")
    rows = []
    for lvl in ["strict", "ext"]:
        r = R_[R_["level"] == lvl]
        w = r[r["has_formula"]]
        row = {"список": lvl, "сказок с формулой": len(w)}
        for col, name in [("formula", "формула"), ("denouement", "развязка (предложения до блока)"),
                          ("denouement_plus", "развязка + части смешанных до формулы")]:
            v = w[col].dropna()
            row[name] = fmt(*ci_auth(v.to_numpy(), au_of(w.loc[v.index, "doc_id"]), rng, B))
        dd = (w["formula"] - w["denouement_plus"]).dropna()
        row["формула − развязка+"] = fmt(*ci_auth(dd.to_numpy(), au_of(w.loc[dd.index, "doc_id"]), rng, B))
        rows.append(row)
    L.append(md(pd.DataFrame(rows)) + "\n")
    # 2.4 — сравнение с рассказами
    L.append("### 2.4. Держится ли светлая концовка сказки без формулы — сравнение с рассказами той же длины\n")
    L.append("Сырые баллы по предложениям и разница «минус середина своего текста» (середина — LLM-оценки сегментов "
             f"{cfg['contrast']['middle'][0]}–{cfg['contrast']['middle'][1]}, той же моделью). Единицы «предложение» и «сегмент» "
             "различаются по уровню, поэтому сравнивается разница сказка − рассказ (сдвиг уровня одинаков для обеих групп, "
             "если не зависит от формы).\n")
    rows, store = [], []
    sv = so.set_index("doc_id")
    sa = au_of(sv.index)
    for lab, col_t, col_s in [("все предложения последних 2 сегментов (с формулой)", "all", "all"),
                              ("последние 3 предложения (с формулой)", "last3", "last3"),
                              ("без формулы: развязка + части до формулы", "denouement_plus", "all"),
                              ("без формулы: последние 3 предложения", "last3_wo", "last3")]:
        for lvl in (["strict", "ext"] if "без" in lab else ["strict"]):
            r = R_[R_["level"] == lvl].set_index("doc_id")
            vt = r[col_t].astype(float).dropna()
            ta = au_of(vt.index)
            raw = diff_ci(vt.to_numpy(), ta, sv[col_s].to_numpy(), sa, rng, B)
            rel_t = (vt - seg_mid_raw.reindex(vt.index)).dropna()
            rel_s = (sv[col_s] - seg_mid_raw.reindex(sv.index)).dropna()
            rel = diff_ci(rel_t.to_numpy(), au_of(rel_t.index), rel_s.to_numpy(), au_of(rel_s.index), rng, B)
            rows.append({"мера концовки": lab, "список формул": "—" if "без" not in lab else lvl,
                         "сказка": f"{vt.mean():+.2f}", "рассказ": f"{sv[col_s].mean():+.2f}",
                         "сказка − рассказ, сырые": fmt(*raw),
                         "(конец − середина): сказка": f"{rel_t.mean():+.2f}", "(конец − середина): рассказ": f"{rel_s.mean():+.2f}",
                         "(конец − середина): сказка − рассказ": fmt(*rel)})
            store.append(dict(measure=lab, level=lvl, raw=raw[0], raw_lo=raw[1], raw_hi=raw[2], rel=rel[0], rel_lo=rel[1], rel_hi=rel[2]))
    L.append(md(pd.DataFrame(rows)) + "\n")
    # сказки без формулы — подгруппа на сегментных LLM-дугах
    Z = zscore(T)
    zd = pd.DataFrame(Z, index=docs["doc_id"])
    L.append("### Сказки с формулой и без неё — подъём последнего сегмента на LLM-дугах (z, позиция 20 − середина)\n")
    rows = []
    for lvl in ["strict", "ext"]:
        has = per[lvl]
        for flag, name in [(True, "с формулой"), (False, "без формулы")]:
            ids = has[has == flag].index
            c = (zd.loc[ids, 19] - zd.loc[ids, mid].mean(1)).dropna()
            c2 = (zd.loc[ids, [17, 18]].mean(1) - zd.loc[ids, mid].mean(1)).dropna()
            rows.append({"список": lvl, "сказки": name, "текстов": len(c),
                         "z(20) − середина": fmt(*ci_auth(c.to_numpy(), au_of(c.index), rng, B)),
                         "z(18–19) − середина": fmt(*ci_auth(c2.to_numpy(), au_of(c2.index), rng, B))})
    stz = docs.loc[docs["stratum"] == "story", "doc_id"]
    c = (zd.loc[stz, 19] - zd.loc[stz, mid].mean(1)).dropna()
    c2 = (zd.loc[stz, [17, 18]].mean(1) - zd.loc[stz, mid].mean(1)).dropna()
    rows.append({"список": "—", "сказки": "рассказы той же длины (для сравнения)", "текстов": len(c),
                 "z(20) − середина": fmt(*ci_auth(c.to_numpy(), au_of(c.index), rng, B)),
                 "z(18–19) − середина": fmt(*ci_auth(c2.to_numpy(), au_of(c2.index), rng, B))})
    L.append(md(pd.DataFrame(rows)) + "\n")
    # последний сегмент переоценён без формулы — дуга сказки
    L.append("### Последний сегмент сказки, заново оценённый без формульного блока (как сегмент, исходная инструкция)\n")
    rows = []
    for lvl in ["strict", "ext"]:
        rr = x[x["kind"] == f"lastseg_wo_{lvl}"].set_index("doc_id")["tone"]
        T2 = pd.DataFrame(T.copy(), index=docs["doc_id"])
        ids = [d for d in rr.index if d in T2.index]
        old = T2.loc[ids, 19].copy()
        T2.loc[ids, 19] = rr.loc[ids].to_numpy()
        tl = docs.loc[docs["stratum"] == "afanasyev", "doc_id"]
        Zt = pd.DataFrame(zscore(T2.loc[tl].to_numpy()), index=tl)
        Zs = pd.DataFrame(zscore(T2.loc[stz].to_numpy()), index=stz)
        ct = (Zt[19] - Zt[mid].mean(1)).dropna()
        cs = (Zs[19] - Zs[mid].mean(1)).dropna()
        dif = diff_ci(ct.to_numpy(), au_of(ct.index), cs.to_numpy(), au_of(cs.index), rng, B)
        rows.append({"список": lvl, "переоценено сегментов": len(ids),
                     "сырой балл сегмента 20: с формулой → без": f"{old.mean():+.2f} → {rr.loc[ids].mean():+.2f}",
                     "сказки: z(20) − середина": fmt(*ci_auth(ct.to_numpy(), au_of(ct.index), rng, B)),
                     "сказка − рассказ": fmt(*dif)})
        store.append(dict(measure="lastseg_rerated", level=lvl, raw=np.nan, raw_lo=np.nan, raw_hi=np.nan,
                          rel=dif[0], rel_lo=dif[1], rel_hi=dif[2]))
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("Для сравнения: исходная LLM-дуга, z(20) − середина, сказка − рассказ — см. 2.1.\n")
    # рисунок: тональность предложений по позиции от конца
    fig, ax = plt.subplots(figsize=(7, 3.6))
    K = 12
    for grp, colr, nm in [("tale", "#c0392b", "сказка (все предложения)"), ("story", "#2a5fb0", "рассказ")]:
        g = sent[(sent["group"] == grp) & ~sent["is_editorial"]].copy()
        g["from_end"] = g.groupby("doc_id").cumcount(ascending=False) + 1
        m = g[g["from_end"] <= K].groupby("from_end")["tone"].mean()
        ax.plot(-m.index, m.values, marker="o", ms=3, color=colr, label=nm)
    g = tales[~tales["is_editorial"] & ~tales["in_block_strict"]].copy()
    g["from_end"] = g.groupby("doc_id").cumcount(ascending=False) + 1
    m = g[g["from_end"] <= K].groupby("from_end")["tone"].mean()
    ax.plot(-m.index, m.values, marker="o", ms=3, color="#c0392b", ls="--", label="сказка без формульного блока (строгий список)")
    ax.set_xlabel("предложение от конца текста (−1 — последнее)")
    ax.set_ylabel("средняя тональность предложения (LLM)")
    ax.axhline(0, color="#777", lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fp = ROOT / "reports" / "figures" / "05_ending_sentences.png"
    fig.savefig(fp, dpi=120)
    plt.close(fig)
    L.append(f"![предложения концовки](figures/{fp.name})\n")
    pd.DataFrame(store).to_parquet(out / "ending_tests.parquet", index=False)
    R_.to_parquet(out / "tale_parts.parquet", index=False)
    head = (out / "contrast.md").read_text(encoding="utf-8") if (out / "contrast.md").exists() else ""
    REPORT.write_text("# Ворота 5, п. 2. Подъём концовки волшебной сказки: развязка или финальная формула\n\n" + head + "\n"
                      + "\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--step", choices=["contrast", "sample", "analyze"], required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"contrast": step_contrast, "sample": step_sample, "analyze": step_analyze}[a.step](cfg)


if __name__ == "__main__":
    main()
