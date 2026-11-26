"""Правка v3, п. 4: видят ли лексические сигналы финальную формулу сказки.

Классификатор rubert-tiny2 (P(pos) − P(neg)) и RuSentiLex ((pos − neg) / слов) считаются на тех же предложениях
последних 2 сегментов, что размечала LLM на этапе 5 (п. 2.2), и сравниваются с LLM:
формула (строгий список, блок) / развязка (до блока) у сказок с формулой / все предложения концовок рассказов.
Чтобы сигналы были сравнимы, каждый делится на своё стандартное отклонение по всем этим предложениям.
Различия — среднее по текстам, ДИ — пуассоновский бутстрэп по авторам (как в s05_ending.py).
Классификатор и словарь видят только само предложение; LLM видела ещё 2 предыдущих (контекст не оценивался).

Запуск: .venv/bin/python src/s05_ending_lex.py configs/s05_ending.yaml
Выход:  reports/05_ending_lex.md, data/processed/s05/ending/sent_lex.parquet (кэш: повторно не считается)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s04_nulls import md  # noqa: E402
from s05_ending import ci_auth, diff_ci  # noqa: E402
import s02_validate as V  # noqa: E402

REPORT = ROOT / "reports" / "05_ending_lex.md"
WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁёA-Za-z]+)*")


def lex_scores(texts, ecfg):
    import pymorphy3
    from s02_lexicon import build_lexicons
    rsl = build_lexicons(ecfg)["rsl"]
    morph = pymorphy3.MorphAnalyzer()
    out = []
    for t in texts:
        ws = WORD.findall(t)
        lem = [morph.parse(w.lower())[0].normal_form.replace("ё", "е") for w in ws]
        v = [rsl.get(x) for x in lem]
        pos, neg = sum(1 for a in v if a is not None and a > 0), sum(1 for a in v if a is not None and a < 0)
        out.append((pos - neg) / max(1, len(ws)))
    return np.array(out)


def clf_scores(texts, ecfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    m = ecfg["models"]["sentiment"]
    dev = ecfg["models"]["device"]
    tok = AutoTokenizer.from_pretrained(m["hf"], revision=m["revision"])
    mod = AutoModelForSequenceClassification.from_pretrained(m["hf"], revision=m["revision"]).to(dev).eval()
    lab = {v.lower(): int(k) for k, v in mod.config.id2label.items()}
    res = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(list(texts[i:i + 64]), truncation=True, max_length=ecfg["models"]["max_tokens"],
                    padding=True, return_tensors="pt").to(dev)
            p = torch.softmax(mod(**b).logits.float(), -1).cpu().numpy()
            res.append(p[:, lab["positive"]] - p[:, lab["negative"]])
    return np.concatenate(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    ecfg = load_config(ROOT / cfg["emotion_config"])
    out = ROOT / cfg["out_dir"]
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["contrast"]["bootstrap"]
    key = pd.read_parquet(out / "sent_key.parquet")
    x = key[key["kind"] == "sentence"].merge(V.read_ann(out / f"sent__{cfg['models']['main']}.csv"), on="uid", how="left")
    x = x[(x["skip"] == 0) & ~x["is_editorial"]].copy()
    cache = out / "sent_lex.parquet"
    if cache.exists():
        lx = pd.read_parquet(cache)
    else:
        lx = pd.DataFrame({"uid": x["uid"].to_numpy(), "clf": clf_scores(x["text"].tolist(), ecfg),
                           "rsl": lex_scores(x["text"].tolist(), ecfg)})
        lx.to_parquet(cache, index=False)
    x = x.merge(lx, on="uid")
    sig = {"tone": "LLM", "clf": "классификатор", "rsl": "RuSentiLex"}
    for s_ in sig:
        x[s_ + "_z"] = x[s_] / x[s_].std()
    tales = x[x["group"] == "tale"]
    has = tales.groupby("doc_id")["in_block_strict"].any()
    w = tales[tales["doc_id"].isin(has[has].index)]
    form, den = w[w["in_block_strict"]], w[~w["in_block_strict"]]
    story = x[x["group"] == "story"]
    au = lambda d: d.groupby("doc_id")["author_id"].first()  # noqa: E731
    L = ["# Правка v3, п. 4: видят ли классификатор и RuSentiLex финальную формулу\n",
         f"Предложения последних 2 сегментов (этап 5, п. 2.2; без skip и редакторских пометок). Сказки с формулой "
         f"(строгий список): {w['doc_id'].nunique()}; предложений формулы {len(form)}, развязки {len(den)}; рассказы: "
         f"{story['doc_id'].nunique()} текстов, {len(story)} предложений. Каждый сигнал поделён на своё SD по всем "
         "этим предложениям (единицы SD). Классификатор и словарь видят только предложение, LLM — ещё 2 предыдущих.\n"]
    rows = []
    for s_, nm in sig.items():
        c = s_ + "_z"
        f_ = form.groupby("doc_id")[c].mean()
        d_ = den.groupby("doc_id")[c].mean()
        s0 = story.groupby("doc_id")[c].mean()
        dd = (f_ - d_).dropna()
        a_w, a_s = au(w), au(story)
        r = {"сигнал": nm,
             "формула": f"{f_.mean():+.2f}", "развязка": f"{d_.mean():+.2f}", "рассказ": f"{s0.mean():+.2f}",
             "формула − развязка": "{:+.2f} [{:+.2f}; {:+.2f}]".format(*ci_auth(dd.to_numpy(), a_w[dd.index].to_numpy(), rng, B)),
             "формула − рассказ": "{:+.2f} [{:+.2f}; {:+.2f}]".format(*diff_ci(f_.to_numpy(), a_w[f_.index].to_numpy(), s0.to_numpy(), a_s[s0.index].to_numpy(), rng, B)),
             "развязка − рассказ": "{:+.2f} [{:+.2f}; {:+.2f}]".format(*diff_ci(d_.dropna().to_numpy(), a_w[d_.dropna().index].to_numpy(), s0.to_numpy(), a_s[s0.index].to_numpy(), rng, B)),
             "доля формул > 0": f"{(form[s_] > 0).mean():.0%}", "доля развязки > 0": f"{(den[s_] > 0).mean():.0%}"}
        if s_ != "tone":
            r["ρ с LLM: формула / развязка / рассказ"] = " / ".join(
                f"{st.spearmanr(g['tone'], g[s_]).statistic:.2f}" for g in (form, den, story))
        r["у формул нет ни одного слова словаря"] = f"{(form['rsl'] == 0).mean():.0%}" if s_ == "rsl" else ""
        rows.append(r)
    L.append(md(pd.DataFrame(rows)) + "\n")
    L.append("Примеры формул с наименьшим сигналом классификатора:\n")
    ex = form.sort_values("clf").head(8)[["text", "tone", "clf", "rsl"]]
    ex["text"] = ex["text"].str.slice(0, 120)
    L.append(md(ex.round(2)) + "\n")
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
