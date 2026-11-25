"""Этап 3. Отчёт для Ворот 3: темы, позиционные профили (Schmidt 2015), метрики траектории по формам.

Запуск:
    .venv/bin/python src/s03_report.py configs/s03_topics.yaml --K 50 [--sample]
Пишет reports/03_topics[_sample]_K{K}.md и рисунки reports/figures/03_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

FORMS = ["tale_ru", "tale_world", "story", "novella", "povest", "novel", "fanfic", "short_series"]
FORM_RU = {"tale_ru": "волшебная сказка (Афанасьев)", "tale_world": "сказка (fairytalez)", "story": "рассказ",
           "novella": "новелла", "povest": "повесть", "novel": "роман", "fanfic": "фанфик", "short_series": "короткая серия"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (not isinstance(v, str) and pd.isna(v)) else
                                     (f"{v:,}".replace(",", " ") if isinstance(v, (int, np.integer)) else
                                      f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def form_key(a: pd.DataFrame) -> pd.Series:
    f = a["form"].copy()
    f[a["corpus"] == "afanasyev"] = None
    f[a["base_magic600"].fillna(False).astype(bool)] = "tale_ru"
    f[a["corpus"] == "fairytales"] = "tale_world"
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    K = args.K
    tag = "sample" if args.sample else "full"
    base = ROOT / cfg["out_dir"] / tag
    t = json.loads((base / f"topics_K{K}.json").read_text())
    voc = json.loads((base / "vocab.json").read_text())
    a = pd.read_parquet(ROOT / cfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(["text", "linked_work"])].copy()
    a["fk"] = form_key(a)
    a = a[a["fk"].notna()]
    rep = ROOT / "reports"
    figd = rep / "figures"
    figd.mkdir(parents=True, exist_ok=True)
    sfx = f"{'_sample' if args.sample else ''}_K{K}"

    # ---------------- темы
    prev_s = pd.DataFrame(t["prevalence_by_stratum"]).T
    prev_c = pd.DataFrame(t["prevalence_by_corpus"]).T
    th20 = pd.read_parquet(base / f"theta_K{K}" / "theta_N20.parquet").merge(a[["doc_id", "fk"]], on="doc_id")
    X = np.vstack(th20["theta"].to_numpy())
    pos = th20["seg"].to_numpy()
    fk = th20["fk"].to_numpy()
    rows = []
    for k in range(K):
        kk = str(k)
        r = dict(тема=k, c_v=t["coherence_c_v"][kk], слова=", ".join(t["top_words"][kk][:12]))
        for c_ in ["afanasyev", "fairytales"]:
            r[f"доля_{c_}"] = prev_c.loc[c_, kk] if c_ in prev_c.index else np.nan
        others = [s for s in prev_s.index if s != "tale"]
        r["доля_прочие_формы"] = float(prev_s.loc[others, kk].mean())
        rows.append(r)
    topics = pd.DataFrame(rows)
    # «сказочные» темы: доля у Афанасьева или fairytalez в 3+ раза выше средней по прочим формам
    topics["сказочная"] = (topics[["доля_afanasyev", "доля_fairytales"]].max(axis=1) > 3 * topics["доля_прочие_формы"])
    L = [f"# Этап 3. Темы и позиционные профили — LDA K = {K}{' (ВЫБОРКА)' if args.sample else ''}\n",
         f"Обучающая выборка: {t['n_docs']} документов по 200 содержательных лемм, поровну на 6 групп форм "
         f"(сказка = Афанасьев + fairytalez до нормы); словарь {t['vocab']} лемм; log-likelihood/слово {t['ll_per_word']:.3f}. "
         f"Слишком частые (> 30 % документов) и удалённые: {', '.join(voc['removed_frequent'][:25])}.\n",
         f"Средняя когерентность c_v: {np.mean(list(t['coherence_c_v'].values())):.3f}. "
         f"«Сказочных» тем (доля у Афанасьева или fairytalez > 3× средней по прочим формам): {int(topics['сказочная'].sum())}.\n",
         "## Темы\n", md_table(topics) + "\n"]

    # ---------------- позиционные профили (Schmidt): средняя доля темы по позиции, по формам
    L.append("## Позиционные профили (N = 20)\n")
    L.append("Для каждой формы — темы с самым сильным позиционным эффектом: отношение средней доли темы в последних 3 "
             "сегментах к первым 3 (≫ 1 — тема концовки, ≪ 1 — тема начала); показаны темы со средней долей ≥ 1 %.\n")
    prof = {}
    for f in FORMS:
        m = fk == f
        if not m.any():
            continue
        P = np.vstack([X[m & (pos == p)].mean(0) for p in range(20)])
        prof[f] = P
        mean = P.mean(0)
        ratio = (P[-3:].mean(0) + 1e-6) / (P[:3].mean(0) + 1e-6)
        ok = np.where(mean >= 0.01)[0]
        end = ok[np.argsort(-ratio[ok])][:4]
        beg = ok[np.argsort(ratio[ok])][:4]
        fmt = lambda ks: "; ".join(f"{k} ({', '.join(t['top_words'][str(k)][:4])}) ×{ratio[k]:.2f}" for k in ks)  # noqa: E731
        L.append(f"- **{FORM_RU[f]}** — концовка: {fmt(end)}")
        L.append(f"  — начало: {fmt(beg)}")
    L.append("")
    # рисунок: для каждой формы профиль 6 тем с самым большим размахом по позиции
    fig, axes = plt.subplots(len(prof), 1, figsize=(8, 2.0 * len(prof)), sharex=True)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    for ax, (f, P) in zip(np.atleast_1d(axes), prof.items()):
        rng_ = P.max(0) - P.min(0)
        top = np.argsort(-rng_)[:6]
        for c_, k in zip(pal, top):
            ax.plot(np.arange(1, 21), P[:, k], color=c_, lw=2, label=f"{k}: {', '.join(t['top_words'][str(k)][:3])}")
        ax.set_title(FORM_RU[f], fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.grid(axis="y", color=GRID, lw=0.5)
    np.atleast_1d(axes)[-1].set_xlabel("позиция в тексте (сегмент из 20)", fontsize=8, color=INK2)
    fig.suptitle(f"Позиционные профили тем по формам (LDA K = {K})", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fp = figd / f"03_positional_profiles{sfx}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    L.append(f"![Позиционные профили](figures/{fp.name})\n")

    # ---------------- траектории
    L.append("## Метрики траектории по формам\n")
    for n in [10, 20]:
        tr = pd.read_parquet(base / f"topic_traj_N{n}_K{K}.parquet").merge(a[["doc_id", "fk", "author_id"]], on="doc_id")
        g = tr.groupby("fk").agg(текстов=("doc_id", "size"), лемм_в_сегменте=("n_content_median", "median"),
                                 скорость=("speed", "median"), объём=("volume", "median"),
                                 извилистость=("circuitousness", "median"), поз_концентрация=("pos_concentration", "median"))
        g = g.reindex([f for f in FORMS if f in g.index]).reset_index()
        g["fk"] = g["fk"].map(FORM_RU)
        L.append(f"### N = {n} (медианы)\n")
        L.append(md_table(g.rename(columns={"fk": "форма"})) + "\n")
        if n == 20:
            cor = tr[["speed", "volume", "circuitousness", "pos_concentration"]].corrwith(np.log(tr["n_content_median"].clip(lower=1)),
                                                                                       method="spearman")
            L.append("Ранговая корреляция метрик с log(содержательных лемм в сегменте), все тексты: "
                     + ", ".join(f"{k} {v:+.2f}" for k, v in cor.items()) + ".\n")
            # внутри одного диапазона размера сегмента
            tr["bin"] = pd.cut(tr["n_content_median"], [0, 20, 40, 80, 160, 400, 1e9],
                               labels=["≤20", "21–40", "41–80", "81–160", "161–400", ">400"])
            bt = tr.groupby(["bin", "fk"], observed=True).agg(n=("doc_id", "size"), скорость=("speed", "median"),
                                                              поз_концентрация=("pos_concentration", "median")).reset_index()
            bt = bt[bt["n"] >= 30]
            bt["fk"] = bt["fk"].map(FORM_RU)
            L.append("Внутри диапазонов размера сегмента (содержательных лемм на сегмент; группы ≥ 30 текстов):\n")
            L.append(md_table(bt.rename(columns={"bin": "лемм_в_сегм", "fk": "форма"})) + "\n")
    L.append("**Важно:** скорость, объём и позиционная концентрация сильно зависят от размера сегмента: длинный сегмент "
             "усредняет темы, и траектория становится глаже независимо от сюжета (тот же эффект, что для эмоциональных дуг, "
             "Ворота 2, п. 4). Сравнение форм по этим метрикам — только при сопоставимом размере сегмента и с контролем шума "
             "(этап 4).\n")
    (rep / f"03_topics{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    print("записано:", rep / f"03_topics{sfx}.md")


if __name__ == "__main__":
    main()
