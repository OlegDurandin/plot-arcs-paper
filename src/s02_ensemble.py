"""Этап 2, Ворота 2: (1) ансамбль сигналов тональности, откалиброванный на серебре по LLM;
(2.1) зависит ли ошибка сигнала от позиции сегмента в тексте (формулы зачина / концовки сказок).

(1) Признаки — вероятности классификатора, RSL, LinisCrowd, NRC, log(слов); цель — LLM-тональность серебра
    (deepseek-v4.1-flash). Модели: ridge и монотонный градиентный бустинг. Кросс-валидация GroupKFold по авторам
    (out-of-fold ρ по стратам). Отложенная проверка — консенсус 6 LLM на золоте (другие модели и другие сегменты):
    обучение на всём серебре, ρ на художественной прозе и на Афанасьеве, парная разница с классификатором
    (бутстрэп по авторам).
(2.1) Для каждой формы и сигнала: калибровка внутри формы tone ~ a + b·сигнал (снимает сдвиг уровня и масштаб),
    остаток = LLM − калибровка; средний остаток по номеру сегмента 1…20 и контрасты «первые/последние 2 против
    середины», ДИ — бутстрэп по авторам. Если ошибка зависит от позиции, z-нормализация дуги её не уберёт.

Запуск:  .venv/bin/python src/s02_ensemble.py configs/s02_ensemble.yaml
Выход:   reports/02_ensemble_position.md, reports/figures/02_position_residuals.png,
         data/processed/s02/llm_api/ensemble_{ridge,hgb}.json (коэффициенты / параметры)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
import s02_validate as V  # noqa: E402
from s02_validate_llm import consensus  # noqa: E402

ST = ["afanasyev", "fairytales", "fanfic", "story", "novella", "povest", "novel"]
ST_RU = {**V.ST_RU, "fairytales": "сказка (fairytalez)", "fiction": "художеств. проза"}


def feats(df: pd.DataFrame, names) -> np.ndarray:
    d = df.copy()
    d["log_words"] = np.log(d["shown_n_words"].clip(lower=1))
    return d[names].fillna(0.0).to_numpy(float)


def boot_groups(groups, rng, B):
    g = np.asarray(groups)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    for _ in range(B):
        yield np.concatenate([idx[u] for u in rng.choice(ug, len(ug), replace=True)])


def rho_ci(x, y, groups, rng, B):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = stats.spearmanr(x, y).statistic
    bs = [stats.spearmanr(x[t], y[t]).statistic for t in boot_groups(groups, rng, B)]
    return r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def paired_diff(a, b, y, groups, rng, B):
    """ρ(a, y) − ρ(b, y) с бутстрэпом по авторам."""
    a, b, y = (np.asarray(v, float) for v in (a, b, y))
    d0 = stats.spearmanr(a, y).statistic - stats.spearmanr(b, y).statistic
    bs = [stats.spearmanr(a[t], y[t]).statistic - stats.spearmanr(b[t], y[t]).statistic for t in boot_groups(groups, rng, B)]
    return d0, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def f3(r, lo, hi):
    return f"{r:.2f} [{lo:.2f}; {hi:.2f}]"


def fd(r, lo, hi):
    return f"{r:+.2f} [{lo:+.2f}; {hi:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    lcfg = load_config(ROOT / cfg["llm_config"])
    ecfg = load_config(ROOT / lcfg["emotion_config"])
    B = cfg["bootstrap"]
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / lcfg["out_dir"]
    ec = cfg["ensemble"]
    names = ec["features"]

    # ---------- данные ----------
    key = pd.read_parquet(out / "silver_key.parquet", columns=["uid", "seg"])
    sig = pd.read_parquet(out / "silver_signals.parquet").drop(columns=["text"])
    ann = V.read_ann(out / f"silver__{lcfg['models']['main']}.csv")
    d = sig.merge(ann, on="uid").merge(key, on="uid")
    d = d[d["skip"] == 0].reset_index(drop=True)
    X, y, g = feats(d, names), d["tone"].to_numpy(float), d["author_id"].astype(str).to_numpy()

    def ridge():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=ec["ridge_alphas"]))

    def hgb():
        return HistGradientBoostingRegressor(random_state=cfg["seed"], monotonic_cst=[ec["monotone"][n] for n in names],
                                             **ec["hgb"])

    models = {"ensemble_ridge": ridge, "ensemble_hgb": hgb}
    oof = {m: np.zeros(len(d)) for m in models}
    for tr, te in GroupKFold(n_splits=ec["cv_folds"]).split(X, y, g):
        for m, mk in models.items():
            oof[m][te] = mk().fit(X[tr], y[tr]).predict(X[te])
    for m in models:
        d[m] = oof[m]

    R = ["# Этап 2. Ансамбль сигналов и позиционная ошибка сигнала (Ворота 2, п. 1 и 2.1)\n",
         f"Серебро: {len(d)} сегментов N = 20, цель — LLM-тональность ({lcfg['models']['main']}). Признаки ансамбля: "
         f"{', '.join(names)}; форма текста в признаки не входит. ρ Спирмена [95 % ДИ, бутстрэп по авторам, B = {B}].\n",
         "## 1. Ансамбль\n", "### 1.1. Серебро, out-of-fold (GroupKFold по авторам, 5 блоков)\n"]
    sig_cols = ["Классификатор_тональности", "RSL", "ensemble_ridge", "ensemble_hgb"]
    rows = []
    for s_ in sig_cols:
        r = {"сигнал": s_}
        for st in ["fiction"] + ST:
            x = d[~d["stratum"].isin(["afanasyev", "fairytales"])] if st == "fiction" else d[d["stratum"] == st]
            r[ST_RU.get(st, st)] = f3(*rho_ci(x[s_], x["tone"], x["author_id"], rng, B))
        rows.append(r)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    R.append("Парная разница ρ(ансамбль) − ρ(классификатор), серебро out-of-fold:\n")
    rows = []
    for m in models:
        r = {"ансамбль": m}
        for st in ["fiction", "afanasyev", "fairytales"]:
            x = d[~d["stratum"].isin(["afanasyev", "fairytales"])] if st == "fiction" else d[d["stratum"] == st]
            r[ST_RU.get(st, st)] = fd(*paired_diff(x[m], x["Классификатор_тональности"], x["tone"], x["author_id"], rng, B))
        rows.append(r)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")

    # финальные модели на всём серебре
    fitted = {m: mk().fit(X, y) for m, mk in models.items()}
    rp = fitted["ensemble_ridge"]
    coef = dict(zip(names, rp[-1].coef_.round(4).tolist()))
    (out / "ensemble_ridge.json").write_text(json.dumps({"features": names, "alpha": float(rp[-1].alpha_),
                                                          "scaler_mean": rp[0].mean_.tolist(), "scaler_scale": rp[0].scale_.tolist(),
                                                          "coef_std": rp[-1].coef_.tolist(), "intercept": float(rp[-1].intercept_)},
                                                         ensure_ascii=False, indent=1))
    with open(out / "ensemble_models.pkl", "wb") as f:
        pickle.dump(fitted, f)
    R.append(f"Ridge (стандартизованные признаки, α = {rp[-1].alpha_:g}): " +
             ", ".join(f"{k} {v:+.3f}" for k, v in coef.items()) + "\n")

    # ---------- отложенная проверка: консенсус золота ----------
    s02 = ROOT / ecfg["out_dir"]
    gkey = pd.read_parquet(s02 / "gold_key.parquet")
    orig = gkey[gkey["repeat_of"].isna()]
    anns = {}
    for f in sorted(out.glob("gold__*.csv")):
        a = V.read_ann(f)
        if a["uid"].isin(orig["uid"]).sum() >= 0.9 * len(orig):
            anns[f.stem.split("__", 1)[1]] = a
    cons = consensus(anns, list(orig["uid"]))
    gs = pd.read_parquet(s02 / "gold_signals.parquet").drop(columns=["text"]).merge(cons, on="uid")
    gs = gs[gs["skip"] == 0].reset_index(drop=True)
    Xg = feats(gs, names)
    for m in models:
        gs[m] = fitted[m].predict(Xg)
    R.append(f"### 1.2. Отложенная проверка: консенсус {len(anns)} LLM на золоте (обучение — всё серебро)\n")
    rows = []
    for s_ in sig_cols:
        r = {"сигнал": s_}
        for nm, x in [("художеств. проза", gs[gs["stratum"] != "afanasyev"]), ("Афанасьев", gs[gs["stratum"] == "afanasyev"])]:
            r[f"{nm} (n = {len(x)})"] = f3(*rho_ci(x[s_], x["tone"], x["author_id"], rng, B))
        rows.append(r)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    rows = []
    verdict = {}
    for m in models:
        r = {"ансамбль − классификатор": m}
        for nm, x in [("художеств. проза", gs[gs["stratum"] != "afanasyev"]), ("Афанасьев", gs[gs["stratum"] == "afanasyev"])]:
            res = paired_diff(x[m], x["Классификатор_тональности"], x["tone"], x["author_id"], rng, B)
            r[nm] = fd(*res)
            verdict[(m, nm)] = res
        rows.append(r)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")

    # критерий пользователя
    R.append("### 1.3. Критерий (решение Ворот 2)\n")
    for m in models:
        tale_rho = stats.spearmanr(d.loc[d["stratum"] == "afanasyev", m], d.loc[d["stratum"] == "afanasyev", "tone"]).statistic
        dt_s = paired_diff(*(d.loc[d["stratum"] == "afanasyev", c] for c in (m, "RSL", "tone")),
                           d.loc[d["stratum"] == "afanasyev", "author_id"], rng, B)
        fic = d[~d["stratum"].isin(["afanasyev", "fairytales"])]
        dp_s = paired_diff(fic[m], fic["Классификатор_тональности"], fic["tone"], fic["author_id"], rng, B)
        dp_g = verdict[(m, "художеств. проза")]
        ok_tale = tale_rho >= ec["tale_rho_ref"] + ec["tale_margin"] and dt_s[1] > 0
        ok_prose = dp_s[1] > -ec["prose_noninferiority"] and dp_g[1] > -0.10
        R.append(f"- **{m}**: Афанасьев (серебро, OOF) ρ = {tale_rho:.2f}; разница с RSL {fd(*dt_s)} → "
                 f"{'заметно выше' if ok_tale else 'не заметно выше'} 0,37. Проза: разница с классификатором "
                 f"серебро {fd(*dp_s)}, золото {fd(*dp_g)} → {'не хуже' if ok_prose else 'хуже'}. "
                 f"**Итог: {'кандидат в главный сигнал' if ok_tale and ok_prose else 'не проходит'}**.")
    R.append("")

    # ---------- 2.1 позиционная ошибка ----------
    pc = cfg["position"]
    E = pc["edges"]
    R.append("## 2. Зависит ли ошибка сигнала от позиции сегмента (п. 2.1)\n")
    R.append("Калибровка внутри формы tone ~ a + b·сигнал (снимает сдвиг уровня и масштаб формы); остаток = LLM − калибровка, "
             "в баллах шкалы −2…+2. Положительный остаток — сигнал **занижает** тональность на этой позиции. "
             f"Контраст: средний остаток первых {E} / последних {E} сегментов минус средний остаток середины "
             f"(сегменты {E + 1}…{20 - E}); 95 % ДИ — бутстрэп по авторам. Для ансамбля — out-of-fold прогноз.\n")
    d["pos"] = d["seg"] + 1
    resid = {}
    rows = []
    for s_ in pc["signals"]:
        for st in ST:
            x = d[d["stratum"] == st]
            b, a = np.polyfit(x[s_], x["tone"], 1)
            r_ = (x["tone"] - (a + b * x[s_])).to_numpy()
            pos, grp = x["pos"].to_numpy(), x["author_id"].astype(str).to_numpy()
            resid[(s_, st)] = (pos, r_, grp)
            head, tail, mid = pos <= E, pos > 20 - E, (pos > E) & (pos <= 20 - E)

            def contrasts(t):
                p_, rr = pos[t], r_[t]
                return rr[p_ <= E].mean() - rr[(p_ > E) & (p_ <= 20 - E)].mean(), rr[p_ > 20 - E].mean() - rr[(p_ > E) & (p_ <= 20 - E)].mean()
            c0 = contrasts(np.arange(len(r_)))
            bs = np.array([contrasts(t) for t in boot_groups(grp, rng, B)])
            lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
            flag = lambda i: " ⚠" if lo[i] > 0 or hi[i] < 0 else ""  # noqa: E731
            rows.append({"сигнал": s_, "форма": ST_RU.get(st, st), "n": len(x), "n начало": int(head.sum()), "n конец": int(tail.sum()),
                         "начало − середина": fd(c0[0], lo[0], hi[0]) + flag(0),
                         "конец − середина": fd(c0[1], lo[1], hi[1]) + flag(1)})
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    R.append("⚠ — 95 % ДИ не содержит 0.\n")

    # по позициям — для сказок и рассказов, главный и запасной сигнал
    R.append("### Средний остаток по позиции 1…20 (сказки Афанасьева)\n")
    rows = []
    for s_ in ["Классификатор_тональности", "RSL", "ensemble_ridge"]:
        pos, r_, grp = resid[(s_, "afanasyev")]
        r = {"сигнал": s_}
        for p_ in range(1, 21):
            r[str(p_)] = f"{r_[pos == p_].mean():+.2f}" if (pos == p_).sum() else "—"
        rows.append(r)
    nrow = {"сигнал": "n сегментов", **{str(p_): str(int((resid[('RSL', 'afanasyev')][0] == p_).sum())) for p_ in range(1, 21)}}
    rows.append(nrow)
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")

    # рисунок
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    colors = {"afanasyev": "#c0392b", "fairytales": "#e67e22", "story": "#2a5fb0", "fanfic": "#27ae60",
              "novella": "#8e44ad", "povest": "#7f8c8d", "novel": "#2c3e50"}
    bins = [(1, 2), (3, 5), (6, 8), (9, 12), (13, 15), (16, 18), (19, 20)]
    centers = [np.mean(b_) for b_ in bins]
    for ax, s_ in zip(axes, ["Классификатор_тональности", "RSL", "ensemble_ridge"]):
        for st in ST:
            pos, r_, _ = resid[(s_, st)]
            ax.plot(centers, [r_[(pos >= lo_) & (pos <= hi_)].mean() for lo_, hi_ in bins], lw=2 if st == "afanasyev" else 1.2,
                    color=colors[st], label=ST_RU.get(st, st), marker="o", ms=4)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_title(s_)
        ax.set_xlabel("позиция сегмента (N = 20, интервалы)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("остаток LLM − калибровка (баллы)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fp = ROOT / "reports" / "figures" / "02_position_residuals.png"
    fig.savefig(fp, dpi=130)
    R.append(f"![Остаток по позиции](figures/{fp.name})\n")
    (ROOT / "reports" / "02_ensemble_position.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
