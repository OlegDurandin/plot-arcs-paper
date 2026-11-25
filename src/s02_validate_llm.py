"""Этап 2. Валидация сигналов по LLM-разметке (решение пользователя на Воротах 2: человеческая разметка отложена,
студенты — позже; до неё опорой служат LLM).

Золото (430 = 400 + 30 скрытых повторов), несколько моделей:
  * согласие каждой модели с собой (повторы) и моделей между собой;
  * консенсус (медиана тональности по моделям, эмоция — строгое большинство) и согласие каждой модели
    с консенсусом остальных (leave-one-out);
  * сигналы против консенсуса.
Серебро (11 400 сегментов, основная модель):
  * сигналы против LLM по стратам (ρ, бутстрэп по авторам);
  * сдвиг уровня по форме: калибровка tone ~ a + b·z(сигнал) на всех стратах, средний остаток по страте
    (если сигнал систематически занижает/завышает тональность одной формы — сравнение форм по нему смещено).

Запуск:
    .venv/bin/python src/s02_validate_llm.py configs/s02_llm_api.yaml
Выход: reports/02_validation_llm.md; кэш сигналов silver_signals.parquet.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
import s02_validate as V  # noqa: E402

V.STRATA = ["afanasyev", "fairytales", "fanfic", "story", "novella", "povest", "novel"]
V.ST_RU["fairytales"] = "сказка (fairytalez, перевод)"


def consensus(anns: dict, uids, min_models=3) -> pd.DataFrame:
    t = pd.DataFrame({m: a.set_index("uid")["tone"].where(a.set_index("uid")["skip"] == 0) for m, a in anns.items()}).reindex(uids)
    e = pd.DataFrame({m: a.set_index("uid")["emotion"].where(a.set_index("uid")["skip"] == 0) for m, a in anns.items()}).reindex(uids)
    n = t.notna().sum(1)
    tone = t.median(1).where(n >= min_models)

    def maj(row):
        v = row.dropna()
        v = v[v != ""]
        if len(v) < min_models:
            return ""
        c = v.value_counts()
        return c.index[0] if c.iloc[0] > len(v) / 2 else ""
    emo = e.apply(maj, axis=1)
    return pd.DataFrame({"uid": uids, "tone": tone.to_numpy(), "emotion": emo.to_numpy(),
                         "skip": tone.isna().astype(int).to_numpy(), "n_models": n.to_numpy()})


def fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ecfg = load_config(ROOT / cfg["emotion_config"])
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / cfg["out_dir"]
    s02 = ROOT / ecfg["out_dir"]
    R = ["# Этап 2. Валидация сигналов по LLM-разметке\n",
         "**Статус.** Человеческая разметка отложена (решение Ворот 2); опорная разметка — LLM через provod.ai, "
         "температура 0, одна и та же инструкция, модели видят только текст фрагмента. Это **не** замена человеческому "
         "золоту: результаты ниже — согласие сигналов с LLM, их нужно будет перепроверить на разметке студентов.\n"]

    # ---------- золото ----------
    key = pd.read_parquet(s02 / "gold_key.parquet")
    orig = key[key["repeat_of"].isna()]
    rep = key[key["repeat_of"].notna()][["uid", "repeat_of"]]
    raw = {Path(f).stem.split("__", 1)[1]: pd.read_csv(f) for f in sorted(glob.glob(str(out / "gold__*.csv")))}
    anns = {m: V.read_ann(f) for m, f in ((m, out / f"gold__{m}.csv") for m in raw)}
    anns = {m: a for m, a in anns.items() if a["uid"].isin(orig["uid"]).sum() >= 0.9 * len(orig)}   # только полные прогоны
    R.append(f"## 1. Золото: {len(orig)} фрагментов + {len(rep)} скрытых повторов, моделей: {len(anns)}\n")
    rows = []
    for m, a in anns.items():
        r0 = raw[m]
        ao = a[a["uid"].isin(orig["uid"])]
        g = a.set_index("uid")
        p = rep.join(g, on="uid").join(g, on="repeat_of", rsuffix="_1")
        p = p[(p["skip"] == 0) & (p["skip_1"] == 0)]
        rows.append({"модель": m, "размечено": int((ao["skip"] == 0).sum()), "пропуск/брак": int((ao["skip"] == 1).sum()),
                     "средняя тональность": ao.loc[ao["skip"] == 0, "tone"].mean(),
                     "доля «нейтрально»": float((ao.loc[ao["skip"] == 0, "emotion"] == "нейтрально").mean()),
                     "повторы: ρ": stats.spearmanr(p["tone"], p["tone_1"]).statistic if len(p) > 5 else np.nan,
                     "повторы: совпад. тональн.": float((p["tone"] == p["tone_1"]).mean()) if len(p) else np.nan,
                     "повторы: κ эмоции": V.cohen_kappa(p["emotion"], p["emotion_1"]) if len(p) > 5 else np.nan,
                     "рассужд. ток./ответ": pd.to_numeric(r0["reasoning_tokens"], errors="coerce").fillna(0).mean(),
                     "₽": pd.to_numeric(r0["cost_rub"], errors="coerce").sum()})
    R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    R.append("Повторы — те же 30 фрагментов, отправленные повторно (температура 0); согласие ниже 1 — недетерминизм API.\n")

    ms = list(anns)
    T = pd.DataFrame({m: anns[m].set_index("uid")["tone"].where(anns[m].set_index("uid")["skip"] == 0) for m in ms}).reindex(orig["uid"])
    E = pd.DataFrame({m: anns[m].set_index("uid")["emotion"].where(anns[m].set_index("uid")["skip"] == 0) for m in ms}).reindex(orig["uid"])
    mat = pd.DataFrame(index=ms, columns=ms, dtype=object)
    for i in ms:
        for j in ms:
            if i == j:
                mat.loc[i, j] = "—"
                continue
            ok = T[i].notna() & T[j].notna()
            rho = stats.spearmanr(T.loc[ok, i], T.loc[ok, j]).statistic
            ok2 = ok & (E[i] != "") & (E[j] != "")
            mat.loc[i, j] = f"{rho:.2f} / {V.cohen_kappa(E.loc[ok2, i], E.loc[ok2, j]):.2f}"
    R.append("### Согласие моделей между собой (ρ тональности / κ эмоции)\n")
    R.append(V.md_table(mat.reset_index().rename(columns={"index": "модель"})) + "\n")

    loo = []
    for m in ms:
        c = consensus({k: v for k, v in anns.items() if k != m}, list(orig["uid"]), min_models=min(3, len(ms) - 1))
        x = c.set_index("uid").join(anns[m].set_index("uid"), rsuffix="_m")
        x = x[(x["skip"] == 0) & (x["skip_m"] == 0)]
        x2 = x[(x["emotion"] != "") & (x["emotion_m"] != "")]
        loo.append({"модель": m, "ρ с консенсусом остальных": stats.spearmanr(x["tone"], x["tone_m"]).statistic,
                    "κ эмоции": V.cohen_kappa(x2["emotion"], x2["emotion_m"]), "n": len(x)})
    R.append("### Каждая модель против консенсуса остальных (leave-one-out)\n")
    R.append(V.md_table(pd.DataFrame(loo)) + "\n")

    cons = consensus(anns, list(orig["uid"]))
    R.append(f"Консенсус всех моделей: тональность — медиана (≥ 3 моделей), эмоция — строгое большинство; "
             f"эмоция определена для {int((cons['emotion'] != '').sum())} из {len(cons)}.\n")
    sig = pd.read_parquet(s02 / "gold_signals.parquet")
    R.append("## 2. Сигналы против консенсуса LLM на золоте\n")
    R += V.validate(ecfg, sig, cons, "консенсус LLM, золото", rng)

    # ---------- серебро ----------
    main_m = cfg["models"]["main"]
    sf = out / f"silver__{main_m}.csv"
    if sf.exists():
        skey = pd.read_parquet(out / "silver_key.parquet")
        cache = out / "silver_signals.parquet"
        if cache.exists():
            ssig = pd.read_parquet(cache)
        else:
            ssig = V.gold_signals(ecfg, skey)
            ssig.to_parquet(cache, index=False)
        sann = V.read_ann(sf)
        R.append(f"## 3. Серебро: {len(sann)} из {len(skey)} сегментов размечено моделью {main_m}\n")
        R += V.validate(ecfg, ssig, sann, f"{main_m}, серебро", rng)
        # сдвиг уровня по форме
        d = ssig.merge(sann, on="uid")
        d = d[d["skip"] == 0]
        R.append("### Сдвиг уровня по форме (серебро)\n")
        R.append("Калибровка tone ~ a + b·z(сигнал) по всем стратам; в ячейке — средний остаток LLM-тональности по страте "
                 "[95 % ДИ, бутстрэп по авторам], в баллах шкалы −2…+2. Ноль — сигнал одинаково откалиброван для всех форм; "
                 "положительный — сигнал **занижает** тональность формы относительно LLM.\n")
        rows = [{"сигнал": "средняя тональность LLM", **{V.ST_RU.get(s, s): fmt(d.loc[d["stratum"] == s, "tone"].mean())
                                                         for s in V.STRATA}}]
        for s_ in V.SIGNALS:
            x = d[[s_, "tone", "stratum", "author_id"]].dropna()
            z = (x[s_] - x[s_].mean()) / x[s_].std()
            b, a = np.polyfit(z, x["tone"], 1)
            x = x.assign(res=x["tone"] - (a + b * z))
            r = {"сигнал": s_}
            for st in V.STRATA:
                y = x[x["stratum"] == st]
                if len(y) < 20:
                    r[V.ST_RU.get(st, st)] = "—"
                    continue
                gi = {g: v.to_numpy() for g, v in y.groupby("author_id")["res"]}
                keys = list(gi)
                bs = [np.concatenate([gi[k] for k in rng.choice(keys, len(keys))]).mean() for _ in range(500)]
                r[V.ST_RU.get(st, st)] = f"{y['res'].mean():+.2f} [{np.percentile(bs, 2.5):+.2f}; {np.percentile(bs, 97.5):+.2f}]"
            rows.append(r)
        R.append(V.md_table(pd.DataFrame(rows)) + "\n")
    (ROOT / "reports" / "02_validation_llm.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
