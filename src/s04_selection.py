"""Ворота 4, п. 3: (а) доля текстов, прошедших тест связности, по формам с контролем длины;
(в) веса по шуму — модель log SNR только от размера сегмента (без формы), обученная на бутстрэп-выборке этапа 4.

(а) GLM (биномиальная) прошёл ~ форма + B-сплайн(log n_words), кластерные ошибки по авторам; стандартизованные доли:
    предсказание для каждого текста при «чужой» форме, усреднённое по распределению длины всех текстов.
(в) log SNR ~ B-сплайн(log слов в сегменте) на 4 509 текстах (сигнал отдельно); прогноз для всех дуг N20;
    вес = прогноз SNR / (1 + прогноз SNR) (доля дисперсии дуги, не объяснённая шумом); сохраняется в weights_N20.parquet.

Запуск:  .venv/bin/python src/s04_selection.py configs/s04_nulls.yaml
Выход:   reports/04_selection_weights.md, data/processed/s04/full/weights_N20.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
from s03_report import FORM_RU  # noqa: E402
from s04_nulls import ORDER, md  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    sc = cfg["selection"]
    out = ROOT / cfg["out_dir"] / "full"
    R = ["# Отбор по тесту связности и веса по шуму (Ворота 4, п. 3)\n"]
    wts = None
    for N in cfg["N"]:
        d = pd.read_parquet(out / f"arcs_N{N}.parquet").drop(columns=cfg["signals"]).merge(pd.read_parquet(out / f"metrics_N{N}.parquet"), on="doc_id")
        d = d[d["within_author_cap"]].copy()
        d["lw"] = np.log(d["n_words"])
        d["author_id"] = d["author_id"].astype(str)
        forms = [f for f in ORDER if (d["fk"] == f).any()]
        R.append(f"## (а) N = {N}: доля прошедших тест связности (lag1, p < 0,05), {len(d)} текстов\n")
        R.append("Сырые доли и стандартизованные по длине (GLM прошёл ~ форма + сплайн log n_words, усреднение по распределению "
                 "длины всех текстов); в скобках — 95 % ДИ разницы с рассказом (кластерные ошибки по авторам, дельта-метод не "
                 "применяется — ДИ коэффициента в логитах).\n")
        rows = []
        for s_ in cfg["signals"]:
            d["passed"] = (d[f"{s_}_lag1_p"] < 0.05).astype(int)
            m = smf.glm(f"passed ~ C(fk, Treatment('story')) + bs(lw, df={sc['spline_df']})", d, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d["author_id"])[0]})
            r = {"сигнал": s_}
            for f in forms:
                raw = d.loc[d["fk"] == f, "passed"].mean()
                std = m.predict(d.assign(fk=f)).mean()
                nm = f"C(fk, Treatment('story'))[T.{f}]"
                ci = "" if f == "story" else " [логит {:+.2f}; {:+.2f}]".format(*m.conf_int().loc[nm])
                r[FORM_RU[f]] = f"{raw:.3f} → {std:.3f}{ci}"
            rows.append(r)
        R.append(md(pd.DataFrame(rows)) + "\n")

    # (в) веса
    nz = pd.read_parquet(out / "noise_N20.parquet")
    nz["lsw"] = np.log(nz["seg_words_median"].clip(lower=1))
    arcs = pd.read_parquet(out / "arcs_N20.parquet", columns=["doc_id", "fk", "seg_words_median", "within_author_cap"])
    arcs["lsw"] = np.log(arcs["seg_words_median"].clip(lower=1))
    R.append(f"## (в) Веса по шуму: модель log SNR ~ сплайн(log слов в сегменте), без формы; обучение — {len(nz)} текстов\n")
    rows = []
    for s_ in cfg["signals"]:
        x = nz[np.isfinite(np.log(nz[f"{s_}_snr"]))].assign(y=lambda t: np.log(t[f"{s_}_snr"]))
        m = smf.ols(f"y ~ bs(lsw, df={sc['spline_df']}, lower_bound={arcs['lsw'].min()}, upper_bound={arcs['lsw'].max()})", x).fit()
        # остаток модели по форме: проверка, что форма не входит в вес напрямую, но и не теряется
        x["res"] = m.resid
        snr = np.exp(m.predict(arcs))
        arcs[f"{s_}_snr_pred"] = snr
        arcs[f"{s_}_w"] = snr / (1 + snr)
        r = {"сигнал": s_, "R² модели": f"{m.rsquared:.3f}"}
        for f in ORDER:
            if (arcs["fk"] == f).any():
                r[FORM_RU[f]] = f"w {arcs.loc[arcs['fk'] == f, f'{s_}_w'].median():.2f}; ост. {x.loc[x['fk'] == f, 'res'].mean():+.2f}"
        rows.append(r)
    R.append("В ячейке: медиана веса по форме; средний остаток log SNR у формы в обучающей выборке (форма, не учтённая "
             "моделью, — ненулевой остаток значит, что вес по размеру сегмента не полностью описывает надёжность формы).\n")
    R.append(md(pd.DataFrame(rows)) + "\n")
    arcs.drop(columns=["lsw"]).to_parquet(out / "weights_N20.parquet", index=False)
    (ROOT / "reports" / "04_selection_weights.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))


if __name__ == "__main__":
    main()
