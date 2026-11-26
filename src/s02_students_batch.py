"""Партия 1 для студентов, версия 2 (решение Ворот 5, п. 3, 2026-11-26).

Человек проверяет подъём концовки волшебной сказки контрастом ВНУТРИ текста: на каждый текст 2 фрагмента —
последний сегмент N20 (позиция 20) и один случайный сегмент из середины (позиции 6–15). Тексты — из выборки LLM-дуг:
сказки базы Афанасьева и рассказы в диапазоне длины сказок.

Критерий (до разметки): (концовка − середина) у сказок больше, чем у рассказов; 95 % ДИ разницы — бутстрэп по текстам
(внутри групп) — не включает 0. Размер партии n (текстов на группу) — наименьший из n_grid, при котором этот же критерий
на LLM-оценках выполняется в ≥ 90 % бутстрэп-симуляций (тексты — с возвращением из выборки LLM-дуг, сегмент середины —
случайный в каждой симуляции). Проверка чувствительности: к оценкам добавляется шум N(0, σ²) — человек шумнее LLM.

Разметчики A и B; 30 % текстов (оба фрагмента) размечают оба — для согласия между людьми; у каждого — скрытые повторы.
Фрагменты одного текста попадают к одному разметчику (сдвиг уровня разметчика сокращается в контрасте) и стоят
не ближе pair_min_gap позиций друг к другу.

Запуск:  .venv/bin/python src/s02_students_batch.py configs/s02_students.yaml
Выход:   data/processed/s02/students/batch1_key.parquet (+ batch1_order.parquet, batch1_power.parquet),
         reports/02_students_batch1.html (инструкция), reports/02_students_batch1_{A,B}.html, reports/02_students_power.md
         (пути — из конфига; configs/s02_students_batch1_v1.yaml воспроизводит устаревшую партию 120/1)
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402
import s02_validate as V  # noqa: E402
from s02_gold import TEMPLATE  # noqa: E402
from s04_nulls import md  # noqa: E402


def load_pool(cfg):
    acfg = load_config(ROOT / cfg["arcs_config"])
    ak = pd.read_parquet(ROOT / acfg["out_dir"] / "arcs_key.parquet")
    ann = V.read_ann(ROOT / acfg["out_dir"] / f"arcs__{acfg['models']['main']}.csv")
    ak = ak.merge(ann[["uid", "tone", "skip"]], on="uid")
    ak = ak[(ak.groupby("doc_id")["skip"].transform("sum") == 0) & ak["stratum"].isin(cfg["groups"])]
    return ak.sort_values(["doc_id", "seg"])


def power(cfg, ak, rng, n_mid=1, noise=None):
    pc = cfg["power"]
    e = cfg["end_seg"] - 1
    m0, m1 = cfg["middle_segs"][0] - 1, cfg["middle_segs"][1] - 1
    T = {g: ak[ak["stratum"] == g].pivot(index="doc_id", columns="seg", values="tone").to_numpy(float) for g in cfg["groups"]}
    ga, gb = cfg["groups"]
    rows = []
    # на всей выборке LLM-дуг (сегмент середины — среднее по 6–15, как ожидаемое значение)
    ca = T[ga][:, e] - T[ga][:, m0:m1 + 1].mean(1)
    cb = T[gb][:, e] - T[gb][:, m0:m1 + 1].mean(1)
    full = (ca.mean() - cb.mean(), len(ca), len(cb))
    for sd in ([0.0] + list(pc["noise_sd"]) if noise is None else noise):
        for n in pc["n_grid"]:
            ok, est = 0, []
            for _ in range(pc["n_sim"]):
                c = []
                for g in (ga, gb):
                    X = T[g][rng.integers(0, len(T[g]), n)]
                    # n_mid разных сегментов середины на текст
                    ms = np.argsort(rng.random((n, m1 - m0 + 1)), 1)[:, :n_mid] + m0
                    mid = np.take_along_axis(X, ms, 1)
                    end = X[:, e]
                    if sd > 0:
                        end = np.clip(np.round(end + rng.normal(0, sd, n)), -2, 2)
                        mid = np.clip(np.round(mid + rng.normal(0, sd, mid.shape)), -2, 2)
                    c.append(end - mid.mean(1))
                d = c[0].mean() - c[1].mean()
                bi = rng.integers(0, n, (pc["n_boot"], n))
                bj = rng.integers(0, n, (pc["n_boot"], n))
                bs = c[0][bi].mean(1) - c[1][bj].mean(1)
                lo = np.percentile(bs, 2.5)
                ok += lo > 0
                est.append(d)
            rows.append(dict(noise_sd=sd, n=n, n_mid=n_mid, power=ok / pc["n_sim"], mean_diff=float(np.mean(est))))
    return pd.DataFrame(rows), full


def order_items(items, rng, pair_gap, tries=200):
    """Случайный порядок фрагментов, где два фрагмента одного текста не ближе pair_gap позиций
    (жадно: на каждом шаге — случайный из допустимых; при тупике — заново)."""
    doc = {u: d for u, d in items}
    for _ in range(tries):
        left = list(rng.permutation([u for u, _ in items]))
        o, last = [], {}
        while left:
            ok = [u for u in left if doc[u] not in last or len(o) - last[doc[u]] >= pair_gap]
            if not ok:
                break
            u = ok[int(rng.integers(len(ok)))]
            last[doc[u]] = len(o)
            o.append(u)
            left.remove(u)
        if not left:
            return o
    raise RuntimeError("не удалось развести фрагменты одного текста")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    ecfg = load_config(ROOT / cfg["emotion_config"])
    rng = np.random.default_rng(cfg["seed"])
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    ak = load_pool(cfg)
    pre = cfg["key_file"].replace("_key.parquet", "")          # batch1_v2 / batch1_v1
    pf, kf, of = out / f"{pre}_power.parquet", out / cfg["key_file"], out / f"{pre}_order.parquet"
    fj = out / f"{pre}_power_full.json"
    n_mid = cfg.get("n_mid", 1)
    # 1. мощность
    if pf.exists():
        pw = pd.read_parquet(pf)
        full = tuple(json.loads(fj.read_text()))
    else:
        pw, full = power(cfg, ak, rng)
        alt, _ = power(cfg, ak, rng, n_mid=cfg["power"]["alt_n_mid"], noise=[0.0, 1.0])
        pw = pd.concat([pw, alt], ignore_index=True)
        pw.to_parquet(pf, index=False)
        fj.write_text(json.dumps(list(map(float, full))))
    base = pw[(pw["noise_sd"] == 0) & (pw["n_mid"] == n_mid)]
    good = base[base["power"] >= cfg["power"]["target"]]
    if cfg.get("n_fixed"):                                   # n задан решением пользователя
        n = int(cfg["n_fixed"])
        reached = bool((base.loc[base["n"] == n, "power"] >= cfg["power"]["target"]).all())
    else:
        n = int(good["n"].min()) if len(good) else int(base["n"].max())
        reached = bool(len(good))
    pw_n = float(base.loc[base["n"] == n, "power"].iloc[0])
    # 2. ключ
    e = cfg["end_seg"] - 1
    m0, m1 = cfg["middle_segs"][0] - 1, cfg["middle_segs"][1] - 1
    if kf.exists():
        key = pd.read_parquet(kf)
        order = pd.read_parquet(of)
        print("ключ уже есть — страницы пересобираются по нему:", kf)
    else:
        rb = np.random.default_rng(cfg["seed"] + 7)
        rows, assign = [], {}
        for g in cfg["groups"]:
            docs = np.array(sorted(ak.loc[ak["stratum"] == g, "doc_id"].unique()))
            pick = rb.choice(docs, n, replace=False)
            n_ov = int(round(cfg["overlap_frac"] * n))
            for i, d in enumerate(pick):
                assign[d] = "AB" if i < n_ov else cfg["annotators"][(i - n_ov) % 2]
                t = ak[ak["doc_id"] == d].set_index("seg")
                ms = sorted(int(v) for v in rb.choice(np.arange(m0, m1 + 1), n_mid, replace=False))
                for zone, s in [("конец", e)] + [("середина", v) for v in ms]:
                    r = t.loc[s]
                    rows.append(dict(doc_id=d, stratum=g, author_id=r["author_id"], doc_n_words=int(r["doc_n_words"]), zone=zone,
                                     seg=s + 1, n_words=int(r["shown_n_words"]), llm_tone=float(r["tone"]), text=r["text"],
                                     annotators=assign[d]))
        key = pd.DataFrame(rows).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
        key.insert(0, "uid", [f"c{i + 1:03d}" for i in range(len(key))])
        key["repeat_of"] = None
        orows, reps = [], []
        for a in cfg["annotators"]:
            mine = key[key["annotators"].str.contains(a)]
            o = order_items(list(zip(mine["uid"], mine["doc_id"])), rb, cfg["pair_min_gap"])
            src = np.sort(rb.choice(60, cfg["n_repeats"], replace=False))
            o2 = list(o)
            for j, p in enumerate(src[::-1]):
                ru = f"r{a}{cfg['n_repeats'] - j:02d}"
                r = key[key["uid"] == o[p]].iloc[0].copy()
                r["repeat_of"], r["uid"], r["annotators"] = r["uid"], ru, a
                reps.append(r)
                lo = o2.index(o[p]) + cfg["repeat_min_gap"]
                o2.insert(int(rb.integers(lo, len(o2) + 1)), ru)
            ix = {u: i for i, u in enumerate(o2)}
            assert all(ix[r.uid] - ix[r.repeat_of] >= cfg["repeat_min_gap"] for r in reps if r.annotators == a)
            orows += [dict(annotator=a, pos=i, uid=u) for i, u in enumerate(o2)]
        key = pd.concat([key, pd.DataFrame(reps)], ignore_index=True)
        order = pd.DataFrame(orows)
        key.to_parquet(kf, index=False)
        order.to_parquet(of, index=False)
    # 3. страницы
    tm = cfg["time_model"]
    texts = dict(zip(key["uid"], key["text"]))
    words = dict(zip(key["uid"], key["n_words"]))
    per = {}
    for a in cfg["annotators"]:
        o = list(order.loc[order["annotator"] == a].sort_values("pos")["uid"])
        sk = cfg["storage_key"].format(a=a)
        items = {u: texts[u] for u in o}
        page = (TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
                .replace("__FRESH__", json.dumps(o)).replace("__APPEND__", json.dumps(o))
                .replace("__EMOTIONS__", json.dumps(ecfg["gold"]["emotions"], ensure_ascii=False))
                .replace('const KEY = "gold_annotation_v1";', f'const KEY = "{sk}";')
                .replace('"gold_annotation_" + new Date()', f'"students_batch1_{a}_" + new Date()')
                .replace("<title>Золотая разметка — тональность и эмоции</title>", f"<title>Разметка, партия 1 — разметчик {a}</title>")
                .replace("<b>Разметка тональности и эмоций</b>", f"<b>Партия 1 — разметчик {a}</b>")
                .replace("Форма и источник текста скрыты намеренно.", "Источник текста скрыт намеренно."))
        chrome = page.replace(json.dumps(items, ensure_ascii=False).replace("</", "<\\/"), "")   # без текстов фрагментов
        assert "сказк" not in chrome.lower() and "Форма и источник" not in chrome
        assert sk in page and f"students_batch1_{a}_" in page
        (ROOT / cfg["page_annotator"].format(a=a)).write_text(page, encoding="utf-8")
        w = sum(words[u] for u in o)
        per[a] = dict(items=len(o), words=w, minutes=w / tm["words_per_min"] + len(o) * tm["sec_per_item"] / 60)
    o = key[key["repeat_of"].isna()]
    emos = ", ".join(ecfg["gold"]["emotions"])
    tmax = max(v["minutes"] for v in per.values())
    assert "сказк" not in emos.lower()
    links = "".join(f'<li><a href="{Path(cfg["page_annotator"].format(a=a)).name}">Разметчик {a}</a> — {v["items"]} фрагментов, '
                    f'≈ {v["words"]} слов, ≈ {v["minutes"]:.0f} мин</li>' for a, v in per.items())
    idx = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка, партия 1 — инструкция</title>
<style>:root{{--bg:#fbfaf7;--ink:#1d1c1a;--ink2:#5b5953;--line:#e3e0d8;--acc:#2a5fb0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171716;--ink:#ecebe6;--ink2:#aeaca4;--line:#34332f;--acc:#7aa7ec}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:760px;margin:0 auto;padding:24px 20px}}h1{{font-size:22px}}a{{color:var(--acc)}}li{{margin:4px 0}}
.note{{color:var(--ink2);font-size:14px;border-top:1px solid var(--line);padding-top:12px;margin-top:24px}}</style></head><body><main>
<h1>Разметка тональности, партия 1 — инструкция</h1>
<p>Вы оцениваете короткие фрагменты художественных текстов (обычно 40–120 слов). Каждый фрагмент оценивайте сам по себе, целиком.</p>
<ol>
<li><b>Тональность фрагмента в целом</b>: −2 — очень негативная, −1 — скорее негативная, 0 — нейтральная или без перевеса,
+1 — скорее позитивная, +2 — очень позитивная. Оценивайте то, как звучит и что описывает сам фрагмент, а не то, чем,
по-вашему, кончится история.</li>
<li><b>Доминирующая эмоция</b> — ровно одна из списка: {html.escape(emos)}.</li>
<li><b>Пропустить</b> — только если фрагмент невозможно понять.</li>
</ol>
<p>Откуда фрагмент и из какого места текста — скрыто намеренно; не пытайтесь это угадать и не ищите тексты в интернете.
Не обсуждайте фрагменты с другим разметчиком до сдачи результатов.</p>
<p>Ответы сохраняются в браузере автоматически, можно прерываться. Работайте в {int(np.ceil(tmax / 45))}–{int(np.ceil(tmax / 35))} приёмов по 35–45 минут, с перерывами: усталость портит оценки.
В конце нажмите «Выгрузить CSV» и пришлите файл.</p>
<h2 style="font-size:18px">Ваша страница</h2><ul>{links}</ul>
<p><b>Нагрузка</b>: {max(v["items"] for v in per.values())} фрагментов на разметчика. <b>Время</b>: около {tmax:.0f} минут (≈ {tmax / 60:.1f} ч) на разметчика (из расчёта {tm['words_per_min']} слов в минуту и {tm['sec_per_item']} с на выбор ответа).</p>
<p class="note">Откройте только свою страницу. Если открываете в другом браузере или на другом устройстве — начатая разметка туда не переносится.</p>
</main></body></html>"""
    assert "сказк" not in idx.lower() and "форм" not in idx.lower(), "в инструкции не должно быть жанра и форм"
    (ROOT / cfg["page_index"]).write_text(idx, encoding="utf-8")
    # 4. отчёт о мощности и составе
    pv = pw.pivot(index="n", columns=["n_mid", "noise_sd"], values="power")
    pv.columns = [("LLM-оценки (критерий выбора n)" if sd == 0 else f"+ шум σ = {sd}") if nm == 1 else
                  f"вариант: {nm} середины, " + ("LLM" if sd == 0 else f"шум σ = {sd}") for nm, sd in pv.columns]
    pv = pv.reset_index().rename(columns={"n": "текстов на группу"})
    pv = pv.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v)
    k = cfg["fragments_per_text"] = 1 + n_mid
    nt = o["doc_id"].nunique()
    L = [f"# Партия 1 для студентов ({pre}): мощность и состав\n",
         f"{cfg.get('decision', '')}\n",
         "Критерий (зафиксирован до разметки): контраст внутри текста — (последний сегмент − среднее сегментов середины "
         f"{cfg['middle_segs'][0]}–{cfg['middle_segs'][1]}) у сказок больше, чем у рассказов той же длины; 95 % ДИ разницы — "
         "бутстрэп по текстам — не включает 0 (знак +, как у LLM).\n",
         f"На всей выборке LLM-дуг ({full[1]:.0f} сказок, {full[2]:.0f} рассказов) разница контрастов в сырых баллах: "
         f"{full[0]:+.2f}.\n",
         f"### Доля симуляций, в которых критерий выполняется ({cfg['power']['n_sim']} симуляций, бутстрэп {cfg['power']['n_boot']})\n",
         md(pv) + "\n",
         f"Сказок базы в выборке LLM-дуг — {full[1]:.0f}, это верхняя граница n.\n",
         f"**Партия: n = {n} текстов на группу, {n_mid} фрагмент(а) середины на текст; мощность на LLM-оценках {pw_n:.3f}**"
         + ("" if reached else " — ЦЕЛЬ 90 % НЕ ДОСТИГНУТА") + ". Колонки с шумом — проверка чувствительности: человек "
         "шумнее LLM.\n",
         "### Состав\n",
         md(o.groupby(["stratum", "zone"]).agg(фрагментов=("uid", "size"), слов_медиана=("n_words", "median"),
                                               LLM_среднее=("llm_tone", "mean")).round(2).reset_index()) + "\n",
         md(pd.DataFrame([dict(разметчик=a, **{"фрагментов (с повторами)": v["items"], "слов": v["words"],
                                               "минут (оценка)": round(v["minutes"])}) for a, v in per.items()])) + "\n",
         f"Текстов, которые размечают оба: {o.loc[o['annotators'] == 'AB', 'doc_id'].nunique()} из {nt} "
         f"({cfg['overlap_frac']:.0%}); скрытых повторов — {cfg['n_repeats']} на разметчика (разрыв ≥ {cfg['repeat_min_gap']}); "
         f"все {k} фрагмента одного текста — у одного разметчика, не ближе {cfg['pair_min_gap']} позиций друг к другу.\n",
         f"Страницы: `{cfg['page_index']}` (инструкция), `{cfg['page_annotator'].format(a='A')}`, "
         f"`{cfg['page_annotator'].format(a='B')}`.\n"]
    (ROOT / cfg["power_report"]).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
