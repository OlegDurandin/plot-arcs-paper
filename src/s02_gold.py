"""Этап 2. Золотая выборка (400 сегментов N = 20) и интерфейс слепой разметки.

Запуск:
    .venv/bin/python src/s02_gold.py configs/s02_emotion.yaml
Выход:
    data/processed/s02/gold_key.parquet   — ключ: uid -> doc_id, корпус, форма, сегмент, показанный диапазон предложений
    reports/02_gold_annotation.html       — локальный файл без сервера: только uid и текст (форма/корпус скрыты)
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402


def pick_docs(pool: pd.DataFrame, n: int, rng, one_per_author: bool) -> pd.DataFrame:
    pool = pool.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    if one_per_author:
        pool = pool.drop_duplicates("author_id")
    return pool.head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    gc = cfg["gold"]
    rng = np.random.default_rng(cfg["seed"])
    s01 = ROOT / cfg["s01_run"]
    a = pd.read_parquet(ROOT / cfg["analysis_set"])
    a = a[a["include"] & a["unit"].isin(gc["units"]) & a["within_author_cap"]]
    seg = pd.read_parquet(s01 / "segments_rel_N20.parquet")
    have = set(seg["doc_id"])
    a = a[a["doc_id"].isin(have)]

    out = ROOT / cfg["out_dir"]
    kf = out / "gold_key.parquet"
    if kf.exists():
        # выборка уже показана разметчику — не пересоздаём, только (до)строим повторы и страницу
        key = pd.read_parquet(kf)
        key = key[~key["uid"].isin([f"s{i:03d}" for i in range(401, 1000)])] if "repeat_of" not in key else key[key["repeat_of"].isna()]
        build_page(cfg, key, rng)
        return
    n_af = gc["afanasyev_min"]
    rest = gc["n_total"] - n_af
    per = rest // len(gc["forms"])
    extra = rest - per * len(gc["forms"])
    picks = [pick_docs(a[a["base_magic600"]], n_af, rng, False).assign(stratum="afanasyev")]
    for i, f in enumerate(gc["forms"]):
        k = per + (1 if i < extra else 0)
        pool = a[(a["form"] == f) & (a["corpus"] != "afanasyev")]
        picks.append(pick_docs(pool, k, rng, gc["one_text_per_author"]).assign(stratum=f))
    docs = pd.concat(picks, ignore_index=True)

    # по одному случайному сегменту на текст (≤ max_per_text)
    segd = dict(tuple(seg[seg["doc_id"].isin(set(docs["doc_id"]))].groupby("doc_id")))
    rows = []
    for r in docs.itertuples():
        sg = segd[r.doc_id].sort_values("seg")
        s = sg.iloc[int(rng.integers(len(sg)))]
        rows.append(dict(doc_id=r.doc_id, corpus=r.corpus, form=r.form, unit=r.unit, stratum=r.stratum,
                         author_id=r.author_id, seg=int(s["seg"]), seg_sent_start=int(s["sent_start"]),
                         seg_sent_end=int(s["sent_end"]), seg_n_words=int(s["n_words"])))
    key = pd.DataFrame(rows)

    # тексты сегментов
    want = set(key["doc_id"])
    sents = []
    for f in sorted(glob.glob(str(s01 / "sentences" / "*.parquet"))):
        x = pd.read_parquet(f, columns=["doc_id", "sent_idx", "text", "n_words"], filters=[("doc_id", "in", list(want))])
        if len(x):
            sents.append(x)
    sents = pd.concat(sents)
    sd = {k: g.set_index("sent_idx").sort_index() for k, g in sents.groupby("doc_id")}
    shown_a, shown_z, texts, excerpt = [], [], [], []
    lim, ew = gc["excerpt_if_longer"], gc["excerpt_words"]
    for r in key.itertuples():
        g = sd[r.doc_id].loc[r.seg_sent_start:r.seg_sent_end - 1]
        a_, z_ = r.seg_sent_start, r.seg_sent_end
        if lim and r.seg_n_words > lim:
            # связный отрывок ≈ ew слов из случайного места сегмента, целыми предложениями
            cum = g["n_words"].cumsum().to_numpy()
            total = cum[-1]
            start_w = int(rng.integers(0, max(1, total - ew)))
            i0 = int(np.searchsorted(cum, start_w, side="right"))
            i1 = int(np.searchsorted(cum, cum[i0] - g["n_words"].iloc[i0] + ew, side="left")) + 1
            i1 = min(max(i1, i0 + 1), len(g))
            a_, z_ = int(g.index[i0]), int(g.index[i1 - 1]) + 1
            g = g.loc[a_:z_ - 1]
        shown_a.append(a_)
        shown_z.append(z_)
        excerpt.append((a_, z_) != (r.seg_sent_start, r.seg_sent_end))
        texts.append(" ".join(g["text"]))
    key["shown_sent_start"], key["shown_sent_end"], key["is_excerpt"] = shown_a, shown_z, excerpt
    key["shown_n_words"] = [int(sd[d].loc[a_:z_ - 1, "n_words"].sum()) for d, a_, z_ in zip(key["doc_id"], shown_a, shown_z)]
    key["text"] = texts
    # случайный порядок и слепые идентификаторы
    key = key.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    key.insert(0, "uid", [f"s{i + 1:03d}" for i in range(len(key))])
    out.mkdir(parents=True, exist_ok=True)
    key.to_parquet(kf, index=False)
    build_page(cfg, key, rng)
    summ = key.groupby("stratum").agg(сегментов=("uid", "size"), отрывков=("is_excerpt", "sum"),
                                      слов_показано=("shown_n_words", "sum"), медиана_слов=("shown_n_words", "median"))
    print(summ.to_string())
    print("всего слов к прочтению:", int(key["shown_n_words"].sum()), "| авторов:", key["author_id"].nunique(),
          "| текстов:", key["doc_id"].nunique())


def build_page(cfg: dict, key: pd.DataFrame, rng):
    """Страница разметки: 400 сегментов + повторы для проверки согласованности разметчика с собой.

    Повторы: n_repeats сегментов из первых repeat_source_max позиций; при чистом старте каждый вставляется не раньше,
    чем через repeat_min_gap позиций после первого показа; если разметка уже начата (в localStorage есть ответы) —
    повторы идут в конец очереди (оригиналы из первых 250 позиций => разрыв ≥ 150 сохраняется). uid на экран не выводится.
    """
    gc = cfg["gold"]
    out = ROOT / cfg["out_dir"]
    key = key.sort_values("uid").reset_index(drop=True)
    base = list(key["uid"])                       # порядок показа = порядок uid (выборка уже перемешана)
    rr = np.random.default_rng(cfg["seed"] + 1)
    src_pos = np.sort(rr.choice(min(gc["repeat_source_max"], len(base)), gc["n_repeats"], replace=False))
    reps = []
    for j, p in enumerate(src_pos):
        reps.append(dict(uid=f"s{len(base) + j + 1:03d}", repeat_of=base[p], src_pos=int(p)))
    # чистый старт: вставки на случайные позиции в [p + gap, конец]
    fresh = list(base)
    ins = []
    for r in reps:
        lo = r["src_pos"] + gc["repeat_min_gap"]
        ins.append((int(rr.integers(lo, len(base) + 1)), r["uid"]))
    for posn, u in sorted(ins, key=lambda x: (-x[0], x[1])):
        fresh.insert(posn, u)
    # проверка разрыва в обоих порядках
    for order in (fresh, base + [r["uid"] for r in reps]):
        ix = {u: i for i, u in enumerate(order)}
        assert all(ix[r["uid"]] - ix[r["repeat_of"]] >= gc["repeat_min_gap"] for r in reps), "разрыв < gap"
    rep = pd.DataFrame(reps)
    src = key.set_index("uid")
    rep_rows = src.loc[rep["repeat_of"]].reset_index(drop=True).copy()
    rep_rows["uid"] = rep["uid"].to_numpy()
    rep_rows["repeat_of"] = rep["repeat_of"].to_numpy()
    full = pd.concat([key.assign(repeat_of=None), rep_rows], ignore_index=True)
    full.to_parquet(out / "gold_key.parquet", index=False)
    texts = dict(zip(full["uid"], full["text"]))
    items = {u: texts[u] for u in full["uid"]}
    page = (TEMPLATE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__FRESH__", json.dumps(fresh)).replace("__APPEND__", json.dumps(base + [r["uid"] for r in reps]))
            .replace("__EMOTIONS__", json.dumps(gc["emotions"], ensure_ascii=False)))
    (ROOT / "reports" / "02_gold_annotation.html").write_text(page, encoding="utf-8")
    print(f"страница: {len(base)} сегментов + {len(reps)} повторов; разрыв ≥ {gc['repeat_min_gap']} в обоих порядках")


TEMPLATE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Золотая разметка — тональность и эмоции</title>
<style>
:root{--bg:#fbfaf7;--card:#fff;--ink:#1d1c1a;--ink2:#5b5953;--line:#e3e0d8;--acc:#2a5fb0;--acc2:#e8eef8;--warn:#a24a12}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--card:#1f1f1d;--ink:#ecebe6;--ink2:#aeaca4;--line:#34332f;--acc:#7aa7ec;--acc2:#24324a;--warn:#e19a62}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;z-index:2}
header b{font-size:15px}.bar{flex:1;min-width:140px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
main{max-width:860px;margin:0 auto;padding:20px}
.text{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px 24px;font:18px/1.65 Georgia,"Times New Roman",serif;white-space:pre-wrap;max-height:55vh;overflow:auto}
.row{margin:18px 0 6px;color:var(--ink2);font-size:14px}
.opts{display:flex;gap:8px;flex-wrap:wrap}
.opts button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:8px 14px;font-size:15px;cursor:pointer;min-width:48px}
.opts button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.opts button kbd{font:11px monospace;color:var(--ink2);margin-left:6px}.opts button.on kbd{color:#dfe8f7}
.nav{display:flex;gap:10px;margin-top:22px;align-items:center;flex-wrap:wrap}
.nav button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:9px 16px;cursor:pointer;font-size:15px}
.nav .primary{background:var(--acc);border-color:var(--acc);color:#fff}
textarea{width:100%;min-height:44px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);padding:8px;font:14px/1.4 inherit}
.hint{color:var(--ink2);font-size:13px}.skip.on{background:var(--warn)!important;border-color:var(--warn)!important}
.status{font-size:13px;color:var(--ink2)}
</style></head><body>
<header><b>Разметка тональности и эмоций</b><div class="bar"><i id="bar"></i></div><span class="status" id="prog"></span>
<button onclick="exportCSV()">Выгрузить CSV</button></header>
<main>
<div class="row" id="pos"></div>
<div class="text" id="text"></div>
<div class="row">Тональность фрагмента в целом</div>
<div class="opts" id="tone"></div>
<div class="row">Доминирующая эмоция</div>
<div class="opts" id="emo"></div>
<div class="row">Не получается оценить</div>
<div class="opts"><button class="skip" id="skip" onclick="toggleSkip()">пропустить / непонятно <kbd>S</kbd></button></div>
<div class="row">Комментарий (необязательно)</div>
<textarea id="note" oninput="save()"></textarea>
<div class="nav"><button onclick="go(-1)">← Назад</button><button class="primary" onclick="go(1)">Дальше →</button>
<button onclick="goFirstEmpty()">К первому неразмеченному</button><span class="hint">Клавиши: 1–5 = −2…+2, Q–U = эмоции, S = пропуск, ← →</span></div>
<p class="hint">Ответы сохраняются в этом браузере автоматически (localStorage). Чтобы передать результат — «Выгрузить CSV».
Форма и источник текста скрыты намеренно.</p>
</main>
<script>
const TEXTS = __ITEMS__;
const FRESH = __FRESH__, APPEND = __APPEND__;
const EMO = __EMOTIONS__;
const TONES = [-2,-1,0,1,2];
const KEY = "gold_annotation_v1";
let st; try { st = JSON.parse(localStorage.getItem(KEY)) || {}; } catch(e) { st = {}; }
st.ans = st.ans || {}; st.i = st.i || 0;
// порядок очереди: при начатой разметке повторы — в конец (старые индексы позиций сохраняются)
if (!st.order || st.order.length !== FRESH.length) { st.order = Object.keys(st.ans).length ? APPEND : FRESH; }
const ITEMS = st.order.map(u => ({uid: u, text: TEXTS[u]}));
try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {}
const keysT = ["1","2","3","4","5"], keysE = ["q","w","e","r","t","y","u"];
function save(){ const u = ITEMS[st.i].uid; const a = st.ans[u] || (st.ans[u] = {});
  a.note = document.getElementById("note").value; a.ts = new Date().toISOString();
  try { localStorage.setItem(KEY, JSON.stringify(st)); } catch(e) {} }
function done(a){ return a && (a.skip || (a.tone !== undefined && a.emo)); }
function render(){
  const it = ITEMS[st.i], a = st.ans[it.uid] || {};
  document.getElementById("text").textContent = it.text;
  document.getElementById("text").scrollTop = 0;
  document.getElementById("pos").textContent = `Фрагмент ${st.i+1} из ${ITEMS.length}`;
  const tone = document.getElementById("tone"); tone.innerHTML = "";
  TONES.forEach((v,k) => { const b = document.createElement("button"); b.innerHTML = (v>0?"+":"")+v+`<kbd>${keysT[k]}</kbd>`;
    if (a.tone === v) b.className = "on"; b.onclick = () => setTone(v); tone.appendChild(b); });
  const emo = document.getElementById("emo"); emo.innerHTML = "";
  EMO.forEach((v,k) => { const b = document.createElement("button"); b.innerHTML = v+`<kbd>${keysE[k].toUpperCase()}</kbd>`;
    if (a.emo === v) b.className = "on"; b.onclick = () => setEmo(v); emo.appendChild(b); });
  document.getElementById("skip").className = "skip" + (a.skip ? " on" : "");
  document.getElementById("note").value = a.note || "";
  const n = ITEMS.filter(x => done(st.ans[x.uid])).length;
  document.getElementById("prog").textContent = `готово ${n} / ${ITEMS.length}`;
  document.getElementById("bar").style.width = (100*n/ITEMS.length) + "%";
}
function cur(){ const u = ITEMS[st.i].uid; return st.ans[u] || (st.ans[u] = {}); }
function setTone(v){ const a = cur(); a.tone = v; a.skip = false; save(); render(); }
function setEmo(v){ const a = cur(); a.emo = v; a.skip = false; save(); render(); }
function toggleSkip(){ const a = cur(); a.skip = !a.skip; save(); render(); }
function go(d){ save(); st.i = Math.max(0, Math.min(ITEMS.length-1, st.i+d)); save(); render(); }
function goFirstEmpty(){ const k = ITEMS.findIndex(x => !done(st.ans[x.uid])); if (k >= 0) { st.i = k; save(); render(); } }
document.addEventListener("keydown", e => { if (e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase(); const ti = keysT.indexOf(k), ei = keysE.indexOf(k);
  if (ti >= 0) setTone(TONES[ti]); else if (ei >= 0) setEmo(EMO[ei]); else if (k === "s") toggleSkip();
  else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1); });
function exportCSV(){
  save();
  const esc = s => '"' + String(s ?? "").replace(/"/g,'""') + '"';
  const lines = ["uid,tone,emotion,skip,note,timestamp"];
  ITEMS.forEach(it => { const a = st.ans[it.uid] || {};
    lines.push([it.uid, a.tone ?? "", esc(a.emo ?? ""), a.skip ? 1 : 0, esc(a.note ?? ""), a.ts ?? ""].join(",")); });
  const blob = new Blob(["﻿" + lines.join("\n")], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob); const l = document.createElement("a");
  l.href = url; l.download = "gold_annotation_" + new Date().toISOString().slice(0,10) + ".csv";
  document.body.appendChild(l); l.click(); l.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
