"""Этап 3, Ворота 3 п. 2: эмбеддинги сегментов N = 20 (multilingual-e5-small) на всём корпусе — без усечения.

Кусок — подряд идущие предложения сегмента, ≤ max_tokens (с префиксом и спецтокенами); предложение длиннее —
окнами по токенам (слова предложения делятся между окнами пропорционально токенам). Эмбеддинг куска — среднее
по токенам (mean pooling) с L2-нормировкой; эмбеддинг сегмента — среднее кусков с весами по словам, L2-нормировка.

Запуск:
    .venv/bin/python src/s03_embed.py configs/s03_robust.yaml --scope sample   # 2 % текстов: оценка времени
    .venv/bin/python src/s03_embed.py configs/s03_robust.yaml --scope full     # весь корпус (продолжает с места)
Выход: data/processed/s03/robust/emb_{sample,full}/shard-*.parquet (doc_id, seg, n_words, n_chunks, emb float16)
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, in_sample, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--scope", choices=["sample", "full"], required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = load_config(ROOT / cfg["topics_config"])
    ec, cc = cfg["embeddings"], cfg["embeddings_corpus"]
    import torch
    from transformers import AutoModel, AutoTokenizer
    od = ROOT / cfg["out_dir"] / f"emb_{args.scope}"
    od.mkdir(parents=True, exist_ok=True)
    s01 = ROOT / tcfg["s01_run"]
    seg = pd.read_parquet(s01 / f"segments_rel_N{cfg['N']}.parquet")
    if args.scope == "sample":
        keep = {d for d in seg["doc_id"].unique() if in_sample(d, cc["sample_frac"])}
        seg = seg[seg["doc_id"].isin(keep)]
    segd = {d: g.sort_values("seg") for d, g in seg.groupby("doc_id")}
    dev = ec["device"] if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ec["model"])
    model = AutoModel.from_pretrained(ec["model"]).to(dev).eval()
    pre = tok(ec["prefix"], add_special_tokens=False)["input_ids"]
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    room = ec["max_tokens"] - 2 - len(pre)
    files = sorted(glob.glob(str(s01 / "sentences" / "*.parquet")))
    t0, words_done, segs_done, chunks_done = time.time(), 0, 0, 0
    total_words = int(seg["n_words"].sum())
    for fi, f in enumerate(files):
        of = od / Path(f).name
        if of.exists():
            continue
        s = pd.read_parquet(f, columns=["doc_id", "sent_idx", "text", "n_words"])
        s = s[s["doc_id"].isin(segd.keys())]
        if not len(s):
            pd.DataFrame(columns=["doc_id", "seg", "n_words", "n_chunks", "emb"]).to_parquet(of, index=False)
            continue
        ids_all = tok(list(s["text"]), add_special_tokens=False)["input_ids"]
        s = s.assign(ids=ids_all)
        chunks, owner, weight = [], [], []
        for d, g in s.groupby("doc_id", sort=False):
            g = g.set_index("sent_idx").sort_index()
            for r in segd[d].itertuples():
                x = g.loc[r.sent_start:r.sent_end - 1]
                buf, bw = [], 0.0

                def flush():
                    nonlocal buf, bw
                    if buf:
                        chunks.append(buf); owner.append((d, int(r.seg))); weight.append(max(bw, 1e-6))
                    buf, bw = [], 0.0
                for ids, w_ in zip(x["ids"], x["n_words"]):
                    if len(ids) > room:            # длинное предложение — окнами
                        flush()
                        for k in range(0, len(ids), room):
                            part = ids[k:k + room]
                            chunks.append(part); owner.append((d, int(r.seg))); weight.append(max(w_ * len(part) / len(ids), 1e-6))
                        continue
                    if len(buf) + len(ids) > room:
                        flush()
                    buf = buf + list(ids); bw += w_
                flush()
        order = np.argsort([len(c) for c in chunks])
        E = np.zeros((len(chunks), model.config.hidden_size), np.float32)
        with torch.no_grad():
            bs = ec["batch_size"]
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                seqs = [[cls] + pre + chunks[j] + [sep] for j in idx]
                L = max(len(q) for q in seqs)
                inp = torch.full((len(seqs), L), pad, dtype=torch.long)
                att = torch.zeros((len(seqs), L), dtype=torch.long)
                for k, q in enumerate(seqs):
                    inp[k, :len(q)] = torch.tensor(q); att[k, :len(q)] = 1
                inp, att = inp.to(dev), att.to(dev)
                h = model(input_ids=inp, attention_mask=att).last_hidden_state
                m_ = att.unsqueeze(-1).float()
                v = torch.nn.functional.normalize((h * m_).sum(1) / m_.sum(1), dim=-1)
                E[idx] = v.cpu().numpy()
        df = pd.DataFrame({"key": owner, "w": weight})
        df["i"] = np.arange(len(df))
        rows = []
        for (d, sg), g in df.groupby("key", sort=False):
            v = (E[g["i"].to_numpy()] * g["w"].to_numpy()[:, None]).sum(0)
            rows.append((d, sg, len(g), (v / np.linalg.norm(v)).astype(cc["store_dtype"]).tolist()))
        out = pd.DataFrame(rows, columns=["doc_id", "seg", "n_chunks", "emb"])
        nw = seg.set_index(["doc_id", "seg"])["n_words"]
        out["n_words"] = nw.reindex(list(zip(out["doc_id"], out["seg"]))).to_numpy()
        out.to_parquet(of, index=False)
        words_done += int(out["n_words"].sum()); segs_done += len(out); chunks_done += len(chunks)
        el = time.time() - t0
        if fi % 10 == 0 or args.scope == "sample":
            print(f"[{el:7.0f}s] шард {fi + 1}/{len(files)}: сегментов {segs_done}, кусков {chunks_done}, "
                  f"слов {words_done / 1e6:.1f} млн из {total_words / 1e6:.1f} млн", flush=True)
    el = time.time() - t0
    print(f"готово: {el:.0f} с; сегментов {segs_done}, кусков {chunks_done}, слов {words_done}")
    if args.scope == "sample" and words_done:
        full_words = int(pd.read_parquet(s01 / f"segments_rel_N{cfg['N']}.parquet", columns=["n_words"])["n_words"].sum())
        est = el / words_done * full_words / 3600
        print(f"ОЦЕНКА полного корпуса ({full_words / 1e6:.0f} млн слов): ≈ {est:.1f} ч (порог {cc['max_hours_full']} ч)")


if __name__ == "__main__":
    main()
