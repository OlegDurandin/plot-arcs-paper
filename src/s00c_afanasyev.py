"""Этап 0c. Корпус «Народные русские сказки» А. Н. Афанасьева (Викитека, общественное достояние).

1) download: вики-разметка всех подстраниц сборника -> raw_dir/pages.jsonl (если файла нет);
2) parse: подстраница = сюжет; разделы «== N ==» = варианты под номерами Афанасьева.
   Каждый вариант — отдельный текст. Разметка (шаблоны, сноски, ссылки) удаляется.

Запуск:
    .venv/bin/python src/s00c_afanasyev.py configs/s00_inventory.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config, words  # noqa: E402


def api(cfg, params):
    data = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"}).encode()
    req = urllib.request.Request(cfg["api"], data=data, headers={"User-Agent": cfg["user_agent"]})  # POST: длинные списки заголовков
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("  повтор:", e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("API недоступно")


def download(cfg, raw: Path):
    out = raw / "pages.jsonl"
    if out.exists():
        print("сырьё уже скачано:", out)
        return out
    titles, cont = [], {}
    while True:
        r = api(cfg, {"action": "query", "list": "allpages", "apprefix": cfg["prefix"], "apnamespace": 0,
                      "aplimit": 500, **cont})
        titles += [p["title"] for p in r["query"]["allpages"]]
        if "continue" not in r:
            break
        cont = r["continue"]
    print("подстраниц:", len(titles))
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i in range(0, len(titles), cfg["batch"]):
            r = api(cfg, {"action": "query", "prop": "revisions", "rvprop": "content|ids|timestamp",
                          "rvslots": "main", "titles": "|".join(titles[i:i + cfg["batch"]])})
            for p in r["query"]["pages"]:
                rev = p["revisions"][0]
                f.write(json.dumps(dict(title=p["title"], pageid=p["pageid"], revid=rev["revid"],
                                        timestamp=rev["timestamp"], wikitext=rev["slots"]["main"]["content"]),
                                   ensure_ascii=False) + "\n")
            time.sleep(1)
    tmp.rename(out)
    return out


HEAD = re.compile(r"^==\s*(\d+)\s*(?:<ref[^>]*>(.*?)</ref>|<ref[^>]*/>)?\s*(.*?)==\s*$", re.M)
ATU = re.compile(r"\[\[Категория:Сказочный сюжет\s*([\d*A-Za-zА-Яа-я]+)\s*:\s*([^\]|]+)")


def strip_wiki(t: str) -> tuple[str, list[str]]:
    notes = re.findall(r"<ref[^>]*>(.*?)</ref>", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", t, flags=re.S)
    for _ in range(3):                                   # вложенные шаблоны
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\[\[(?:Категория|Файл|File|Image):[^\]]*\]\]", "", t)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("'''", "").replace("''", "")
    t = re.sub(r"&nbsp;", " ", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip(), notes


def parse(cfg, raw_file: Path) -> pd.DataFrame:
    rows = []
    for line in raw_file.open(encoding="utf-8"):
        p = json.loads(line)
        wt = p["wikitext"]
        atu = ATU.findall(wt)
        wt_body = re.split(r"^==\s*Примечания\s*==", wt, flags=re.M)[0]
        heads = list(HEAD.finditer(wt_body))
        plot = p["title"].split("/", 1)[1]
        if not heads:  # страница без нумерованных вариантов (напр., оглавление)
            rows.append(dict(af_page=plot, af_no=None, text=strip_wiki(wt_body)[0], note="", parsed=False,
                             atu=";".join(a for a, _ in atu), atu_name=";".join(n.strip() for _, n in atu),
                             revid=p["revid"]))
            continue
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(wt_body)
            body = wt_body[h.end():end]
            body = re.split(r"^==\s*[^=\d\s][^=]*==\s*$", body, maxsplit=1, flags=re.M)[0]  # «== Ссылки ==» и т.п.
            text, _ = strip_wiki(body)
            note = strip_wiki(h.group(2) or "")[0]
            rows.append(dict(af_page=plot, af_no=int(h.group(1)), af_suffix=(h.group(3) or "").strip(), text=text,
                             note=note, parsed=True, atu=";".join(a for a, _ in atu),
                             atu_name=";".join(n.strip() for _, n in atu), revid=p["revid"]))
    df = pd.DataFrame(rows)
    df["n_words"] = df["text"].map(lambda s: len(words(s)))
    # Украинские / белорусские тексты в сборнике: по буквам, которых нет в русском
    ua = df["text"].str.count(r"[іїєґІЇЄҐ]")
    by = df["text"].str.count(r"[ўЎ]")
    df["lang"] = "ru"
    df.loc[(ua >= 3) & (ua >= by), "lang"] = "uk"
    df.loc[(by >= 3) & (by > ua), "lang"] = "be"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ac = cfg["afanasyev"]
    raw = ROOT / ac["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    f = download(ac, raw)
    df = parse(ac, f)
    df.to_parquet(ROOT / ac["out"], index=False)
    ok = df[df["parsed"]]
    print(f"подстраниц: {df['af_page'].nunique()}, вариантов с номером: {len(ok)}, без разбора: {int((~df['parsed']).sum())}")
    print(f"номера: {ok['af_no'].min()}–{ok['af_no'].max()}, уникальных {ok['af_no'].nunique()}, дубли номеров: {int(ok['af_no'].duplicated().sum())}")
    print(ok["n_words"].describe().round(0).to_dict())
    print("языки:", ok["lang"].value_counts().to_dict(), "| пустых:", int((ok["n_words"] == 0).sum()),
          "| остатки разметки:", int(ok["text"].str.contains(r"\{\{|\}\}|<|\[\[|^==", regex=True).sum()))


if __name__ == "__main__":
    main()
