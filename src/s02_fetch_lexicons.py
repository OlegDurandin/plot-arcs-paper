"""Этап 2. Загрузка лексиконов (один раз) с записью происхождения: адрес, дата, SHA-256, лицензия.

Запуск:
    .venv/bin/python src/s02_fetch_lexicons.py configs/s02_emotion.yaml
Файлы — в external_dir/<имя>/ (после загрузки — только чтение), provenance.json рядом.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

UA = {"User-Agent": "research-corpus-builder/0.1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    base = ROOT / cfg["external_dir"]
    for name, lc in cfg["lexicons"].items():
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        prov = d / "provenance.json"
        if prov.exists():
            print(f"{name}: уже загружен")
            continue
        fn = d / Path(lc["url"].split("?")[0]).name
        with urllib.request.urlopen(urllib.request.Request(lc["url"], headers=UA), timeout=120) as r:
            data = r.read()
        fn.write_bytes(data)
        extracted = []
        if fn.suffix == ".zip":
            with zipfile.ZipFile(fn) as z:
                for m in z.namelist():
                    if m.startswith("__MACOSX") or m.endswith("/"):
                        continue
                    target = (d / "unzipped" / m).resolve()
                    if not str(target).startswith(str((d / "unzipped").resolve())):
                        continue  # защита от путей вида ../
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(m))
                    extracted.append(m)
        prov.write_text(json.dumps(dict(name=name, url=lc["url"], license=lc["license"],
                                        downloaded=dt.datetime.now().isoformat(timespec="seconds"),
                                        file=fn.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                        extracted=extracted[:200], n_extracted=len(extracted)),
                                   ensure_ascii=False, indent=2))
        print(f"{name}: {len(data)} байт, извлечено файлов: {len(extracted)}")


if __name__ == "__main__":
    main()
