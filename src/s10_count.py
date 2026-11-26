"""Счёт знаков статьи по правилам «Диалога»: основной текст без аннотации и литературы, знаки с пробелами.

Считается всё от первого раздела («## 1.») до раздела «Литература»: текст, подписи, таблицы (содержимое ячеек
и пробелы между ними). Не считаются: HTML-комментарии, строки рисунков ![…](…), разметка Markdown (#, **, *, |, ---).
Худший случай — подписи и таблицы внутри лимита.

Запуск:  .venv/bin/python src/s10_count.py reports/paper/ru/paper_ru.md [--limit 20000]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def clean(line: str) -> str:
    if re.match(r"^\s*!\[", line) or re.match(r"^\s*\|?\s*-{3,}", line) or re.match(r"^\s*-{3,}\s*$", line):
        return ""
    if line.strip().startswith("|"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        line = " ".join(c for c in cells if c)
    line = re.sub(r"^#+\s*", "", line)
    line = re.sub(r"^\s*[-*]\s+", "", line)
    line = line.replace("**", "").replace("*", "")
    return line.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=20000)
    a = ap.parse_args()
    t = re.sub(r"<!--.*?-->", "", Path(a.path).read_text(encoding="utf-8"), flags=re.S)
    i0 = t.index("\n## 1.")
    i1 = t.index("\n## Литература") if "\n## Литература" in t else len(t)
    body = t[i0:i1]
    secs = re.split(r"\n(?=## )", body.strip())
    total = 0
    print(f"{'раздел':<55} знаков")
    for sec in secs:
        lines = [clean(l) for l in sec.split("\n")]
        txt = " ".join(l for l in lines if l)
        n = len(txt)
        total += n
        print(f"{sec.split(chr(10))[0][3:55]:<55} {n:>6}")
    print(f"{'ИТОГО (лимит ' + str(a.limit) + ')':<55} {total:>6}  {'OK' if total <= a.limit else 'ПРЕВЫШЕН'}")


if __name__ == "__main__":
    main()
