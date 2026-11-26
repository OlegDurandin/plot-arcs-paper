"""Анонимная копия репозитория для двойного слепого рецензирования (Ворота 6, 2026-11-26).

Берёт закоммиченное состояние (git archive HEAD — data/, .env и неотслеживаемое не попадают), заменяет в текстовых
файлах имена, e-mail, ссылки на репозиторий и абсолютные пути (configs/s10_anonymize.yaml), проверяет, что запрещённых
шаблонов не осталось (и в путях файлов), и создаёт в out_dir новый git-репозиторий с одним коммитом от анонимного
автора. Ничего никуда не отправляет: загрузку (anonymous.4open.science или аналог) делает пользователь.

Запуск:  .venv/bin/python src/s10_anonymize.py configs/s10_anonymize.yaml
"""
from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    out = (ROOT / cfg["out_dir"]).resolve()
    assert ROOT.resolve() not in out.parents and out != ROOT.resolve(), "out_dir должен быть вне репозитория"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(out, filter="data")
    for e in cfg.get("exclude", []):
        (out / e).unlink(missing_ok=True)
    reps = [(re.compile(a, re.I), b) for a, b in cfg["replace"]]
    forbid = re.compile("|".join(cfg["forbid"]), re.I)
    changed, bad = [], []
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out).as_posix()
        if forbid.search(rel):
            bad.append(f"путь: {rel}")
        if not p.is_file() or p.suffix.lower() not in cfg["text_ext"] and p.name not in cfg["text_ext"]:
            continue
        s = p.read_text(encoding="utf-8", errors="strict")
        t = s
        for rx, b in reps:
            t = rx.sub(b, t)
        if t != s:
            p.write_text(t, encoding="utf-8")
            changed.append(rel)
        for i, line in enumerate(t.splitlines(), 1):
            if forbid.search(line):
                bad.append(f"{rel}:{i}: {line[:120]}")
    if bad:
        print("ОСТАЛИСЬ запрещённые шаблоны:\n" + "\n".join(bad[:50]))
        sys.exit(1)
    name, email = re.match(r"(.*) <(.*)>", cfg["author"]).groups()
    env = {**os.environ, "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_COMMITTER_NAME": name,
           "GIT_COMMITTER_EMAIL": email, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for c in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", cfg["message"]]):
        subprocess.run(c, cwd=out, check=True, env=env)
    log = subprocess.run(["git", "log", "--format=%an <%ae> | %cn <%ce> | %s"], cwd=out, capture_output=True, text=True, env=env).stdout
    n = sum(1 for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"анонимная копия: {out}\nфайлов: {n}; изменено при анонимизации: {len(changed)} — {', '.join(changed)}\nкоммит: {log.strip()}")


if __name__ == "__main__":
    main()
