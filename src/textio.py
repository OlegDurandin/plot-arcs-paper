"""Загрузка исходного текста по text_path из манифеста / analysis_set.

Форматы text_path:
  sqlite:<db>#stories/<story_id>     — Проза.ру (таблица stories, поле text)
  sqlite:<db>#tales/<tale_id>        — fairytalez (таблица tales, поле text)
  parquet:<file>#af_no=<n>           — Афанасьев (data/interim/afanasyev.parquet)
  <dir>/texts/*/*/<work_id>          — Ficbook, произведение: главы из ficbook_chapters.parquet по порядку
  <path>/part-NNN.txt                — Ficbook, одна глава
  members:<parquet>                  — склеенная серия Проза.ру: записи из proza_work_members.parquet по pos

load(doc_id, text_path) -> list[(part_id, text)]: одна часть для обычного текста, несколько — для произведений
из глав (границы глав сохраняются для RQ3).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from common import ROOT


class TextLoader:
    def __init__(self, interim: str = "data/interim"):
        self.inter = ROOT / interim
        self._con: dict[str, sqlite3.Connection] = {}
        self._af: pd.DataFrame | None = None
        self._fbch: dict[str, list[tuple[int, str]]] | None = None
        self._members: dict[str, list[str]] | None = None
        self._pz_paths: dict[str, str] | None = None

    # -- источники
    def _sqlite(self, db: str) -> sqlite3.Connection:
        if db not in self._con:
            self._con[db] = sqlite3.connect(f"file:{ROOT / db}?immutable=1", uri=True)
        return self._con[db]

    def _sqlite_text(self, path: str) -> str:
        db, rest = path[len("sqlite:"):].split("#", 1)
        table, key = rest.split("/", 1)
        col = "story_id" if table == "stories" else "tale_id"
        row = self._sqlite(db).execute(f"select text from {table} where {col}=?", (key,)).fetchone()
        return (row[0] or "") if row else ""

    def _afanasyev(self, path: str) -> str:
        if self._af is None:
            f, _ = path[len("parquet:"):].split("#", 1)
            self._af = pd.read_parquet(ROOT / f).dropna(subset=["af_no"]).set_index("af_no")
        n = int(path.split("af_no=")[1])
        return self._af.loc[n, "text"]

    def _ficbook_parts(self, doc_id: str) -> list[tuple[int, str]]:
        if self._fbch is None:
            ch = pd.read_parquet(self.inter / "ficbook_chapters.parquet",
                                 columns=["doc_id", "chapter_number", "ch_ok", "text_path"])
            ch = ch[ch["ch_ok"]].sort_values(["doc_id", "chapter_number"])
            self._fbch = {k: list(zip(g["chapter_number"], g["text_path"])) for k, g in ch.groupby("doc_id")}
        return self._fbch.get(doc_id, [])

    def _member_ids(self, work_id: str) -> list[str]:
        if self._members is None:
            m = pd.read_parquet(self.inter / "proza_work_members.parquet").sort_values(["work_id", "pos"])
            self._members = {k: list(g["doc_id"]) for k, g in m.groupby("work_id")}
            man = pd.read_parquet(self.inter / "manifest.parquet", columns=["doc_id", "corpus", "text_path"])
            man = man[man["corpus"] == "proza"]
            self._pz_paths = dict(zip(man["doc_id"], man["text_path"]))
        return self._members.get(work_id, [])

    # -- публичный интерфейс
    def load(self, doc_id: str, text_path: str) -> list[tuple[str, str]]:
        if text_path.startswith("sqlite:"):
            return [(doc_id, self._sqlite_text(text_path))]
        if text_path.startswith("parquet:"):
            return [(doc_id, self._afanasyev(text_path))]
        if text_path.startswith("members:"):
            parts = []
            for mid in self._member_ids(doc_id):
                parts.append((mid, self._sqlite_text(self._pz_paths[mid])))
            return parts
        if text_path.endswith(".txt"):
            return [(doc_id, (ROOT / text_path).read_text(encoding="utf-8", errors="replace"))]
        # произведение Ficbook целиком
        return [(f"{doc_id}#{n}", (ROOT / p).read_text(encoding="utf-8", errors="replace"))
                for n, p in self._ficbook_parts(doc_id)]
