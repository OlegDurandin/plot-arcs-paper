"""Общие утилиты: конфиг, хеш конфига, токенизация для подсчёта слов, стабильная выборка."""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Слово = последовательность букв/цифр, допускаются внутренние дефис и апостроф.
# Единый токенизатор для подсчёта длины во всех корпусах (поля n_words источников
# считались по-разному). Точная токенизация razdel — на этапе 1.
WORD_RE = re.compile(r"[^\W_]+(?:[-‐’'][^\W_]+)*")


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def in_sample(key: str, frac: float) -> bool:
    """Воспроизводимая выборка по crc32 ключа (та же схема, что у поля bucket в Проза.ру)."""
    if frac >= 1.0:
        return True
    return (zlib.crc32(key.encode()) % 10000) < frac * 10000


def normalize_for_hash(text: str) -> str:
    return " ".join(words(text.lower().replace("ё", "е")))
