from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List


def list_files(root: str | os.PathLike[str]) -> List[Path]:
    paths: List[Path] = []
    for base, _, files in os.walk(root):
        for f in files:
            paths.append(Path(base) / f)
    return paths


def normalize_whitespace(text: str) -> str:
    text = text.replace('\u3000', ' ')
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[\t\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = text.replace('。', '. ')
    text = text.replace('•', '. ')
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', text)
    results = []
    for part in parts:
        part = normalize_whitespace(part)
        if len(part) >= 8:
            results.append(part)
    return results


def safe_str(v) -> str:
    if v is None:
        return ''
    return str(v).strip()


def first_nonempty(values: Iterable[str]) -> str:
    for v in values:
        if v and str(v).strip():
            return str(v).strip()
    return ''
