"""v6 pipeline – 전체 파이프라인 오케스트레이터"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from .extractors import extract_any
from .anchor_builder import (
    build_equipment_name_map,
    build_name_map_from_lines,
    extract_names_from_text,
    normalize_equipment_no as anchor_normalize,
    _clean_candidate,
    _score_candidate,
)
from .event_builder import build_events_for_equipment
from .repeat_judger import judge_repeat_cases, merge_cases_per_equipment
from .task_builder import build_task_dataframe
from .schemas import MaintenanceEvent, RepeatCase


def _list_files(data_dir: Path) -> List[Path]:
    supported = {".pdf", ".xlsx", ".xls", ".xlsm", ".xlsb", ".txt", ".csv"}
    return [p for p in sorted(data_dir.iterdir()) if p.is_file() and p.suffix.lower() in supported]


def _extract_lines_from_pdf(path: Path) -> List[str]:
    try:
        import pdfplumber
    except ImportError:
        return []
    lines = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    line = re.sub(r"\s+", " ", line).strip()
                    if line:
                        lines.append(line)
    except Exception:
        pass
    return lines


def run_pipeline_v6(
    data_dir: str | Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Tuple[pd.DataFrame, List[RepeatCase], List[MaintenanceEvent], Dict[str, str]]:
    data_dir = Path(data_dir)
    files = _list_files(data_dir)
    total = len(files)

    all_chunks = []
    all_line_candidates: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    for i, path in enumerate(files, start=1):
        if progress_callback:
            progress_callback(i, total, path.name)
        try:
            df = extract_any(path)
            if df is not None and not df.empty:
                all_chunks.append(df)
        except Exception:
            pass
        if path.suffix.lower() == ".pdf":
            lines = _extract_lines_from_pdf(path)
            if lines:
                line_cands = build_name_map_from_lines(lines, path.name)
                for eq, cands in line_cands.items():
                    all_line_candidates[eq].extend(cands)

    if not all_chunks:
        return pd.DataFrame(), [], [], {}

    events_raw = pd.concat(all_chunks, ignore_index=True)

    for eq in events_raw["equipment_no"].dropna().unique():
        eq_norm = anchor_normalize(str(eq))
        if not eq_norm:
            continue
        sub = events_raw[events_raw["equipment_no"] == eq]
        if "equipment_name" in sub.columns:
            for raw_name in sub["equipment_name"].dropna().astype(str).unique():
                raw_name = raw_name.strip()
                if not raw_name:
                    continue
                all_line_candidates[eq_norm].extend(extract_names_from_text(eq_norm, raw_name))
                cleaned = _clean_candidate(raw_name, eq_norm)
                if cleaned:
                    all_line_candidates[eq_norm].append((cleaned, "fallback", _score_candidate(cleaned, "fallback")))
        if "sentence" in sub.columns:
            for sent in sub["sentence"].dropna().astype(str).unique():
                all_line_candidates[eq_norm].extend(extract_names_from_text(eq_norm, sent))

    equipment_names = build_equipment_name_map(dict(all_line_candidates))

    events_raw["equipment_no"] = events_raw["equipment_no"].apply(lambda x: anchor_normalize(str(x)) if pd.notna(x) else "")
    events_raw = events_raw[events_raw["equipment_no"] != ""].copy()

    all_events: List[MaintenanceEvent] = []
    equipment_events: Dict[str, List[MaintenanceEvent]] = {}

    for eq_no, eq_group in events_raw.groupby("equipment_no"):
        eq_name = equipment_names.get(str(eq_no), "")
        eq_events = build_events_for_equipment(str(eq_no), eq_name, eq_group)
        if eq_events:
            all_events.extend(eq_events)
            equipment_events[str(eq_no)] = eq_events

    repeat_cases = merge_cases_per_equipment(judge_repeat_cases(equipment_events, equipment_names))
    task_df = build_task_dataframe(repeat_cases)
    return task_df, repeat_cases, all_events, equipment_names
