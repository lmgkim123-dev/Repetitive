"""v6 pipeline – refixed minimal pipeline for fixed equipment."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from .anchor_builder import build_equipment_name_map, normalize_equipment_no as anchor_normalize, _clean_candidate, _score_candidate
from .event_builder import build_events_for_equipment
from .extractors import extract_any
from .repeat_judger import judge_repeat_cases, merge_cases_per_equipment
from .schemas import MaintenanceEvent, RepeatCase
from .task_builder import build_task_dataframe



def _list_files(data_dir: Path) -> List[Path]:
    supported = {".pdf", ".xlsx", ".xls", ".xlsm", ".xlsb", ".txt", ".csv"}
    return [p for p in sorted(data_dir.iterdir()) if p.is_file() and p.suffix.lower() in supported]



def run_pipeline_v6(
    data_dir: str | Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Tuple[pd.DataFrame, List[RepeatCase], List[MaintenanceEvent], Dict[str, str]]:
    data_dir = Path(data_dir)
    files = _list_files(data_dir)
    total = len(files)

    all_chunks = []
    all_line_candidates: Dict[str, List[tuple[str, str, float]]] = defaultdict(list)

    for i, path in enumerate(files, start=1):
        if progress_callback:
            progress_callback(i, total, path.name)
        try:
            df = extract_any(path)
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            continue
        all_chunks.append(df)
        if "equipment_no" in df.columns and "equipment_name" in df.columns:
            for _, row in df[["equipment_no", "equipment_name"]].dropna().drop_duplicates().iterrows():
                eq_norm = anchor_normalize(row["equipment_no"])
                name = _clean_candidate(row["equipment_name"], eq_norm)
                if eq_norm and name:
                    all_line_candidates[eq_norm].append((name, "column_equipment_name", _score_candidate(name, "column_equipment_name")))

    if not all_chunks:
        return pd.DataFrame(), [], [], {}

    events_raw = pd.concat(all_chunks, ignore_index=True)
    if events_raw.empty:
        return pd.DataFrame(), [], [], {}

    events_raw["equipment_no"] = events_raw["equipment_no"].apply(lambda x: anchor_normalize(x) if pd.notna(x) else "")
    events_raw = events_raw[events_raw["equipment_no"] != ""].copy()
    if events_raw.empty:
        return pd.DataFrame(), [], [], {}

    equipment_names = build_equipment_name_map(dict(all_line_candidates))

    all_events: List[MaintenanceEvent] = []
    equipment_events: Dict[str, List[MaintenanceEvent]] = {}

    for eq_no, eq_group in events_raw.groupby("equipment_no"):
        eq_name = equipment_names.get(str(eq_no), "")
        if not eq_name and "equipment_name" in eq_group.columns:
            names = [
                _clean_candidate(v, str(eq_no))
                for v in eq_group["equipment_name"].dropna().astype(str).tolist()
            ]
            names = [n for n in names if n]
            if names:
                eq_name = sorted(names, key=lambda n: (-names.count(n), -len(n), n))[0]
        eq_events = build_events_for_equipment(str(eq_no), eq_name, eq_group)
        if eq_events:
            all_events.extend(eq_events)
            equipment_events[str(eq_no)] = eq_events
            if eq_events[0].equipment_name:
                equipment_names[str(eq_no)] = eq_events[0].equipment_name

    repeat_cases = merge_cases_per_equipment(judge_repeat_cases(equipment_events, equipment_names))
    task_df = build_task_dataframe(repeat_cases)
    return task_df, repeat_cases, all_events, equipment_names
