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


_PDF_EQUIP_HEADER_RE = re.compile(r"\b([0-9]{2,3}[A-Z]{1,3}-[0-9]{3,4}[A-Z]?)\b(?:\s*\(([^)]+)\))?", re.I)
_PDF_ACTION_HEADER_RE = re.compile(r"^(?:#+\s*)?(?:\d+[.)]?\s*)?(?:금번\s*TA\s*조치사항|보수\s*/\s*개선\s*내용|보수\s*개선\s*내용)\s*$", re.I)
_PDF_ACTION_END_RE = re.compile(r"^(?:#+\s*)?(?:\d+[.)]?\s*)?(?:향후\s*조치사항|차기\s*TA|Recommend|검사사진|사진)\s*$", re.I)
_PDF_IGNORE_LINE_RE = re.compile(r"^(?:#+\s*)?(?:구분|초기\s*검사|상세\s*검사|장치\s*기본\s*정보|개방검사\s*결과|N/?A|Service|Material|Operating|검사\s*결과)\s*$", re.I)
_PDF_BULLET_RE = re.compile(r"^\s*(?:[-•·▪◦]|\(?\d+\)?[.)])\s*")
_PDF_YEAR_RE = re.compile(r"(20\s*\d{2})\s*년")


def _infer_report_year(path: Path, lines: List[str]) -> int | None:
    m = _PDF_YEAR_RE.search(path.name)
    if m:
        return int(re.sub(r"\s+", "", m.group(1)))
    for line in lines[:60]:
        m = _PDF_YEAR_RE.search(line)
        if m:
            return int(re.sub(r"\s+", "", m.group(1)))
    return None


def _normalize_pdf_line(line: str) -> str:
    line = re.sub(r"\s+", " ", str(line or "")).strip()
    line = re.sub(r"^#+\s*", "", line)
    return line


def _fallback_action_tags(text: str) -> str:
    t = str(text or "")
    tags = []
    if re.search(r"교체|replace|retube|retubing|신규\s*제작|신규\s*교체|제작\s*후\s*교체|신규\s*자재", t, re.I):
        tags.append("replace")
    if re.search(r"육성\s*용접|용접\s*보수|보수\s*용접|weld|overlay|hardfacing|stitch\s*welding", t, re.I):
        tags.append("weld_repair")
    if re.search(r"도장|coating|paint", t, re.I):
        tags.append("coating_repair")
    if re.search(r"plugging|plug|blind\s*처리|막음", t, re.I):
        tags.append("plugging")
    if re.search(r"보수|보강|재시공|설치|시공", t, re.I):
        tags.append("structural_repair")
    return ", ".join(dict.fromkeys(tags))


def _fallback_damage_tags(text: str) -> str:
    t = str(text or "")
    tags = []
    if re.search(r"부식|corrosion|pitting|pit", t, re.I):
        tags.append("corrosion")
    if re.search(r"균열|crack|linear indication|선형\s*결함", t, re.I):
        tags.append("cracking")
    if re.search(r"누설|leak|천공", t, re.I):
        tags.append("leak")
    if re.search(r"감육|두께감소|thinning", t, re.I):
        tags.append("thinning")
    if re.search(r"손상|damage|파손|변형|마모|탈락", t, re.I):
        tags.append("damage")
    return ", ".join(dict.fromkeys(tags))


def _extract_fallback_rows_from_pdf(path: Path) -> pd.DataFrame:
    lines = _extract_lines_from_pdf(path)
    if not lines:
        return pd.DataFrame()

    year = _infer_report_year(path, lines)
    current_eq = ""
    current_name = ""
    in_action = False
    items: List[str] = []
    rows = []

    def flush_items():
        nonlocal items, rows, current_eq, current_name, year
        if not current_eq or not year:
            items = []
            return
        for item in items:
            item = _normalize_pdf_line(item)
            item = re.sub(r"^[-•·▪◦]\s*", "", item)
            item = re.sub(r"^\(?\d+\)?[.)]?\s*", "", item)
            item = re.sub(r"\s+", " ", item).strip()
            if len(item) < 8:
                continue
            if not re.search(r"교체|replace|retube|retubing|보수|repair|보강|용접|weld|도장|coating|plugging|plug|blind\s*처리|재시공|설치|시공", item, re.I):
                continue
            rows.append({
                'equipment_no': current_eq,
                'equipment_name': current_name,
                'year': year,
                'sentence': item,
                'source_file': path.name,
                'action_tags': _fallback_action_tags(item),
                'damage_tags': _fallback_damage_tags(item),
            })
        items = []

    for raw_line in lines:
        line = _normalize_pdf_line(raw_line)
        if not line:
            continue

        m = _PDF_EQUIP_HEADER_RE.search(line)
        if m and not _PDF_ACTION_HEADER_RE.search(line):
            if in_action:
                flush_items()
                in_action = False
            current_eq = m.group(1).upper()
            current_name = (m.group(2) or "").strip()
            continue

        if _PDF_ACTION_HEADER_RE.search(line):
            if in_action:
                flush_items()
            in_action = True
            continue

        if in_action and _PDF_ACTION_END_RE.search(line):
            flush_items()
            in_action = False
            continue

        if not in_action:
            continue
        if _PDF_IGNORE_LINE_RE.search(line):
            continue

        if not items:
            items.append(line)
        elif _PDF_BULLET_RE.match(line):
            items.append(line)
        else:
            items[-1] = f"{items[-1]} {line}".strip()

    if in_action:
        flush_items()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['equipment_no', 'year', 'sentence', 'source_file']).reset_index(drop=True)
    return df


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
        except Exception:
            df = pd.DataFrame()

        if path.suffix.lower() == ".pdf":
            fallback_df = _extract_fallback_rows_from_pdf(path)
            if fallback_df is not None and not fallback_df.empty:
                if df is None or df.empty:
                    df = fallback_df
                else:
                    df = pd.concat([df, fallback_df], ignore_index=True)
                    keep_cols = [c for c in ["equipment_no", "equipment_name", "year", "sentence", "source_file", "action_tags", "damage_tags"] if c in df.columns]
                    if keep_cols:
                        df = df.drop_duplicates(subset=keep_cols).reset_index(drop=True)

            lines = _extract_lines_from_pdf(path)
            if lines:
                line_cands = build_name_map_from_lines(lines, path.name)
                for eq, cands in line_cands.items():
                    all_line_candidates[eq].extend(cands)

        if df is not None and not df.empty:
            all_chunks.append(df)

    if not all_chunks:
        return pd.DataFrame(), [], [], {}

    events_raw = pd.concat(all_chunks, ignore_index=True)
    dedup_cols = [c for c in ["equipment_no", "equipment_name", "year", "sentence", "source_file", "action_tags", "damage_tags"] if c in events_raw.columns]
    if dedup_cols:
        events_raw = events_raw.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

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
