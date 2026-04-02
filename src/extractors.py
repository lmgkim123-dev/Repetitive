from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .task_builder import build_category_extract_dataframe

_HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F4E78")
_HEADER_FONT = Font(color="FFFFFF", bold=True)

_CENTER_HEADERS = {
    "NO",
    "Equipment No",
    "발생구분",
    "발생년도수",
    "발생년도",
    "발췌 Category",
    "검토필요여부",
    "year",
    "event_year",
    "event_date",
    "source_count",
    "confidence",
    "발생횟수",
    "최근발생일",
    "source_type",
}

_WIDE_TEXT_HEADERS = {
    "설비명",
    "반복부위",
    "TA 조치사항",
    "추후 권고사항",
    "제목",
    "상세 내용",
    "source_files",
    "finding_location",
    "finding_damage",
    "finding_measurement",
    "action_type",
    "action_detail",
    "recommendation",
    "evidence_summary",
    "repeat_reason",
    "action_cluster",
    "location_cluster",
    "damage_cluster",
    "sources",
    "titles",
    "details",
    "출처",
    "대표조치",
    "상세이력",
    "검토메모",
    "title",
    "detail",
    "exclude_reason",
}

_FIXED_WIDTHS = {
    "NO": 7,
    "Equipment No": 15,
    "설비명": 28,
    "발생구분": 13,
    "발생년도수": 11,
    "발생년도": 18,
    "발췌 Category": 18,
    "반복부위": 24,
    "TA 조치사항": 52,
    "추후 권고사항": 52,
    "제목": 38,
    "상세 내용": 80,
    "검토필요여부": 12,
    "equipment_no": 15,
    "equipment_name": 28,
    "year": 10,
    "source_files": 30,
    "finding_location": 22,
    "finding_damage": 24,
    "finding_measurement": 20,
    "action_type": 22,
    "action_detail": 56,
    "recommendation": 56,
    "evidence_summary": 70,
    "action_cluster": 20,
    "location_cluster": 20,
    "damage_cluster": 20,
    "repeat_reason": 44,
    "confidence": 10,
    "Line Number": 18,
    "발생횟수": 10,
    "최근발생일": 14,
    "출처": 24,
    "대표조치": 42,
    "상세이력": 60,
    "검토메모": 34,
    "line_no": 18,
    "event_date": 14,
    "event_year": 10,
    "sources": 24,
    "source_count": 10,
    "titles": 36,
    "details": 64,
    "source_type": 16,
    "title": 28,
    "detail": 70,
    "exclude_reason": 28,
}


def _safe_df(df) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df.copy()


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _visible_len(value) -> int:
    text = _stringify(value)
    if not text:
        return 0
    return max((len(line) for line in text.splitlines()), default=0)


def _estimate_width(ws, col_idx: int, header: str) -> int:
    if header in _FIXED_WIDTHS:
        return _FIXED_WIDTHS[header]
    max_len = len(header)
    max_row = min(ws.max_row, 200)
    for row_idx in range(2, max_row + 1):
        max_len = max(max_len, _visible_len(ws.cell(row=row_idx, column=col_idx).value))
    if header in _WIDE_TEXT_HEADERS:
        return max(24, min(60, int(max_len * 1.05) + 2))
    return max(10, min(24, int(max_len * 1.1) + 2))


def _estimated_wrapped_lines(text: str, width: int) -> int:
    if not text:
        return 1
    usable = max(12, width - 2)
    total = 0
    for raw_line in text.splitlines() or [text]:
        line = raw_line.strip() or " "
        total += max(1, ceil(len(line) / usable))
    return total


def _apply_sheet_formatting(ws) -> None:
    if ws.max_row == 0 or ws.max_column == 0:
        return

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = True

    headers = []
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        header = _stringify(cell.value)
        headers.append(header)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        width = _estimate_width(ws, col_idx, header)
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.row_dimensions[1].height = 24

    for row_idx in range(2, ws.max_row + 1):
        row_line_estimate = 1
        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)
            text = _stringify(cell.value)
            width = int(ws.column_dimensions[get_column_letter(col_idx)].width or 12)
            if header in _CENTER_HEADERS:
                cell.alignment = Alignment(horizontal="center", vertical="top", wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            if header in _WIDE_TEXT_HEADERS or len(text) > 24 or "\n" in text:
                row_line_estimate = max(row_line_estimate, _estimated_wrapped_lines(text, width))
        ws.row_dimensions[row_idx].height = min(180, max(20, 16 * row_line_estimate + 4))


def export_dataframes(sheet_frames: Iterable[tuple[str, pd.DataFrame]], output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        written_names: list[str] = []
        for sheet_name, df in sheet_frames:
            safe_sheet = str(sheet_name)[:31] or "Sheet1"
            _safe_df(df).to_excel(writer, index=False, sheet_name=safe_sheet)
            written_names.append(safe_sheet)
        workbook = writer.book
        for sheet_name in written_names:
            _apply_sheet_formatting(workbook[sheet_name])
    return output_path


def export_excel(task_df, repeat_cases, all_events, output_path, category_source_df=None, extra_sheets: dict[str, pd.DataFrame] | None = None):
    task_df = _safe_df(task_df)
    category_source_df = task_df if category_source_df is None else _safe_df(category_source_df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_rows = []
    for c in repeat_cases or []:
        case_rows.append({
            "equipment_no": c.equipment_no,
            "설비명": c.equipment_name,
            "발생년도수": len(sorted(set(c.years))),
            "발생년도": ", ".join(map(str, sorted(set(c.years)))),
            "action_cluster": c.action_cluster,
            "location_cluster": c.location_cluster,
            "damage_cluster": c.damage_cluster,
            "repeat_reason": c.repeat_reason,
            "confidence": c.confidence,
        })
    cases_df = pd.DataFrame(case_rows)

    event_rows = []
    for e in all_events or []:
        event_rows.append({
            "equipment_no": e.equipment_no,
            "equipment_name": e.equipment_name,
            "year": e.report_year,
            "source_files": ", ".join(e.source_files or []),
            "finding_location": e.finding_location,
            "finding_damage": e.finding_damage,
            "finding_measurement": e.finding_measurement,
            "action_type": e.action_type,
            "action_detail": e.action_detail,
            "recommendation": e.recommendation,
            "evidence_summary": e.evidence_summary,
        })
    events_df = pd.DataFrame(event_rows)
    category_df = build_category_extract_dataframe(category_source_df)

    sheet_frames: list[tuple[str, pd.DataFrame]] = [
        ("과제후보_등록형식", task_df),
        ("카테고리별_발췌", category_df),
        ("반복정비_Case", cases_df),
        ("연도별_정비이벤트", events_df),
    ]
    for sheet_name, df in (extra_sheets or {}).items():
        sheet_frames.append((sheet_name, _safe_df(df)))

    return export_dataframes(sheet_frames, output_path)
