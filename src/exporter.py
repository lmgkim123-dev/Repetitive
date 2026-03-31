"""v6 exporter – 엑셀 출력"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .schemas import MaintenanceEvent, RepeatCase
from .task_builder import TASK_COLUMNS, build_category_extract_dataframe


def _apply_sheet_style(ws, sheet_name: str) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap_alignment = Alignment(vertical="top", wrap_text=True)
    center_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.zoomScale = 90
    ws.row_dimensions[1].height = 30

    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_alignment
        cell.border = border

    col_widths = {
        "NO": 7, "Equipment No": 16, "equipment_no": 16,
        "설비명": 28, "equipment_name": 28,
        "발생구분": 14, "발생년도수": 10, "발생년도": 16,
        "발췌 Category": 18, "Category": 18,
        "반복부위": 22, "반복손상": 18, "대표조치": 22,
        "TA 조치사항": 68, "추후 권고사항": 60,
        "제목": 36, "상세 내용": 100,
        "반복판정근거": 54, "검토필요여부": 10,
        "year": 10, "source_files": 36,
        "finding_location": 24, "finding_damage": 20, "finding_measurement": 18,
        "action_type": 22, "action_detail": 60,
        "recommendation": 60, "evidence_summary": 80,
        "repeat_key": 36, "action_cluster": 20, "location_cluster": 22,
        "damage_cluster": 20, "years": 16, "repeat_reason": 54, "confidence": 10,
    }

    for idx, column_cells in enumerate(ws.columns, start=1):
        header = str(column_cells[0].value or "")
        ws.column_dimensions[get_column_letter(idx)].width = col_widths.get(header, 16)

    for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        max_lines = 1
        for cell in row:
            cell.alignment = wrap_alignment
            cell.border = border
            value = "" if cell.value is None else str(cell.value)
            line_count = max(value.count("\n") + 1, len(value) // 44 + 1)
            max_lines = max(max_lines, line_count)
        base = 26 if sheet_name in {"과제후보_등록형식", "카테고리별_발췌"} else 22
        cap = 200 if sheet_name in {"과제후보_등록형식", "카테고리별_발췌"} else 90
        ws.row_dimensions[row_idx].height = min(max(base, max_lines * 15), cap)


def _events_to_df(events: List[MaintenanceEvent]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append({
            "equipment_no": e.equipment_no,
            "equipment_name": e.equipment_name,
            "year": e.report_year,
            "source_files": ", ".join(e.source_files),
            "finding_location": e.finding_location,
            "finding_damage": e.finding_damage,
            "finding_measurement": e.finding_measurement,
            "action_type": e.action_type,
            "action_detail": e.action_detail,
            "recommendation": e.recommendation,
            "evidence_summary": e.evidence_summary,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _cases_to_df(cases: List[RepeatCase]) -> pd.DataFrame:
    rows = []
    for c in cases:
        rows.append({
            "equipment_no": c.equipment_no,
            "설비명": c.equipment_name,
            "repeat_key": c.repeat_key,
            "action_cluster": c.action_cluster,
            "location_cluster": c.location_cluster,
            "damage_cluster": c.damage_cluster,
            "years": ", ".join(map(str, c.years)),
            "발생년도수": len(c.years),
            "repeat_reason": c.repeat_reason,
            "confidence": f"{c.confidence:.2f}",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def export_excel(task_df: pd.DataFrame, repeat_cases: List[RepeatCase], all_events: List[MaintenanceEvent], output_path: str | Path) -> str:
    output_path = str(output_path)
    events_df = _events_to_df(all_events)
    cases_df = _cases_to_df(repeat_cases)
    category_df = build_category_extract_dataframe(task_df) if task_df is not None else pd.DataFrame(columns=TASK_COLUMNS)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        (task_df if task_df is not None else pd.DataFrame(columns=TASK_COLUMNS)).to_excel(writer, index=False, sheet_name="과제후보_등록형식")
        category_df.to_excel(writer, index=False, sheet_name="카테고리별_발췌")
        if not cases_df.empty:
            cases_df.to_excel(writer, index=False, sheet_name="반복정비_Case")
        if not events_df.empty:
            events_df.to_excel(writer, index=False, sheet_name="연도별_정비이벤트")
        for sheet_name in writer.book.sheetnames:
            _apply_sheet_style(writer.book[sheet_name], sheet_name)
    return output_path
