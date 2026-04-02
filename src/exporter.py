from __future__ import annotations

from pathlib import Path
import pandas as pd

from .task_builder import build_category_extract_dataframe



def export_excel(task_df, repeat_cases, all_events, output_path):
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
    category_df = build_category_extract_dataframe(task_df if task_df is not None else pd.DataFrame())

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        (task_df if task_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="과제후보_등록형식")
        category_df.to_excel(writer, index=False, sheet_name="카테고리별_발췌")
        cases_df.to_excel(writer, index=False, sheet_name="반복정비_Case")
        events_df.to_excel(writer, index=False, sheet_name="연도별_정비이벤트")
