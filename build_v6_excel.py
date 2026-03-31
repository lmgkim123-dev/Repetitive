"""v6 최종 엑셀 생성 스크립트"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.anchor_builder import (
    build_equipment_name_map,
    extract_names_from_text,
    normalize_equipment_no,
    _clean_candidate,
    _score_candidate,
)
from src.event_builder import build_events_for_equipment
from src.repeat_judger import judge_repeat_cases, merge_cases_per_equipment
from src.task_builder import build_equipment_summary_dataframe
from src.exporter import export_excel

SRC = Path("/home/user/downloads/repeat_result_user_review.xlsx")
OUT_DIR = Path("/mnt/user-data/outputs/repeat_task_v6")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "반복정비_조치요약_v6_개선판.xlsx"

detail_df = pd.read_excel(SRC, sheet_name="근거상세")

all_candidates = defaultdict(list)
for eq in detail_df["equipment_no"].dropna().unique():
    eq_norm = normalize_equipment_no(str(eq))
    if not eq_norm:
        continue
    sub = detail_df[detail_df["equipment_no"] == eq]
    if "equipment_name" in sub.columns:
        for name in sub["equipment_name"].dropna().astype(str).unique():
            name = name.strip()
            if not name:
                continue
            found = extract_names_from_text(eq_norm, name)
            all_candidates[eq_norm].extend(found)
            cleaned = _clean_candidate(name, eq_norm)
            if cleaned:
                all_candidates[eq_norm].append((cleaned, "fallback", _score_candidate(cleaned, "fallback")))
    if "sentence" in sub.columns:
        for sent in sub["sentence"].dropna().astype(str).unique():
            all_candidates[eq_norm].extend(extract_names_from_text(eq_norm, sent))

equipment_names = build_equipment_name_map(dict(all_candidates))

detail_df["equipment_no"] = detail_df["equipment_no"].apply(lambda x: normalize_equipment_no(str(x)) if pd.notna(x) else "")
detail_df = detail_df[detail_df["equipment_no"] != ""].copy()

all_events = []
equipment_events = {}
for eq_no, eq_group in detail_df.groupby("equipment_no"):
    eq_name = equipment_names.get(str(eq_no), "")
    events = build_events_for_equipment(str(eq_no), eq_name, eq_group)
    if events:
        all_events.extend(events)
        equipment_events[str(eq_no)] = events

repeat_cases = merge_cases_per_equipment(judge_repeat_cases(equipment_events, equipment_names))
overview_df = build_equipment_summary_dataframe(all_events)
export_excel(overview_df, repeat_cases, all_events, OUT)

print(f"OUTPUT: {OUT}")
print(f"요약 rows: {len(overview_df)}")
print(f"반복 케이스: {len(repeat_cases)}")
print(f"전체 이벤트: {len(all_events)}")

for eq in ["02C-1001", "02C-1002", "02C-1003"]:
    print(f"\n=== {eq} ===")
    sub = overview_df[overview_df["Equipment No"] == eq]
    if sub.empty:
        print("(no rows)")
    else:
        print(sub[["Equipment No", "설비명", "발생구분", "발생년도수", "발생년도", "발췌 Category", "TA 조치사항"]].to_string(index=False))
