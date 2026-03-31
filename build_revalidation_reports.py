import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.anchor_builder import normalize_equipment_no
from src.event_builder import (
    _parse_tag_list,
    _normalize_sentence,
    _is_verified_bundle_replacement_sentence,
    _is_verified_nozzle_replacement_sentence,
    _is_verified_internal_replacement_sentence,
    _is_verified_assembly_replacement_sentence,
)

SRC = Path("/home/user/downloads/repeat_result_user_review.xlsx")
OUT_DIR = Path("/mnt/user-data/outputs/repeat_task_v6")
SUMMARY_XLSX = OUT_DIR / "반복정비_조치요약_v6_개선판.xlsx"
CATEGORY_OUT = OUT_DIR / "반복정비_카테고리별_누락0건_재검증리포트.xlsx"
BUNDLE_OUT = OUT_DIR / "반복정비_bundle교체_재검증리포트.xlsx"

CATEGORY_CONFIG = {
    "Nozzle 교체": {
        "category": "Nozzle 교체",
        "detector": _is_verified_nozzle_replacement_sentence,
        "anchors": [r"nozzle", r"노즐", r"\bnzl\b"],
        "summary_category": "Nozzle 교체",
    },
    "단순 내부 구성품 교체": {
        "category": "단순 내부 구성품 교체",
        "detector": _is_verified_internal_replacement_sentence,
        "anchors": [
            r"packing", r"tray", r"mesh", r"screen", r"clip", r"baffle", r"demister",
            r"internal", r"distributor", r"collector", r"entry\s*horn", r"riser\s*pipe\s*hat",
            r"punch\s*plate", r"support",
        ],
        "summary_category": "단순 내부 구성품 교체",
    },
    "Assembly 교체": {
        "category": "Assembly 교체",
        "detector": _is_verified_assembly_replacement_sentence,
        "anchors": [
            r"bundle", r"retube", r"new\s*vessel", r"신규\s*용기", r"shell\s*cover",
            r"floating\s*head", r"channel\b", r"backing\s*device", r"probe\s*assembly",
            r"\bassembly\b", r"\bassy\b", r"valve", r"elbow", r"duct", r"damper",
            r"vortex\s*breaker", r"radiant\s*tube\s*support", r"support",
        ],
        "summary_category": "Assembly 교체",
    },
}

BUNDLE_CONFIG = {
    "Bundle 교체": {
        "category": "Bundle 교체",
        "detector": _is_verified_bundle_replacement_sentence,
        "anchors": [r"bundle", r"retube", r"tube\s*bundle", r"번들"],
        "summary_category": "Assembly 교체",
    }
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_years(value) -> set[int]:
    return {int(y) for y in re.findall(r"20\d{2}", str(value or ""))}


def extract_anchor_hits(text: str, anchor_patterns: list[str]) -> list[str]:
    hits = []
    t = norm(text)
    for pat in anchor_patterns:
        if re.search(pat, t, re.I):
            hits.append(re.sub(r"\\b", "", pat))
    return list(dict.fromkeys(hits))


def build_candidates(detail_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for _, row in detail_df.iterrows():
        eq = normalize_equipment_no(str(row.get("equipment_no", "")))
        year = pd.to_numeric(row.get("year"), errors="coerce")
        if not eq or pd.isna(year):
            continue
        sentence = _normalize_sentence(row.get("sentence", ""))
        if not sentence:
            continue
        raw_action_tags = _parse_tag_list(row.get("action_tags"))
        for name, meta in config.items():
            detector = meta["detector"]
            if detector(sentence, raw_action_tags):
                rows.append({
                    "Equipment No": eq,
                    "year": int(year),
                    "target_category": meta["summary_category"],
                    "scope": name,
                    "sentence": sentence,
                    "source_file": norm(row.get("source_file", "")),
                    "anchors": ", ".join(extract_anchor_hits(sentence, meta["anchors"])),
                })
    if not rows:
        return pd.DataFrame(columns=["Equipment No", "year", "target_category", "scope", "sentence", "source_file", "anchors"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["Equipment No", "year", "target_category", "sentence"]).reset_index(drop=True)
    return df


def row_match(candidate_sentence: str, candidate_year: int, category: str, anchors: str, summary_rows: pd.DataFrame) -> tuple[bool, str]:
    cand = norm(candidate_sentence).lower()
    anchor_list = [a.strip() for a in str(anchors or "").split(",") if a.strip()]
    subset = summary_rows[summary_rows["발췌 Category"].astype(str) == str(category)].copy()
    if subset.empty:
        return False, "카테고리 행 없음"
    subset = subset[subset["발생년도"].apply(lambda x: candidate_year in parse_years(x))]
    if subset.empty:
        return False, "동일 연도 행 없음"
    for _, srow in subset.iterrows():
        ta = norm(srow.get("TA 조치사항", "")).lower()
        if not ta:
            continue
        if cand in ta:
            return True, "문장 직접 일치"
        if anchor_list and any(re.search(a, ta, re.I) for a in anchor_list):
            if category == "Nozzle 교체" and re.search(r"nozzle|노즐|\bnzl\b", ta, re.I):
                return True, "핵심 키워드 일치"
            if category == "단순 내부 구성품 교체" and re.search(r"packing|tray|mesh|screen|clip|baffle|demister|internal|distributor|collector|entry\s*horn|riser\s*pipe\s*hat|punch\s*plate|support", ta, re.I):
                return True, "핵심 키워드 일치"
            if category == "Assembly 교체" and re.search(r"bundle|retube|vessel|shell\s*cover|floating\s*head|channel\b|backing\s*device|probe\s*assembly|\bassembly\b|\bassy\b|valve|elbow|duct|damper|vortex\s*breaker|support", ta, re.I):
                return True, "핵심 키워드 일치"
    return False, "근거 문장 미매칭"


def verify_candidates(candidates: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.assign(status=[], match_note=[])
    out_rows = []
    for _, row in candidates.iterrows():
        eq = row["Equipment No"]
        subset = summary_df[summary_df["Equipment No"].astype(str) == eq]
        ok, note = row_match(row["sentence"], int(row["year"]), row["target_category"], row.get("anchors", ""), subset)
        new_row = row.to_dict()
        new_row["status"] = "OK" if ok else "MISS"
        new_row["match_note"] = note
        out_rows.append(new_row)
    return pd.DataFrame(out_rows)


def summary_table(verified_df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for scope in order:
        sub = verified_df[verified_df["scope"] == scope]
        rows.append({
            "scope": scope,
            "candidate_count": int(len(sub)),
            "verified_ok": int((sub["status"] == "OK").sum()),
            "remaining_miss": int((sub["status"] == "MISS").sum()),
        })
    return pd.DataFrame(rows)


def export_report(path: Path, verified_df: pd.DataFrame, summary_df: pd.DataFrame, scope_order: list[str]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        verified_df.to_excel(writer, index=False, sheet_name="detail")
        miss_df = verified_df[verified_df["status"] == "MISS"].copy()
        miss_df.to_excel(writer, index=False, sheet_name="miss")


def main() -> None:
    detail_df = pd.read_excel(SRC, sheet_name="근거상세")
    summary_df = pd.read_excel(SUMMARY_XLSX, sheet_name="과제후보_등록형식")
    summary_df["Equipment No"] = summary_df["Equipment No"].astype(str).map(lambda x: normalize_equipment_no(x) if x else "")

    category_candidates = build_candidates(detail_df, CATEGORY_CONFIG)
    category_verified = verify_candidates(category_candidates, summary_df)
    category_summary = summary_table(category_verified, list(CATEGORY_CONFIG.keys()))
    export_report(CATEGORY_OUT, category_verified, category_summary, list(CATEGORY_CONFIG.keys()))

    bundle_candidates = build_candidates(detail_df, BUNDLE_CONFIG)
    bundle_verified = verify_candidates(bundle_candidates, summary_df)
    bundle_summary = summary_table(bundle_verified, list(BUNDLE_CONFIG.keys()))
    export_report(BUNDLE_OUT, bundle_verified, bundle_summary, list(BUNDLE_CONFIG.keys()))

    print(f"CATEGORY_REPORT: {CATEGORY_OUT}")
    print(category_summary.to_string(index=False))
    print(f"BUNDLE_REPORT: {BUNDLE_OUT}")
    print(bundle_summary.to_string(index=False))


if __name__ == "__main__":
    main()
