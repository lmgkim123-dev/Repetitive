from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

from .anchor_builder import normalize_equipment_no

_GENERATED_SHEETS = {"과제후보_등록형식", "카테고리별_발췌", "연도별_정비이벤트", "반복정비_Case", "summary", "반복배관후보", "occurrences", "excluded_review"}
_LIST_SHEET_HINTS = {"목록", "list", "sheet1"}



def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        for low, orig in lower.items():
            if cand.lower() in low:
                return orig
    return None



def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    s = str(v).strip()
    return not s or s.lower() in {"nan", "n/a", "none", "null"}



def _clean_text(v) -> str:
    s = str(v or "")
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()



def _extract_year(value) -> int | None:
    if _is_missing(value):
        return None
    try:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.notna(dt):
            return int(dt.year)
    except Exception:
        pass
    m = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(m.group(1)) if m else None



def _split_field_text(text: str, section: str) -> list[str]:
    t = _clean_text(text)
    if not t:
        return []
    t = re.sub(r"\*\s*초기검사", "\n초기검사", t, flags=re.I)
    t = re.sub(r"\*\s*상세검사", "\n상세검사", t, flags=re.I)
    t = re.sub(r"\*\s*보수\s*/\s*개선\s*내용", "\n보수/개선 내용", t, flags=re.I)
    t = re.sub(r"\*\s*보수\s*개선\s*내용", "\n보수/개선 내용", t, flags=re.I)
    t = re.sub(r"(?m)^\s*(\d+[.)])\s*", r"\n\1 ", t)
    t = re.sub(r"(?m)^\s*[-•]\s*", "\n- ", t)
    parts = [p.strip() for p in re.split(r"\n+", t) if p and p.strip()]
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 2:
            continue
        if section == "recommendation" and re.fullmatch(r"(?:\d+[.)]?|차기\s*Recommendation|Recommendation|N/A|NA)", p, re.I):
            continue
        out.append(p)
    return out or [t]



def _records_from_list_sheet(df: pd.DataFrame, source_file: str, sheet_name: str, year_range: tuple[int,int] | None = None) -> list[dict]:
    eq_no_col = _pick_column(df, ["설비번호", "equipment_no", "equipment no"])
    eq_name_col = _pick_column(df, ["설비명", "equipment_name", "equipment name"])
    date_col = _pick_column(df, ["검사일", "inspection_date", "date"])
    detail_col = _pick_column(df, ["상세내용", "상세내역", "detail", "details"])
    rec_col = _pick_column(df, ["차기고려사항", "추후권고사항", "recommendation", "recommendations"])
    if not eq_no_col or not eq_name_col or not date_col:
        return []

    rows: list[dict] = []
    for _, r in df.iterrows():
        raw_eq = r.get(eq_no_col)
        eq_no = normalize_equipment_no(raw_eq)
        eq_name = _clean_text(r.get(eq_name_col))
        year = _extract_year(r.get(date_col))
        if not eq_no or _is_missing(eq_name) or not year:
            continue
        if year_range and not (year_range[0] <= year <= year_range[1]):
            continue
        base = {
            "equipment_no": eq_no,
            "equipment_name": eq_name,
            "year": year,
            "inspection_date": r.get(date_col),
            "source_file": source_file,
            "source_sheet": sheet_name,
        }
        detail = _clean_text(r.get(detail_col)) if detail_col else ""
        rec = _clean_text(r.get(rec_col)) if rec_col else ""
        if detail and detail.lower() not in {"n/a", "na", "none"}:
            for part in _split_field_text(detail, "detail"):
                rows.append({**base, "sentence": part, "section": "detail", "action_tags": "", "damage_tags": ""})
        if rec and rec.lower() not in {"n/a", "na", "none"}:
            for part in _split_field_text(rec, "recommendation"):
                rows.append({**base, "sentence": part, "section": "recommendation", "action_tags": "", "damage_tags": ""})
    return rows



def extract_any(path, year_range: tuple[int,int] | None = None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv"}:
        return pd.DataFrame()
    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
            records = _records_from_list_sheet(df, path.name, "csv", year_range=year_range)
            return pd.DataFrame(records)
        excel = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame()

    all_rows: list[dict] = []
    for sheet in excel.sheet_names:
        if sheet in _GENERATED_SHEETS:
            continue
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if sheet in _LIST_SHEET_HINTS or any(str(c).strip() in {"설비번호", "설비명", "검사일", "상세내용", "차기고려사항"} for c in df.columns):
            all_rows.extend(_records_from_list_sheet(df, path.name, sheet, year_range=year_range))

    return pd.DataFrame(all_rows)
