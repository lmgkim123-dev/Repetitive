from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    import pdfplumber
except Exception:
    pdfplumber = None


# -----------------------------
# 공통 정규식 / 키워드
# -----------------------------
YEAR_RE = re.compile(r"(20\d{2})")

# 설비번호 예시:
# 02E-105A / 02 C-101 / 21-P-302-B4-24" / 02PSV-3007C / 52C-203A
EQUIP_PATTERNS = [
    re.compile(r"\b\d{2,3}\s?[A-Z]{1,4}-\d{3,4}[A-Z]?\b", re.I),                  # 02E-105A, 52C-203A
    re.compile(r"\b\d{2,3}\s?[A-Z]{1,4}\d{3,4}[A-Z]?\b", re.I),                   # 02PSV3007C 유사형
    re.compile(r"\b\d{2,3}[A-Z]{1,4}-\d{3,4}[A-Z]?(?:-[A-Z0-9]+)*-?\d{0,2}\"?\b", re.I),  # 21-P-302-B4-24"
    re.compile(r"\b\d{2,3}[A-Z]{1,4}-\d{3,4}(?:[A-Z0-9-]+)?\b", re.I),            # 확장형
]

DAMAGE_KEYWORDS = {
    "corrosion": ["corrosion", "부식", "pitting", "pit", "general corrosion"],
    "cracking": ["crack", "균열", "linear indication"],
    "leak": ["leak", "누설", "천공"],
    "thinning": ["thinning", "두께감소", "감육", "remaining thickness"],
    "plugging": ["plugging", "plug", "막힘", "coke plugging", "unplugging"],
    "coating": ["paint", "coating", "도장", "touch-up"],
    "fouling": ["fouling", "sludge", "scale", "mud"],
}

ACTION_KEYWORDS = {
    "replace": ["교체", "replace", "신규", "신규 제작"],
    "weld_repair": ["용접보수", "재용접", "용접", "육성용접", "overlay", "grinding"],
    "inspect": ["검사", "점검", "pt", "mt", "ut", "iris", "ect", "power brush"],
    "plug": ["plugging", "plug", "재확관"],
    "temporary": ["임시조치", "box-up", "compound sealing", "clamp"],
    "paint_repair": ["도장보수", "touch-up", "coating repair", "paint repair"],
    "cleaning": ["cleaning", "세척", "청소", "sand blasting", "power brushing"],
}

LOCATION_HINTS = {
    "top_head": ["top head", "상부 head", "tophead"],
    "shell_bottom": ["shell 하부", "shell bottom", "하부", "bottom shell"],
    "nozzle": ["nozzle", "노즐"],
    "internal": ["tray", "packing", "distributor", "collector", "baffle", "entry horn", "wear pad", "internal"],
    "bundle_tube": ["bundle", "tube", "tube sheet", "iris", "ect", "retube"],
    "line_section": ["line", "배관", "culvert", "box-up", "spool", "injection point"],
    "coating_area": ["paint", "coating", "도장", "touch-up"],
}


# -----------------------------
# 공통 유틸
# -----------------------------
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_year_from_text(text: str) -> str:
    m = YEAR_RE.search(str(text))
    return m.group(1) if m else ""


def extract_year_from_filename(path: Path) -> str:
    return extract_year_from_text(path.name)


def normalize_equipment_no(raw: str) -> str:
    if not raw:
        return ""
    text = str(raw).upper().strip()
    text = text.replace("”", "\"").replace("“", "\"")
    text = re.sub(r"\s+", "", text)

    # 02PSV3007C -> 02PSV-3007C 형태 보정
    m = re.match(r"^(\d{2,3})([A-Z]{2,5})(\d{3,4}[A-Z]?)$", text)
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

    # 02E105A -> 02E-105A 형태 보정
    m = re.match(r"^(\d{2,3})([A-Z]{1,4})(\d{3,4}[A-Z]?)$", text)
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

    # 02 E-105A / 02E - 105A 류 정리
    text = re.sub(r"^(\d{2,3})([A-Z]{1,4})-(\d{3,4}[A-Z]?)$", r"\1\2-\3", text)

    return text


def extract_equipment_nos(text: str) -> List[str]:
    found = []
    s = str(text)

    for pat in EQUIP_PATTERNS:
        for m in pat.findall(s):
            eq = normalize_equipment_no(m)
            if eq and eq not in found:
                found.append(eq)

    return found


def extract_damage_tags(text: str) -> str:
    t = str(text).lower()
    tags = []
    for tag, keywords in DAMAGE_KEYWORDS.items():
        if any(k.lower() in t for k in keywords):
            tags.append(tag)
    return ", ".join(sorted(set(tags)))


def extract_action_tags(text: str) -> str:
    t = str(text).lower()
    tags = []
    for tag, keywords in ACTION_KEYWORDS.items():
        if any(k.lower() in t for k in keywords):
            tags.append(tag)
    return ", ".join(sorted(set(tags)))


def extract_location_hint(text: str) -> str:
    t = str(text).lower()
    hits = []
    for loc, keywords in LOCATION_HINTS.items():
        if any(k.lower() in t for k in keywords):
            hits.append(loc)
    return ", ".join(hits)


def clean_equipment_name(text: str, equipment_no: str = "") -> str:
    value = normalize_whitespace(text)
    if not value:
        return ""
    if equipment_no:
        value = re.sub(re.escape(equipment_no), " ", value, flags=re.I)
        value = re.sub(re.escape(equipment_no.replace("-", "")), " ", value, flags=re.I)
    value = re.sub(r"^[\-–—:;,/\[\]\(\)]+", "", value).strip()
    value = re.split(
        r"\b(shell|tube|corrosion|leak|inspection|finding|result|comment|recommend|action|remark|issue|status|조치|검사|비고|손상|부식|감육|누설|균열|권고)\b",
        value,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    value = re.sub(r"\s{2,}", " ", value).strip(" -:;/")
    if len(value) < 2:
        return ""
    return value[:80]


def looks_like_equipment_name(text: str) -> bool:
    value = normalize_whitespace(text)
    if len(value) < 2:
        return False
    if extract_equipment_nos(value):
        return False
    if re.fullmatch(r"[\d\W_]+", value):
        return False
    if len(re.findall(r"[A-Za-z가-힣]", value)) < 2:
        return False
    bad_patterns = [
        r"^(result|comment|recommend|remark|action|inspection|finding)$",
        r"^(검사|조치|비고|내용|결과)$",
        r"^(page|sheet)\s*\d+$",
        r"line$|nozzle$|tray$|internal$|pipe$|shell$|head$|bed$",
        r"o/h\s*line|over\s*flash\s*line|bottom\s*line|line\s*no\.?|nozzle\s*no\.?",
        r"교체|제작|실시|검사|조치|보수|손상|부식|균열|감육|두께|phenolic epoxy|sand blasting",
    ]
    return not any(re.search(p, value, flags=re.I) for p in bad_patterns)


def extract_equipment_name_near_equipment(text: str, equipment_no: str = "") -> str:
    raw = normalize_whitespace(text)
    eq = normalize_equipment_no(equipment_no)
    if not raw or not eq:
        return ""

    variants = [eq, eq.replace("-", ""), eq.replace("-", " - ")]
    scored: List[tuple[int, str]] = []
    seen = set()

    for variant in variants:
        if not variant:
            continue

        strong_patterns = [
            (30, re.compile(rf"{re.escape(variant)}\s*\(([^)]+)\)", re.I)),
            (25, re.compile(rf"([A-Za-z][A-Za-z0-9/&\-\s]{{2,40}})\s*\(\s*{re.escape(variant)}\s*\)", re.I)),
            (18, re.compile(rf"{re.escape(variant)}\s+([A-Za-z][A-Za-z0-9/&\-\s]{{2,40}}?)(?=\s{{2,}}|[|;,:]|$)", re.I)),
        ]
        for bonus, pattern in strong_patterns:
            for match in pattern.finditer(raw):
                cand = clean_equipment_name(match.group(1), eq)
                if looks_like_equipment_name(cand) and cand not in seen:
                    scored.append((bonus + len(cand), cand))
                    seen.add(cand)

        pattern = re.compile(re.escape(variant), re.I)
        for match in pattern.finditer(raw):
            after = raw[match.end():].strip()
            if after:
                for piece in [after, re.split(r"\s{2,}|[|;/]", after)[0]]:
                    cand = clean_equipment_name(piece, eq)
                    if looks_like_equipment_name(cand) and cand not in seen:
                        scored.append((10 + len(cand), cand))
                        seen.add(cand)

            before = raw[:match.start()].strip()
            if before:
                piece = re.split(r"[|;/,:\[\]\(\)]", before)[-1]
                cand = clean_equipment_name(piece, eq)
                if looks_like_equipment_name(cand) and cand not in seen:
                    scored.append((8 + len(cand), cand))
                    seen.add(cand)

    if not scored:
        return ""
    return sorted(scored, key=lambda x: (-x[0], len(x[1]), x[1]))[0][1]


def build_pdf_equipment_name_map(text: str) -> Dict[str, str]:
    lines = [normalize_whitespace(line) for line in str(text).splitlines() if normalize_whitespace(line)]
    candidates: Dict[str, List[tuple[int, str]]] = {}

    for idx, line in enumerate(lines):
        eqs = extract_equipment_nos(line)
        if not eqs:
            continue

        for eq in eqs:
            direct_name = extract_equipment_name_near_equipment(line, eq)
            if direct_name:
                bonus = 20 if re.search(rf"{re.escape(eq)}\s*\(|\(\s*{re.escape(eq)}\s*\)", line, flags=re.I) else 10
                candidates.setdefault(eq, []).append((bonus, direct_name))
                continue

            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                if next_line and not extract_equipment_nos(next_line):
                    next_name = clean_equipment_name(next_line, eq)
                    if looks_like_equipment_name(next_name):
                        candidates.setdefault(eq, []).append((6, next_name))

    result: Dict[str, str] = {}
    for eq, names in candidates.items():
        scores: Dict[str, int] = {}
        for bonus, name in names:
            scores[name] = scores.get(name, 0) + bonus
        result[eq] = sorted(scores.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[0][0]
    return result


def build_event(
    equipment_no: str,
    source_file: str,
    source_type: str,
    year: str,
    sentence: str,
    equipment_name: str = "",
    design_thk=None,
    min_allow_thk=None,
    measured_min_thk=None,
    thinning_pct=None,
) -> Dict:
    sent = normalize_whitespace(sentence)
    eq = normalize_equipment_no(equipment_no)
    return {
        "equipment_no": eq,
        "equipment_name": clean_equipment_name(equipment_name, eq),
        "source_file": source_file,
        "source_type": source_type,
        "year": year,
        "sentence": sent,
        "damage_tags": extract_damage_tags(sent),
        "action_tags": extract_action_tags(sent),
        "location_hint": extract_location_hint(sent),
        "design_thk": design_thk,
        "min_allow_thk": min_allow_thk,
        "measured_min_thk": measured_min_thk,
        "thinning_pct": thinning_pct,
    }


def split_sentences(text: str) -> List[str]:
    raw_lines = re.split(r"[\n\r]+|(?<=[\.\?\!])\s+", str(text))
    lines = []
    for x in raw_lines:
        x = normalize_whitespace(x)
        if len(x) >= 8:
            lines.append(x)
    return lines


def choose_excel_engine(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        return "xlrd"
    if suffix == ".xlsb":
        return "pyxlsb"
    return None


def score_header_row(values: List[object]) -> int:
    texts = [normalize_whitespace(v).lower() for v in values if normalize_whitespace(v)]
    nonempty = len(texts)
    if nonempty < 2:
        return -1

    score = 0
    joined = " | ".join(texts)

    strong_keywords = [
        "equipment", "equip", "설비", "장치", "기기", "tag no", "tagno",
        "검사", "inspection", "조치", "result", "comment", "recommend", "권고",
        "두께", "thickness", "allow", "measured", "요청", "내용",
    ]
    for kw in strong_keywords:
        if kw in joined:
            score += 2

    if any("설비" in t or "equipment" in t or "equip" in t for t in texts):
        score += 4
    if any("두께" in t or "thickness" in t for t in texts):
        score += 2
    if any("조치" in t or "result" in t or "comment" in t or "recommend" in t for t in texts):
        score += 2
    if nonempty >= 4:
        score += 1

    return score


def read_excel_sheet_robust(path: Path, sheet: str) -> pd.DataFrame:
    engine = choose_excel_engine(path)
    raw = pd.read_excel(path, sheet_name=sheet, dtype=object, header=None, engine=engine)
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        return raw

    max_scan = min(len(raw), 10)
    best_idx = 0
    best_score = -1
    for idx in range(max_scan):
        score = score_header_row(raw.iloc[idx].tolist())
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score >= 4:
        header_row = raw.iloc[best_idx].tolist()
        data = raw.iloc[best_idx + 1 :].copy()
    else:
        header_row = raw.iloc[0].tolist()
        data = raw.iloc[1:].copy()

    columns = []
    used = {}
    for i, value in enumerate(header_row):
        col = normalize_whitespace(value) or f"col_{i}"
        used[col] = used.get(col, 0) + 1
        if used[col] > 1:
            col = f"{col}_{used[col]}"
        columns.append(col)

    data.columns = columns
    data = data.dropna(axis=0, how="all").reset_index(drop=True)
    return data


def is_meaningful_excel_row(base_text: str, design_thk, min_allow_thk, measured_min_thk, thinning_pct) -> bool:
    if base_text and (extract_damage_tags(base_text) or extract_action_tags(base_text) or extract_location_hint(base_text)):
        return True

    numeric_values = [design_thk, min_allow_thk, measured_min_thk, thinning_pct]
    return any(pd.notna(v) for v in numeric_values)


def find_equipment_name_columns(columns: List[str], equip_col: str | None = None) -> List[str]:
    name_cols = []
    for col in columns:
        if col == equip_col:
            continue
        c = str(col).lower()
        if any(k in c for k in ["설비명", "장치명", "기기명", "equipment name", "equip name", "tag description", "service", "name"]):
            if not any(k in c for k in ["result", "comment", "recommend", "remark", "조치", "검사", "비고", "내용"]):
                name_cols.append(col)
    return name_cols[:3]


def extract_equipment_name_from_row(row: pd.Series, name_cols: List[str]) -> str:
    parts = []
    for col in name_cols:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    name = " / ".join(dict.fromkeys(parts))
    return clean_equipment_name(name)


# -----------------------------
# PDF 추출
# -----------------------------
def extract_pdf(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    if pdfplumber is None:
        return pd.DataFrame([{
            "equipment_no": "",
            "source_file": path.name,
            "source_type": "PDF",
            "year": extract_year_from_filename(path),
            "sentence": "pdfplumber not installed",
            "equipment_name": "",
            "damage_tags": "",
            "action_tags": "",
            "location_hint": "",
            "design_thk": pd.NA,
            "min_allow_thk": pd.NA,
            "measured_min_thk": pd.NA,
            "thinning_pct": pd.NA,
        }])

    all_text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if txt:
                    all_text.append(txt)
    except Exception as exc:
        return pd.DataFrame([{
            "equipment_no": "",
            "source_file": path.name,
            "source_type": "PDF",
            "year": extract_year_from_filename(path),
            "sentence": f"PDF parse error: {exc}",
            "equipment_name": "",
            "damage_tags": "",
            "action_tags": "",
            "location_hint": "",
            "design_thk": pd.NA,
            "min_allow_thk": pd.NA,
            "measured_min_thk": pd.NA,
            "thinning_pct": pd.NA,
        }])

    full_text = "\n".join(all_text)
    year = extract_year_from_filename(path) or extract_year_from_text(full_text[:2000])
    pdf_name_map = build_pdf_equipment_name_map(full_text)

    current_equipment = ""
    current_equipment_name = ""
    for sent in split_sentences(full_text):
        eqs = extract_equipment_nos(sent)
        if eqs:
            current_equipment = eqs[0]
            current_equipment_name = pdf_name_map.get(current_equipment, "") or extract_equipment_name_near_equipment(sent, current_equipment)
            for eq in eqs:
                eq_name = extract_equipment_name_near_equipment(sent, eq) or pdf_name_map.get(eq, "") or current_equipment_name
                rows.append(build_event(eq, path.name, "PDF", year, sent, equipment_name=eq_name))
        else:
            if current_equipment and (
                extract_damage_tags(sent) or extract_action_tags(sent) or extract_location_hint(sent)
            ):
                rows.append(build_event(current_equipment, path.name, "PDF", year, sent, equipment_name=current_equipment_name or pdf_name_map.get(current_equipment, "")))

    return pd.DataFrame(rows)


# -----------------------------
# Excel 추출
# -----------------------------
def guess_year_from_dataframe(df: pd.DataFrame, filename: str) -> str:
    year = extract_year_from_text(filename)
    if year:
        return year

    sample_text = " ".join(
        df.astype(str).head(30).fillna("").values.flatten().tolist()
    )
    return extract_year_from_text(sample_text)


def find_equipment_column(df: pd.DataFrame) -> str | None:
    columns = df.columns.tolist()
    strong_candidates = []
    weak_candidates = []

    for col in columns:
        c = str(col).lower()
        if any(k in c for k in ["equipment", "equip", "설비", "장치", "기기", "tag"]):
            strong_candidates.append(col)
        elif any(k in c for k in ["번호", "no", "item"]):
            weak_candidates.append(col)

    if strong_candidates:
        return strong_candidates[0]

    best_col = None
    best_ratio = 0.0
    for col in columns:
        series = df[col].dropna().astype(str).head(80)
        if series.empty:
            continue
        hit = sum(1 for v in series if extract_equipment_nos(v))
        ratio = hit / max(len(series), 1)
        if hit >= 2 and ratio > best_ratio:
            best_col = col
            best_ratio = ratio

    if best_col and best_ratio >= 0.2:
        return best_col

    return weak_candidates[0] if weak_candidates else None


def find_text_columns(columns: List[str]) -> List[str]:
    out = []
    for col in columns:
        c = str(col).lower()
        if any(k in c for k in [
            "요청", "내용", "result", "검사", "조치", "recommend", "권고", "comment",
            "비고", "service", "detail", "수행여부", "description", "finding", "remark",
            "action", "inspection", "work", "status", "issue", "원인", "현상"
        ]):
            out.append(col)
    return out


def parse_numeric(x):
    try:
        if pd.isna(x):
            return pd.NA
        s = str(x).replace("%", "").replace(",", "").strip()
        return float(s)
    except Exception:
        return pd.NA


def extract_excel(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    try:
        xls = pd.ExcelFile(path)
        sheet_names = xls.sheet_names
    except Exception as exc:
        return pd.DataFrame([{
            "equipment_no": "",
            "source_file": path.name,
            "source_type": "EXCEL",
            "year": extract_year_from_filename(path),
            "sentence": f"Excel open error: {exc}",
            "equipment_name": "",
            "damage_tags": "",
            "action_tags": "",
            "location_hint": "",
            "design_thk": pd.NA,
            "min_allow_thk": pd.NA,
            "measured_min_thk": pd.NA,
            "thinning_pct": pd.NA,
        }])

    for sheet in sheet_names:
        try:
            df = read_excel_sheet_robust(path, sheet)
        except Exception:
            continue

        if df.empty:
            continue

        df.columns = [normalize_whitespace(c) for c in df.columns]
        year = guess_year_from_dataframe(df, path.name)

        equip_col = find_equipment_column(df)
        text_cols = find_text_columns(df.columns.tolist())
        if not text_cols:
            text_cols = [c for c in df.columns if c != equip_col][:8]
        name_cols = find_equipment_name_columns(df.columns.tolist(), equip_col)

        # 수명평가 자료용 컬럼 탐색
        design_col = next((c for c in df.columns if "설계두께" in str(c) or "design" in str(c).lower()), None)
        allow_col = next((c for c in df.columns if "최소허용" in str(c) or "allow" in str(c).lower()), None)
        measured_col = next((c for c in df.columns if "최소두께" in str(c) or "measured" in str(c).lower()), None)
        thinning_col = next((c for c in df.columns if "감육" in str(c) or "thinning" in str(c).lower()), None)

        last_explicit_equipment: List[str] = []
        last_equipment_name = ""

        for _, row in df.iterrows():
            row_text_parts = []
            for c in text_cols:
                val = row.get(c, "")
                if pd.notna(val) and str(val).strip():
                    row_text_parts.append(f"{c}: {val}")

            if not row_text_parts and equip_col is None:
                continue

            base_text = " / ".join(row_text_parts)
            design_thk = parse_numeric(row.get(design_col, pd.NA)) if design_col else pd.NA
            min_allow_thk = parse_numeric(row.get(allow_col, pd.NA)) if allow_col else pd.NA
            measured_min_thk = parse_numeric(row.get(measured_col, pd.NA)) if measured_col else pd.NA
            thinning_pct = parse_numeric(row.get(thinning_col, pd.NA)) if thinning_col else pd.NA

            equip_candidates = []
            equipment_name = extract_equipment_name_from_row(row, name_cols)

            if equip_col and pd.notna(row.get(equip_col, pd.NA)):
                equip_candidates.extend(extract_equipment_nos(str(row.get(equip_col))))

            if base_text:
                equip_candidates.extend(extract_equipment_nos(base_text))

            equip_candidates = [normalize_equipment_no(x) for x in equip_candidates if normalize_equipment_no(x)]
            equip_candidates = list(dict.fromkeys(equip_candidates))
            if not equipment_name:
                for eq in equip_candidates:
                    equipment_name = extract_equipment_name_near_equipment(base_text, eq)
                    if equipment_name:
                        break

            if not equip_candidates and last_explicit_equipment and is_meaningful_excel_row(
                base_text, design_thk, min_allow_thk, measured_min_thk, thinning_pct
            ):
                equip_candidates = last_explicit_equipment.copy()
                if not equipment_name:
                    equipment_name = last_equipment_name

            if not equip_candidates:
                continue

            if not base_text:
                base_text = " / ".join([str(row.get(c, "")) for c in df.columns[:6] if pd.notna(row.get(c, pd.NA))])

            for eq in equip_candidates:
                rows.append(build_event(
                    eq,
                    path.name,
                    "EXCEL",
                    year,
                    base_text,
                    equipment_name=equipment_name,
                    design_thk=design_thk,
                    min_allow_thk=min_allow_thk,
                    measured_min_thk=measured_min_thk,
                    thinning_pct=thinning_pct,
                ))

            if equip_candidates:
                last_explicit_equipment = equip_candidates.copy()
                if equipment_name:
                    last_equipment_name = equipment_name

    return pd.DataFrame(rows)


# -----------------------------
# TXT / CSV
# -----------------------------
def extract_text_like(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = path.read_text(errors="ignore")

    year = extract_year_from_filename(path) or extract_year_from_text(text[:2000])

    current_equipment = ""
    current_equipment_name = ""
    for sent in split_sentences(text):
        eqs = extract_equipment_nos(sent)
        if eqs:
            current_equipment = eqs[0]
            current_equipment_name = extract_equipment_name_near_equipment(sent, current_equipment)
            for eq in eqs:
                eq_name = extract_equipment_name_near_equipment(sent, eq) or current_equipment_name
                rows.append(build_event(eq, path.name, "TEXT", year, sent, equipment_name=eq_name))
        else:
            if current_equipment and (
                extract_damage_tags(sent) or extract_action_tags(sent) or extract_location_hint(sent)
            ):
                rows.append(build_event(current_equipment, path.name, "TEXT", year, sent, equipment_name=current_equipment_name))

    return pd.DataFrame(rows)


# -----------------------------
# 메인 라우터
# -----------------------------
def extract_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(path)

    if suffix in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
        return extract_excel(path)

    if suffix in [".txt", ".csv"]:
        return extract_text_like(path)

    return pd.DataFrame()