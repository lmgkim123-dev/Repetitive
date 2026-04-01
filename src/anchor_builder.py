"""v6 anchor_builder – 설비번호-설비명 확정 모듈

핵심 원칙:
1) "설비번호 (설비명)" 패턴이 최우선
2) 부위/라인/부속/조치 문구는 설비명에서 금지
3) 모든 파일에서 수집 후, 빈도 × 패턴 가중치로 최종 선정
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

# ── 설비명 금지 패턴 ──
_FORBIDDEN_NAME_RE = re.compile(
    r"교체|설치|제작|실시|요망|검사|점검|조치|비고|내용|손상|부식|누설|균열|감육|두께|상태|표면|도장|"
    r"sandblasting|coating|phenolic|epoxy|"
    r"nozzle\s*no\.?|nozzle|tray|internal|packing|shell|mesh|support|bar|steam|"
    r"line$|^line\b|o/h\s+line|over\s*flash|bottom\s+line|"
    r"pipe|punch\s*plate|wear\s*pad|lining|baffle|bed$|bubble\s*cap|distributor|collector|"
    r"downcomer|riser|plug|gasket|bolt|flange|valve|"
    r"용접|보수|보온|제거|철거|세척|cleaning|결함|시험|mt|pt|ut|rt|"
    r"설계|재질|두께|허용|부식여유",
    re.I,
)

# ── 좋은 설비명 패턴 ──
_GOOD_NAME_RE = re.compile(
    r"column|drum|tower|stripper|receiver|cooler|filter|accumulator|separator|"
    r"reflux|surge|absorber|flare|cracker|stabilizer|"
    r"exchanger|condenser|reboiler|scrubber|knock\s*out|k/o|"
    r"heater|furnace|compressor|pump|blower|ejector|"
    r"reactor|regenerator|deaerator|desalter|mixer",
    re.I,
)

# ── 설비번호 패턴 ──
_EQUIP_RE = re.compile(r"\b\d{2,3}\s?[A-Z]{1,4}-?\d{3,4}[A-Z]?\b", re.I)


def normalize_equipment_no(raw: str) -> str:
    if not raw:
        return ""
    text = str(raw).upper().strip()
    text = re.sub(r"\s+", "", text)
    m = re.match(r"^(\d{2,3})([A-Z]{1,5})(\d{3,4}[A-Z]?)$", text)
    if m:
        return f"{m.group(1)}{m.group(2)}-{m.group(3)}"
    return text


def _clean_candidate(text: str, eq_no: str = "") -> str:
    v = re.sub(r"\s+", " ", str(text)).strip()
    if not v:
        return ""
    if eq_no:
        for variant in [eq_no, eq_no.replace("-", ""), eq_no.replace("-", " ")]:
            v = re.sub(re.escape(variant), " ", v, flags=re.I)
    v = re.sub(r"^[\-–—:;,./\[\]\(\)]+", "", v).strip()
    v = re.sub(r"[\)\]]+$", "", v).strip()
    v = re.split(r"\b(?:inspection|result|comment|recommend|remark|action|조치|검사|비고|내용)\b", v, maxsplit=1, flags=re.I)[0].strip()
    v = re.sub(r"\s{2,}", " ", v).strip(" -:;/")
    return v[:60] if len(v) >= 2 else ""


def _score_candidate(name: str, pattern: str) -> float:
    """설비명 후보 점수: 패턴 가중치 + 내용 가중치"""
    if not name:
        return -999

    # ── 패턴 가중치 ──
    pattern_weight = {
        "paren_exact": 50,     # 02C-101 (Crude Column) → 최고
        "before_paren": 45,    # Crude Column (02C-101)
        "header_match": 35,    # 표 제목에서 설비명
        "table_match": 30,     # 표 셀에서 인접
        "after_code": 20,      # 02C-101 Crude Column 형태
        "next_line": 12,       # 다음 줄
        "fallback": 5,         # 기타
    }.get(pattern, 5)

    score = pattern_weight

    # ── 내용 가중치 ──
    if _GOOD_NAME_RE.search(name):
        score += 25
    if _FORBIDDEN_NAME_RE.search(name):
        score -= 40
    if len(name.split()) > 5:
        score -= 10
    if len(name) < 3:
        score -= 15
    if re.fullmatch(r"[A-Z0-9\-\s/&]+", name, re.I) and len(name) > 3:
        score += 5
    if re.search(r"^\d", name):
        score -= 10

    return score


def extract_names_from_text(equipment_no: str, text: str, source_file: str = "") -> List[Tuple[str, str, float]]:
    """텍스트에서 설비명 후보를 추출한다.

    Returns: [(candidate_name, pattern, score), ...]
    """
    eq = normalize_equipment_no(equipment_no)
    if not eq or not text:
        return []

    results: List[Tuple[str, str, float]] = []
    seen = set()

    def _add(name: str, pattern: str):
        cleaned = _clean_candidate(name, eq)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            s = _score_candidate(cleaned, pattern)
            results.append((cleaned, pattern, s))

    # ── 패턴 1: 02C-101 (Crude Column) ──
    for m in re.finditer(rf"{re.escape(eq)}\s*\(([^)]+)\)", text, re.I):
        _add(m.group(1), "paren_exact")

    # 공백 없는 변형: 02C101 (Crude Column)
    eq_nohyphen = eq.replace("-", "")
    for m in re.finditer(rf"{re.escape(eq_nohyphen)}\s*\(([^)]+)\)", text, re.I):
        _add(m.group(1), "paren_exact")

    # ── 패턴 2: Crude Column (02C-101) ──
    for m in re.finditer(rf"([A-Za-z][A-Za-z0-9/&\-\s]{{2,40}})\s*\(\s*{re.escape(eq)}\s*\)", text, re.I):
        _add(m.group(1), "before_paren")
    for m in re.finditer(rf"([A-Za-z][A-Za-z0-9/&\-\s]{{2,40}})\s*\(\s*{re.escape(eq_nohyphen)}\s*\)", text, re.I):
        _add(m.group(1), "before_paren")

    # ── 패턴 3: 02C-101 Crude Column (공백 후 이어지는 이름) ──
    for m in re.finditer(rf"{re.escape(eq)}\s+([A-Za-z][A-Za-z0-9/&\-\s]{{2,40}}?)(?=\s{{2,}}|[|;,:]|\n|$)", text, re.I):
        _add(m.group(1), "after_code")

    return results


def build_equipment_name_map(
    all_candidates: Dict[str, List[Tuple[str, str, float]]],
) -> Dict[str, str]:
    """모든 파일에서 수집한 후보를 종합하여 설비별 최종 설비명 확정

    규칙:
    - 같은 이름이 여러 번 나오면 빈도 보너스
    - 최종 score = sum(individual scores) + frequency_bonus
    - score < 0 이면 설비명 없음
    """
    result: Dict[str, str] = {}

    for eq_no, candidates in all_candidates.items():
        if not candidates:
            continue

        name_scores: Dict[str, float] = defaultdict(float)
        name_counts: Dict[str, int] = defaultdict(int)

        for name, _pattern, score in candidates:
            name_scores[name] += score
            name_counts[name] += 1

        # 빈도 보너스
        for name in name_scores:
            name_scores[name] += name_counts[name] * 5

        ranked = sorted(name_scores.items(), key=lambda x: (-x[1], len(x[0])))
        best_name, best_score = ranked[0]
        if best_score > 0:
            result[eq_no] = best_name

    return result


def build_name_map_from_lines(lines: List[str], source_file: str = "") -> Dict[str, List[Tuple[str, str, float]]]:
    """PDF/텍스트 줄 목록에서 설비별 이름 후보 수집"""
    all_candidates: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

    for idx, line in enumerate(lines):
        line_clean = re.sub(r"\s+", " ", line).strip()
        if not line_clean:
            continue

        eqs = [normalize_equipment_no(m) for m in _EQUIP_RE.findall(line_clean)]
        eqs = [e for e in eqs if e]

        if not eqs:
            continue

        for eq in eqs:
            found = extract_names_from_text(eq, line_clean, source_file)
            all_candidates[eq].extend(found)

            # 다음 줄 체크
            if not found and idx + 1 < len(lines):
                next_line = re.sub(r"\s+", " ", lines[idx + 1]).strip()
                if next_line and not _EQUIP_RE.search(next_line):
                    cand = _clean_candidate(next_line, eq)
                    if cand and not _FORBIDDEN_NAME_RE.search(cand):
                        s = _score_candidate(cand, "next_line")
                        all_candidates[eq].append((cand, "next_line", s))

    return dict(all_candidates)
