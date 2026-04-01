"""v6 task_builder – 사용자 검토/발췌용 요약 시트 생성

개선 원칙
- 설비+카테고리 단위로 분리한다.
- 연도별 집계는 event 전체가 아니라 '실제 조치 문장(action item)' 단위로 한다.
- recommendation, 과거년도 회고문, 소부품 교체, 조건문은 조치 집계에서 제외/분리한다.
- nozzle / internal / assembly 경계를 보수적으로 분리한다.
- 하나의 긴 OCR 문장을 최대한 절 단위로 나눠 카테고리 오염을 줄인다.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, List

import pandas as pd

from .schemas import RepeatCase, TaskRow

CATEGORY_ORDER = [
    "단순 보수",
    "도장",
    "육성용접",
    "단순 내부 구성품 교체",
    "Nozzle 교체",
    "Assembly 교체",
]

TASK_COLUMNS = [
    "NO", "Equipment No", "설비명", "발생구분",
    "발생년도수", "발생년도",
    "발췌 Category", "반복부위",
    "TA 조치사항", "추후 권고사항",
    "제목", "상세 내용", "검토필요여부",
]

_INTERNAL_EXCLUDE_RE = re.compile(
    r"충진물|\bfiller\b|filter media|\bmedia\b replacement|adsorbent|desiccant|diesel\s*sand",
    re.I,
)
_SMALL_PART_EXCLUDE_RE = re.compile(
    r"\bbolt\b|\bnut\b|\bgasket\b|test\s*ring|collar\s*bolt|floating\s*head\s*bolt|\bf/h\s*bolt\b|keeper|pin|valve\s*wheel|stud|washer|anchor",
    re.I,
)
_INTERNAL_PART_RE = re.compile(
    r"mesh|screen|hold\s*-?down|holdown|clip|saddle\s*clip|grid\s*clip|tray|packing|bubble\s*cap|baffle|weir\s*plate|demister|internal|seal\s*pan|entry\s*horn|distributor|collector|tray\s*cap|riser\s*pipe\s*hat|punch\s*plate|corrosion\s*probe\s*assembly|probe\s*assembly|corrosion\s*probe|heater\s*tube\s*support|radiant\s*tube\s*support|tube\s*casting\s*support|casting\s*support|tube\s*support|vortex\s*breaker|strainer|\bvalve\b",
    re.I,
)
_NOZZLE_RE = re.compile(r"nozzle|노즐|\bnzl\b|\belbow\b", re.I)
_ASSEMBLY_OBJ_RE = re.compile(
    r"new\s*vessel|신규\s*용기|\bvessel\b|\bdrum\b|\bcolumn\b|\btower\b|\bbundle\b|retube|shell\s*cover|floating\s*head|\bchannel\b|top\s*head|bottom\s*head|\bassembly\b|\bassy\b|\bduct\b|\bdamper\b|expansion\s*joint|bellows|saddle(?!\s*clip)",
    re.I,
)
_ASSEMBLY_CONTEXT_RE = re.compile(r"신규\s*제작|사전\s*제작|제작\s*후\s*교체|new|fabricat|retube|retubing|전체\s*교체|assy|assembly|신품\s*교체|pre\s*-?fabricat|bellows|sleeve", re.I)
_COATING_RE = re.compile(r"phenolic\s*epoxy|coating(?!\s*상태)|paint(?!\s*상태)|도장(?!상태)|보수도장|재도장|touch-?up", re.I)
_BLAST_ONLY_RE = re.compile(r"sand\s*blasting|sandblasting", re.I)
_OVERLAY_RE = re.compile(r"육성\s*용접|육성\s*용접|육성용접|overlay|hardfacing|build[- ]?up\s*weld|erni-?cr-?3|er-?nicr-?3|용접보수|보수용접|erni-?cr-?3|er-?nicr-?3|용접보수|보수용접", re.I)
_WELD_REPAIR_RE = re.compile(
    r"seal\s*welding|seal-?weld|stitch\s*welding|weld\s*repair|repair\s*weld(?:ing)?|용접보수|보수용접|재\s*용접|재용접|결함\s*제거\s*후\s*용접|선형\s*결함\s*제거\s*후\s*용접|grinding\s*후\s*용접|용접\s*실시|육성\s*용접|용접\s*후\s*나사산|용접\s*후\s*.*가공|용접\s*후\s*.*탐상",
    re.I,
)
_SIMPLE_REPAIR_RE = re.compile(
    r"보수|repair|grinding|결함\s*제거|defect\s*remov|patch|보강|재시공|시공|임시조치|box-?up|compound\s*sealing|lining\s*repair|보수\s*완료|plug\b|plugging|unplugging|stop\s*hole|막음\s*작업|막음\s*용접|복원\s*후\s*조립",
    re.I,
)
_REPAIR_ACTION_RE = re.compile(
    r"grinding|결함\s*제거|defect\s*remov|patch-?up|patch|box-?up|compound\s*sealing|보수\s*완료|보강함|보강\s*실시|재시공|시공하였음|plug\b|plugging|unplugging|stop\s*hole|막음\s*작업|막음\s*용접|repair(ed)?|보수\s*작업\s*실시",
    re.I,
)
_REPLACE_RE = re.compile(r"교체|replace|replaced|신규\s*제작|신규\s*교체|제작\s*후\s*교체|fabricated?.*replace|retube|retubing|교체\s*설치함|교체\s*완료|교체\s*완료함|교체\s*완료하였음", re.I)
_ACTION_DONE_RE = re.compile(
    r"교체함|교체\s*설치함|교체\s*하였음|교체\s*완료함|교체\s*완료하였음|교체됨|신규\s*교체|신규\s*제작|제작\s*후\s*교체|설치함|설치\s*완료|실시함|실시하였음|실시\s*완료|작업함|작업\s*실시|보수\s*완료|보강함|보강\s*실시|용접\s*실시|blind\s*처리|by-?pass\s*시킴|재시공|시공하였음|repair(ed)?|replace(d)?|fabricated|coating\s*실시|도장\s*실시|완료함",
    re.I,
)
_AIR_COOLER_PLUG_SERVICE_RE = re.compile(r"air\s*cooler\s*plug|a/?c\s*plug|plug\b", re.I)
_AIR_COOLER_PLUG_NONREPAIR_RE = re.compile(r"분해|조립|해체|탈거|재조립|opening|closing|open|close", re.I)
_RECOMMEND_ONLY_RE = re.compile(r"요망|요함|필요|권고|차기\s*TA|다음\s*TA|recommend|검토|적용\s*검토|교체할\s*경우|실시하여야|실시\s*하여야|하여야\s*겠음|해야\s*겠음", re.I)
_TOOLING_RE = re.compile(r"유압\s*토크\s*렌치|토크\s*렌치|hydraulic\s*torque\s*wrench|torque\s*wrench", re.I)
_INSPECTION_ONLY_RE = re.compile(r"\bMT\b|\bPT\b|\bUT\b|검사|점검|확인|power\s*brush|power\s*brushing|세척|clean|청소|수압\s*테스트|RT/?수압\s*테스트|액체침투탐상|침투탐상|자분탐상", re.I)
_HISTORY_PAREN_RE = re.compile(r"\([^)]*(20\d{2})년[^)]*\)", re.I)
_HEADER_TRASH_RE = re.compile(
    r"검사사항\s*\(초기/상세\)|구분\s*Tube\s*Shell|초기\s*상태|상세\s*검사|표면\s*상태|도장\s*상태|^Line\s*no\.?$|^Nozzle\s*No\.?$",
    re.I,
)
_BULLET_SPLIT_RE = re.compile(
    r"(?:\n+|\\n+|\s+(?=\(\d+\))|\s+(?=-\s)|\s+(?=<[^>]+>)|(?<=[다음요권검])\s+(?=차기\s*TA|다음\s*TA|권고|요망|요함|필요|검토))"
)


def _has_explicit_done(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    return bool(re.search(
        r"교체함|교체\s*하였음|교체\s*설치함|교체\s*완료|교체\s*실시|설치함|설치\s*완료|실시함|완료함|보수\s*완료|용접보수|보수용접|재\s*용접|재용접|육성\s*용접|overlay|hardfacing|weld\s*repair|repair\s*weld|replace(d)?|retube|bundle\s*사전\s*신규\s*제작\s*및\s*교체|bundle\s*사전\s*제작\s*및\s*교체|신규\s*bundle\s*로\s*교체함|신규\s*bundle\s*제작\s*되어\s*교체|신규\s*용기\s*제작\s*후\s*교체|제작\s*후\s*교체\s*실시"
        , t, re.I))


def _is_negative_or_empty(text: str) -> bool:
    return bool(re.fullmatch(r"(?:없음|해당없음|none|n/?a)\.?", _normalize_text(text), re.I))


def _looks_like_recommendation(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    if _is_negative_or_empty(t):
        return False
    if re.search(r"차기\s*TA|다음\s*TA|향후|추후|권고|요망|요함|검토|예정|필요|실시하여야|실시\s*하여야|교체할\s*경우", t, re.I):
        if not _has_explicit_done(t):
            return True
        if re.search(r"(교체|보수|설치|제작).*(필요|요망|검토|예정)", t, re.I):
            return True
    return False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = _normalize_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _strip_other_year_history(text: str, report_year: int) -> str:
    t = str(text or "")

    def repl(m):
        years = re.findall(r"(20\d{2})", m.group(0))
        if years and any(int(y) != int(report_year) for y in years):
            return " "
        return m.group(0)

    t = _HISTORY_PAREN_RE.sub(repl, t)
    return _normalize_text(t)


def _clean_clause_text(text: str) -> str:
    t = str(text or "")
    t = re.sub(r"\\n", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = _HEADER_TRASH_RE.sub(" ", t)
    t = re.sub(r"\b[0-9]{2}[A-Z]-\d{3,4}[A-Z]?\b", " ", t)
    t = re.sub(r"\b(?:Crude Column|HK Stripper|Stabilizer|Desalter|Exchanger|Heater|Receiver|Dryer|Drum)\b", lambda m: m.group(0) if len(m.group(0).split()) > 2 else " ", t, flags=re.I)
    t = re.sub(r"\s+/\s+", ". ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" -/,:;")


def _split_clauses(text: str) -> List[str]:
    if not text:
        return []
    raw = _clean_clause_text(text)
    if not raw:
        return []
    first_pass = [p for p in _BULLET_SPLIT_RE.split(raw) if _normalize_text(p)]
    parts: List[str] = []
    for part in first_pass:
        sub_parts = re.split(r"(?<=[\.!?다함음요])\s+(?=(?:\(?\d+\)|[A-Z#0-9\"“]|Nozzle|Tray|Shell|Top|Bottom|내부|외부|차기\s*TA|다음\s*TA|권고|검토))", part)
        for sub in sub_parts:
            sub = _normalize_text(sub)
            if not sub:
                continue
            parts.append(sub)
    return [p for p in parts if _normalize_text(p)]


def _is_recommendation_only(text: str) -> bool:
    t = _normalize_text(text)
    return bool(t and _looks_like_recommendation(t))


def _is_historical_only(text: str, report_year: int) -> bool:
    years = {int(y) for y in re.findall(r"(20\d{2})", str(text or ""))}
    return bool(years and all(y != int(report_year) for y in years))


def categorize_text(text: str, action_type: str = "") -> List[str]:
    combined = _normalize_text(f"{action_type} {text}")
    if not combined:
        return []
    if _INTERNAL_EXCLUDE_RE.search(combined):
        return []
    if _is_negative_or_empty(combined):
        return []
    if _is_recommendation_only(combined):
        return []
    if (
        _AIR_COOLER_PLUG_SERVICE_RE.search(combined)
        and _AIR_COOLER_PLUG_NONREPAIR_RE.search(combined)
        and not re.search(r"누설|leak|교체|replace|보수|repair|용접|weld|균열|결함|손상|damage|막음|plugging|unplugging|재확관|확관|compound\s*sealing|box-?up", combined, re.I)
    ):
        return []
    if _HEADER_TRASH_RE.search(combined) and not (_REPLACE_RE.search(combined) or _SIMPLE_REPAIR_RE.search(combined) or _COATING_RE.search(combined) or _WELD_REPAIR_RE.search(combined)):
        return []
    if _SMALL_PART_EXCLUDE_RE.search(combined) and not _NOZZLE_RE.search(combined):
        return []
    if _BLAST_ONLY_RE.search(combined) and not _COATING_RE.search(combined):
        return []
    if _INSPECTION_ONLY_RE.search(combined) and not (_REPLACE_RE.search(combined) or _COATING_RE.search(combined) or _OVERLAY_RE.search(combined) or _SIMPLE_REPAIR_RE.search(combined)):
        return []

    has_replace = bool(_REPLACE_RE.search(combined)) or "replace" in action_type.lower()
    has_internal = bool(_INTERNAL_PART_RE.search(combined))
    has_nozzle = bool(_NOZZLE_RE.search(combined))
    has_assembly_obj = bool(_ASSEMBLY_OBJ_RE.search(combined))
    has_assembly_ctx = bool(_ASSEMBLY_CONTEXT_RE.search(combined))
    has_small_part = bool(_SMALL_PART_EXCLUDE_RE.search(combined))
    has_tooling = bool(_TOOLING_RE.search(combined))
    has_coating = bool(_COATING_RE.search(combined)) or "coating" in action_type.lower()
    has_overlay = bool(_OVERLAY_RE.search(combined))
    has_weld_repair = bool(_WELD_REPAIR_RE.search(combined)) or "weld_repair" in action_type.lower()
    has_simple = bool(_SIMPLE_REPAIR_RE.search(combined)) or any(x in action_type.lower() for x in ["temporary_fix", "plugging"])
    has_done = _has_explicit_done(combined)

    # 유압토크렌치/토크렌치 교체는 설비 본체 Assembly 교체가 아니라 단순 보수로 본다.
    if has_tooling:
        if has_replace or has_simple or has_done:
            return ["단순 보수"]
        return []

    # 도장은 교체/보수 문장이 섞이지 않은 경우에만 단독 분류
    if has_coating and not has_replace and not has_simple and not has_overlay:
        return ["도장"]

    # 교체는 nozzle > internal > assembly 우선순위로 단일 분류
    if has_replace:
        if _looks_like_recommendation(combined) and not has_done:
            return []
        if has_nozzle:
            if not has_done:
                return []
            if re.search(r"mint|ont|pitting|부식|감육|두께감소", combined, re.I) and not re.search(r"신규|제작|설치|제거\s*후|size-?up|기존", combined, re.I):
                return []
            return ["Nozzle 교체"]
        if has_internal and not has_small_part:
            if not has_done and not has_assembly_ctx:
                return []
            return ["단순 내부 구성품 교체"]
        if (has_assembly_obj or has_assembly_ctx) and not has_small_part and not has_internal and not has_nozzle:
            if not has_done and not has_assembly_ctx:
                return []
            if _looks_like_recommendation(combined) and not has_done:
                return []
            if re.search(r"설계두께|최소허용두께|상태|pitting|general corrosion|부식", combined, re.I) and not has_done:
                return []
            return ["Assembly 교체"]

    if has_coating:
        return ["도장"]
    if has_overlay or has_weld_repair:
        if not (has_done or re.search(r"육성용접|overlay|seal\s*welding|seal-?weld|재\s*용접|재용접|용접보수|보수용접|weld\s*repair|repair\s*weld", combined, re.I)):
            return []
        return ["육성용접"]
    if has_simple:
        if _RECOMMEND_ONLY_RE.search(combined) and not has_done:
            return []
        if not (_REPAIR_ACTION_RE.search(combined) or has_done):
            return []
        return ["단순 보수"]
    return []


def _unique_action_items(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for item in items:
        key = (item["year"], item["category"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _extract_action_items(event) -> List[dict]:
    items: List[dict] = []
    year = int(getattr(event, "report_year", 0) or 0)
    loc = _normalize_text(getattr(event, "finding_location", "") or "")
    for raw in (getattr(event, "action_sentences", []) or []):
        text = _strip_other_year_history(raw, year)
        for clause in _split_clauses(text):
            clause = _clean_clause_text(clause)
            clause = _normalize_text(clause)
            if not clause or _is_negative_or_empty(clause):
                continue
            if _is_historical_only(clause, year):
                continue
            if _is_recommendation_only(clause):
                continue
            if len(clause) < 10 and not re.search(r"교체|보수|용접|도장|coating|replace|repair|plug", clause, re.I):
                continue
            if re.fullmatch(r"(?:LL1|LL2|W|D5|C3|C4|M3|T1/T2|Crude|RC Ex\.?|O/H)\.?", clause, re.I):
                continue
            cats = categorize_text(clause, "")
            for cat in cats:
                items.append({
                    "year": year,
                    "category": cat,
                    "text": clause,
                    "location": loc,
                })
    return _unique_action_items(items)


def _extract_recommendation_items(event, allowed_years: set[int] | None = None) -> List[str]:
    year = int(getattr(event, "report_year", 0) or 0)
    if allowed_years is not None and year not in allowed_years:
        return []
    recs: List[str] = []
    for raw in (getattr(event, "recommendation_sentences", []) or []):
        text = _strip_other_year_history(raw, year)
        for clause in _split_clauses(text):
            clause = _clean_clause_text(clause)
            clause = _normalize_text(clause)
            if not clause or _is_negative_or_empty(clause) or _is_historical_only(clause, year):
                continue
            if _has_explicit_done(clause):
                continue
            if _looks_like_recommendation(clause):
                recs.append(f"[{year}] {clause}")
    return _unique_keep_order(recs)


def categorize_event(event) -> List[str]:
    return _unique_keep_order([item["category"] for item in _extract_action_items(event)])


def categorize_case(case: RepeatCase) -> List[str]:
    cats = []
    for event in case.events:
        cats.extend(categorize_event(event))
    return _unique_keep_order([c for c in CATEGORY_ORDER if c in cats])


def _collect_locations_from_items(items: List[dict], limit: int = 8) -> str:
    locs = _unique_keep_order([item.get("location", "") for item in items if item.get("location")])
    return ", ".join(locs[:limit])


def _occurrence_label(year_count: int) -> str:
    return "2회 이상 반복" if year_count >= 2 else "1회성"


def _build_title(equipment_no: str, equipment_name: str, category: str) -> str:
    base = f"{equipment_no} {equipment_name}".strip()
    return f"{base} - {category}" if category else base


def _build_detail_from_items(equipment_no: str, equipment_name: str, category: str, items: List[dict], recs: List[str]) -> str:
    years = sorted({item['year'] for item in items})
    lines = ["개요"]
    lines.append(f"- {equipment_no} {equipment_name} 설비의 '{category}' 조치만 추려서 정리함.")
    lines.append(f"- 발생구분: {_occurrence_label(len(years))}")
    lines.append(f"- 발생년도: {', '.join(map(str, years))}")
    locations = _collect_locations_from_items(items)
    if locations:
        lines.append(f"- 주요 부위: {locations}")
    lines.append("")
    lines.append("TA 조치사항")
    for item in items[:12]:
        lines.append(f"- [{item['year']}] {item['text']}")
    if not items:
        lines.append("- 명시 조치문 없음")
    lines.append("")
    lines.append("추후 권고사항")
    if recs:
        for r in recs[:8]:
            lines.append(f"- {r}")
    else:
        lines.append("- 명시 권고문 없음")
    return "\n".join(lines)


def _category_row_is_valid(category: str, items: List[dict]) -> bool:
    texts = " ".join(_normalize_text(item.get('text', '')) for item in items)
    if not texts:
        return False
    if _RECOMMEND_ONLY_RE.search(texts) and not _ACTION_DONE_RE.search(texts):
        return False
    if category == "Assembly 교체":
        if _TOOLING_RE.search(texts):
            return False
        return bool(
            re.search(
                r"bundle|tube\s*bundle|new\s*vessel|신규\s*용기|retube|shell\s*cover|floating\s*head|channel\b|backing\s*device|\bassembly\b|\bassy\b|duct|damper|vortex\s*breaker",
                texts,
                re.I,
            )
            and (_ACTION_DONE_RE.search(texts) or _REPLACE_RE.search(texts))
        )
    if category == "Nozzle 교체":
        return bool(re.search(r"nozzle|노즐|\bnzl\b|\belbow\b", texts, re.I))
    if category == "단순 내부 구성품 교체":
        return bool(_INTERNAL_PART_RE.search(texts))
    if category == "도장":
        return bool(_COATING_RE.search(texts))
    if category == "육성용접":
        return bool(_OVERLAY_RE.search(texts))
    if category == "단순 보수":
        return bool(
            _TOOLING_RE.search(texts)
            or _REPAIR_ACTION_RE.search(texts)
            or (_ACTION_DONE_RE.search(texts) and (_SIMPLE_REPAIR_RE.search(texts) or _TOOLING_RE.search(texts)))
        )
    return True


def _row_dict_from_items(no: int, equipment_no: str, equipment_name: str, category: str, items: List[dict], recs: List[str], needs_review: bool = False) -> dict:
    years = sorted({item['year'] for item in items})
    return {
        "NO": no,
        "Equipment No": equipment_no,
        "설비명": equipment_name,
        "발생구분": _occurrence_label(len(years)),
        "발생년도수": len(years),
        "발생년도": ", ".join(map(str, years)),
        "발췌 Category": category,
        "반복부위": _collect_locations_from_items(items),
        "TA 조치사항": "\n".join(f"- [{item['year']}] {item['text']}" for item in items[:12]) if items else "- 명시 조치문 없음",
        "추후 권고사항": "\n".join(f"- {r}" for r in recs[:8]) if recs else "- 명시 권고문 없음",
        "제목": _build_title(equipment_no, equipment_name, category),
        "상세 내용": _build_detail_from_items(equipment_no, equipment_name, category, items, recs),
        "검토필요여부": "예" if needs_review else "",
    }


def build_task_rows(cases: List[RepeatCase]) -> List[TaskRow]:
    rows: List[TaskRow] = []
    row_no = 1
    for case in cases:
        eq_no = case.equipment_no
        eq_name = case.equipment_name
        item_map: dict[str, list] = defaultdict(list)
        for event in case.events:
            for item in _extract_action_items(event):
                item_map[item['category']].append(item)
        for category, items in item_map.items():
            items = sorted(_unique_action_items(items), key=lambda x: (x['year'], x['text']))
            years = sorted({item['year'] for item in items})
            if len(years) < 2:
                continue
            if not _category_row_is_valid(category, items):
                continue
            recs = []
            for event in case.events:
                recs.extend(_extract_recommendation_items(event, allowed_years=set(years)))
            recs = _unique_keep_order(recs)
            row_dict = _row_dict_from_items(row_no, eq_no, eq_name, category, items, recs, needs_review=(case.confidence < 0.8))
            rows.append(TaskRow(
                no=row_no,
                equipment_no=eq_no,
                equipment_name=eq_name,
                year_count=len(years),
                years_str=row_dict["발생년도"],
                repeat_locations=row_dict["반복부위"],
                title=row_dict["제목"],
                detail=row_dict["상세 내용"],
                needs_review=(case.confidence < 0.8),
                maintenance_categories=category,
                ta_actions=row_dict["TA 조치사항"],
                followup_recommendations=row_dict["추후 권고사항"],
                occurrence_class=row_dict["발생구분"],
            ))
            row_no += 1
    return rows


def build_task_dataframe(cases: List[RepeatCase]) -> pd.DataFrame:
    rows = build_task_rows(cases)
    if not rows:
        return pd.DataFrame(columns=TASK_COLUMNS)
    data = []
    for r in rows:
        data.append({
            "NO": r.no,
            "Equipment No": r.equipment_no,
            "설비명": r.equipment_name,
            "발생구분": r.occurrence_class,
            "발생년도수": r.year_count,
            "발생년도": r.years_str,
            "발췌 Category": r.maintenance_categories,
            "반복부위": r.repeat_locations,
            "TA 조치사항": r.ta_actions,
            "추후 권고사항": r.followup_recommendations,
            "제목": r.title,
            "상세 내용": r.detail,
            "검토필요여부": "예" if r.needs_review else "",
        })
    df = pd.DataFrame(data, columns=TASK_COLUMNS)
    df["NO"] = range(1, len(df) + 1)
    return df


def build_equipment_summary_dataframe(all_events: List) -> pd.DataFrame:
    if not all_events:
        return pd.DataFrame(columns=TASK_COLUMNS)

    eq_name_map = {}
    group_map: dict[tuple[str, str], list] = defaultdict(list)
    eq_year_events: dict[str, list] = defaultdict(list)

    for event in all_events:
        eq_no = _normalize_text(getattr(event, "equipment_no", ""))
        if not eq_no:
            continue
        eq_name_map[eq_no] = _normalize_text(getattr(event, "equipment_name", "")) or eq_name_map.get(eq_no, "")
        eq_year_events[eq_no].append(event)
        for item in _extract_action_items(event):
            group_map[(eq_no, item['category'])].append(item)

    rows: List[dict] = []
    row_no = 1
    for eq_no, category in sorted(group_map.keys(), key=lambda x: (x[0], CATEGORY_ORDER.index(x[1]) if x[1] in CATEGORY_ORDER else 999)):
        items = sorted(_unique_action_items(group_map[(eq_no, category)]), key=lambda x: (x['year'], x['text']))
        years = sorted({item['year'] for item in items})
        if not years:
            continue
        if not _category_row_is_valid(category, items):
            continue
        recs = []
        for event in eq_year_events[eq_no]:
            recs.extend(_extract_recommendation_items(event, allowed_years=set(years)))
        recs = _unique_keep_order(recs)
        row = _row_dict_from_items(row_no, eq_no, eq_name_map.get(eq_no, ""), category, items, recs)
        rows.append(row)
        row_no += 1

    if not rows:
        return pd.DataFrame(columns=TASK_COLUMNS)

    df = pd.DataFrame(rows, columns=TASK_COLUMNS)
    order_map = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}
    df["_ord"] = df["발췌 Category"].map(order_map).fillna(999)
    df = df.sort_values(["발생년도수", "Equipment No", "_ord"], ascending=[False, True, True]).drop(columns=["_ord"]).reset_index(drop=True)
    df["NO"] = range(1, len(df) + 1)
    return df


def build_category_extract_dataframe(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df is None or task_df.empty:
        return pd.DataFrame(columns=TASK_COLUMNS)
    out = task_df.copy()
    order_map = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}
    out["_ord"] = out["발췌 Category"].map(order_map).fillna(999)
    out = out.sort_values(["_ord", "Equipment No", "발생년도수"], ascending=[True, True, False]).drop(columns=["_ord"]).reset_index(drop=True)
    out["NO"] = range(1, len(out) + 1)
    return out[TASK_COLUMNS].copy()
