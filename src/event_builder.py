"""v6 event_builder – 연도별 설비별 정비 이벤트 생성

개선 포인트
- 상태표시/헤더성 문장 제거 강화
- recommendation/조건문을 action 집계에서 제외
- raw row의 action_tags / damage_tags를 우선 활용
- 줄바꿈/OCR 파편은 최소 범위에서만 병합하여 category 오염을 줄임
"""
from __future__ import annotations

import re
from typing import List
from uuid import uuid4

import pandas as pd

from .schemas import MaintenanceEvent

ACTION_CLUSTERS = {
    "weld_repair": [
        "육성용접", "육성 용접", "overlay", "hardfacing", "재용접", "용접보수", "용접 보수", "보수용접", "ER-NiCr3", "ErNiCr-3",
        "weld repair", "repair welding", "seal welding", "seal-welding",
        "결함 제거 후 용접", "선형 결함 제거 후 용접", "grinding 후 용접", "stitch welding",
    ],
    "replace": [
        "교체", "replace", "신규 제작", "신규교체", "신규 용기", "제작 후 교체",
        "new vessel", "retube", "retubing", "spool 교체", "nozzle 교체", "bundle 교체", "신규 교체 실시", "bellows", "sleeve",
    ],
    "coating_repair": [
        "도장보수", "보수도장", "재도장", "paint repair", "coating repair",
        "phenolic epoxy", "coating 실시", "도장 실시", "touch-up",
    ],
    "plugging": ["plugging", "plug", "tube plug", "막음", "unplugging"],
    "temporary_fix": ["임시조치", "box-up", "compound sealing", "clamp", "patch"],
    "structural_repair": [
        "packing 교체", "tray 교체", "internal 교체", "distributor", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체",
        "bubble cap", "baffle", "weir plate", "punch plate", "mesh 교체",
        "screen mesh 교체", "clip 교체", "entry horn", "tray cap", "riser pipe hat",
        "tube support", "support 교체", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체",
    ],
}

FINDING_KEYWORDS = {
    "corrosion": ["부식", "corrosion", "pitting", "pit", "general corrosion", "내부부식"],
    "cracking": ["균열", "crack", "linear indication", "선형 결함", "선형결함"],
    "leak": ["누설", "leak", "천공"],
    "thinning": ["두께감소", "감육", "thinning", "두께 감소"],
    "damage": ["손상", "damage", "파손", "망실", "변형", "마모", "undercut", "이탈", "탈락", "풀림"],
    "plugging": ["막힘", "plugging", "coke plugging"],
    "coating_damage": ["도장 손상", "coating damage", "도장 박리"],
}

LOCATION_PATTERNS = {
    "top_head": r"top\s*head|상부\s*head|tophead",
    "lower_head": r"(하부|bottom|북쪽\s*head\s*하부)\s*(head|헤드)?",
    "shell_upper": r"shell\s*(상부|upper)|상부\s*shell",
    "shell_lower": r"shell\s*(하부|bottom|lower)|하부\s*shell|bottom\s*shell",
    "nozzle": r"nozzle|노즐",
    "internal_tray": r"tray|chimney\s*tray|bubble\s*cap|downcomer|weir\s*plate|screen|inner\s*screen|outer\s*screen|mesh|clip|tray\s*cap|grating|beam\s*support|support\s*channel",
    "internal_packing": r"packing|distributor|collector|baffle|demister|internal|entry\s*horn|riser\s*pipe\s*hat|punch\s*plate|panel\s*coil|new\s*coil|old\s*coil|coil\b|support|beam\s*support|support\s*channel",
    "entry_horn": r"entry\s*horn|wear\s*pad",
    "bundle_tube": r"bundle|tube\s*sheet|tube|retube|shell\s*cover|floating\s*head|channel",
    "lining": r"lining|clad|strip\s*lining|concrete\s*lining",
    "flange": r"flange|플랜지",
    "pipe_line": r"배관|line|spool|injection\s*point|culvert|pipe",
    "coating_area": r"도장|paint|coating|phenolic\s*epoxy|sand\s*blasting|sandblasting",
}

_RECOM_RE = re.compile(
    r"차기\s*ta|다음\s*ta|추후|향후|권고|recommend|next\s*ta|차기\s*검사|"
    r"검사.*필요|교체.*요망|교체.*필요|교체할\s*경우|보수.*요함|보수.*필요|요망|실시\s*요함|필요함|적용\s*검토|정밀\s*두께\s*측정\s*필요",
    re.I,
)
_RECOMMENDATION_ACTION_RE = re.compile(r"교체\s*요함|교체\s*필요|교체할\s*경우|적용\s*검토|차기|예정|요망", re.I)
_NOISE_RE = re.compile(
    r"^(?:\(?\d+\)?\s*)?(?:line\s*no\.?|nozzle\s*no\.?|"
    r"설계두께|최소허용두께|설계재질|부식여유|remaining thickness|"
    r"page\s*\d+|sheet\s*\d+|표면\s*상태|도장상태\s*확인|mesh\s*상태|head\s*표면\s*상태|"
    r"검사사항\s*\(초기/상세\)|구분\s*tube\s*shell|초기\s*상태|상세\s*검사)$",
    re.I,
)
_THICKNESS_DATA_RE = re.compile(
    r"설계재질|설계두께|최소허용두께|부식여유|최소두께|ONT\s*:?\s*\d|ACT\s*:?\s*\d|remaining\s*thickness",
    re.I,
)
_PURE_NOACTION_RE = re.compile(
    r"^(?:이상\s*없음|양호|특이사항\s*없음|no abnormal|without corrosion|부식 없이)\s*[\.。]?\s*$",
    re.I,
)
_HEADER_LIKE_RE = re.compile(r"^[A-Z0-9\-() /\"”“.#]+$")
_CONTINUATION_START_RE = re.compile(r"^(?:및|후|또는|발견되어|하여|하였으며|하고|실시|제거|교체|보수|검사|세척|또한|관련하여|차단을\s*위해|위하여)", re.I)
_FRAGMENT_END_RE = re.compile(r"(?:Nozzle\s*No\.?|Line\s*No\.?|구분|상부|하부|위하여|위해|부분|부위|에서|후|및|,|:)\s*$", re.I)
_DONE_RE = re.compile(
    r"교체함|교체\s*설치함|교체\s*하였음|교체\s*완료함|교체\s*완료하였음|교체됨|신규\s*교체|신규\s*제작|제작\s*후\s*교체|설치함|설치\s*완료|실시함|실시하였음|실시\s*완료|작업함|작업\s*실시|보수\s*완료|보강함|보강\s*실시|용접\s*실시|용접\s*보수|재시공|시공하였음|repair(ed)?|replace(d)?|fabricated|도장\s*실시|coating\s*실시|완료함",
    re.I,
)
_AIR_COOLER_PLUG_SERVICE_RE = re.compile(r"air\s*cooler\s*plug|a/?c\s*plug|plug\b", re.I)
_AIR_COOLER_PLUG_NONREPAIR_RE = re.compile(r"분해|조립|해체|탈거|재조립|opening|closing|open|close", re.I)
_BUNDLE_KEYWORD_RE = re.compile(r"\bbundle\b|번들|retube|tube\s*bundle", re.I)
_BUNDLE_PERFORMED_RE = re.compile(
    r"bundle\s*사전\s*신규\s*제작\s*및\s*교체|bundle\s*사전\s*제작\s*및\s*교체|신규\s*bundle\s*제작후\s*교체|신규\s*bundle\s*로\s*교체함|신규\s*bundle\s*제작\s*되어\s*교체하였|신규\s*bundle\s*제작\s*후\s*교체|신규\s*입고된\s*bundle|bundle\s*교체함|retube|제작\s*후\s*교체",
    re.I,
)
_INTERNAL_EXCLUDE_RE = re.compile(r"충진물|\bfiller\b|filter\s*media|adsorbent|desiccant|diesel\s*sand", re.I)
_SMALL_PART_ONLY_RE = re.compile(r"\bbolt\b|\bnut\b|\bgasket\b|washer|stud|pin|keeper", re.I)
_NOZZLE_KEYWORD_RE = re.compile(r"nozzle|노즐|\bnzl\b|\belbow\b", re.I)
_INTERNAL_OBJECT_RE = re.compile(
    r"tray|chimney\s*tray|bubble\s*cap|downcomer|weir\s*plate|screen|inner\s*screen|outer\s*screen|mesh|clip|support\s*clip|packing|distributor|collector|baffle|demister|internal|entry\s*horn|riser\s*pipe\s*hat|punch\s*plate|panel\s*coil|new\s*coil|old\s*coil|coil\b|beam\s*support|support\s*channel|tube\s*support|corrosion\s*probe\s*assembly|probe\s*assembly|corrosion\s*probe|grating|flat\s*form|\bvalve\b",
    re.I,
)
_ASSEMBLY_OBJECT_RE = re.compile(
    r"bundle|tube\s*bundle|retube|retubing|new\s*vessel|신규\s*용기|\bvessel\b|shell\s*cover|floating\s*head|\bchannel\b|\bassembly\b|\bassy\b|\bduct\b|\bdamper\b|expansion\s*joint|bellows|sleeve|saddle(?!\s*clip)",
    re.I,
)
_INSTALL_OR_REPLACE_RE = re.compile(r"교체|replace|설치|install|신규\s*제작|신규\s*교체|제작\s*후\s*교체|fabricat|retube", re.I)
_ACTION_FALLBACK_RE = re.compile(r"교체|replace|retube|retubing|보수|repair|보강|용접|weld|도장|coating|plugging|plug|blind\s*처리|재시공|설치|시공", re.I)
_ASSEMBLY_PERFORMED_RE = re.compile(
    r"신규\s*제작|사전\s*제작|제작\s*후\s*교체|pre\s*-?fabricat|신규\s*교체\s*실시|교체함|교체\s*설치함|설치함|retube|replace(d)?",
    re.I,
)
_SPLIT_LINE_RE = re.compile(r"(?:\\n|\n)+")


def _parse_tag_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[,/;|]\s*", text)
    return [p.strip() for p in parts if p.strip()]


def classify_action(text: str) -> List[str]:
    t = str(text or "").lower()
    hits = []
    for cluster, keywords in ACTION_CLUSTERS.items():
        if any(k.lower() in t for k in keywords):
            hits.append(cluster)
    return hits


def classify_finding(text: str) -> List[str]:
    t = str(text or "").lower()
    hits = []
    for ftype, keywords in FINDING_KEYWORDS.items():
        if any(k.lower() in t for k in keywords):
            hits.append(ftype)
    return hits


def extract_locations(text: str) -> List[str]:
    t = str(text or "").lower()
    hits = []
    for loc, pattern in LOCATION_PATTERNS.items():
        if re.search(pattern, t, re.I):
            hits.append(loc)
    return hits


def extract_measurements(text: str) -> str:
    hits = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*mm", str(text or ""), re.I):
        hits.append(m.group(0))
    for m in re.finditer(r"(?:ONT|ACT|깊이|depth|max|잔여)\s*[\s:]?\s*(\d+(?:\.\d+)?)\s*mm", str(text or ""), re.I):
        hits.append(m.group(0))
    for m in re.finditer(r"약?\s*(\d+(?:\.\d+)?)\s*%", str(text or "")):
        hits.append(m.group(0))
    return ", ".join(hits[:5]) if hits else ""


def _normalize_sentence(text: str) -> str:
    t = str(text or "")
    t = _SPLIT_LINE_RE.sub(" ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_noise_sentence(text: str) -> bool:
    t = _normalize_sentence(text)
    if len(t) < 6:
        return True
    if _NOISE_RE.search(t):
        return True
    if _PURE_NOACTION_RE.match(t):
        return True
    if _HEADER_LIKE_RE.match(t) and not re.search(r"교체|보수|검사|부식|균열|pitting|corrosion|damage|defect|도장|용접", t, re.I):
        return True
    if _THICKNESS_DATA_RE.search(t) and not re.search(r"부식|pitting|corrosion|crack|균열|손상|damage|감육|두께감소", t, re.I):
        return True
    if t.endswith("검사사항 (초기/상세)"):
        return True
    return False


def is_action_sentence(text: str, raw_tags: List[str] | None = None) -> bool:
    t = _normalize_sentence(text)
    raw_tags = raw_tags or []
    if _RECOMMENDATION_ACTION_RE.search(t) and not _DONE_RE.search(t):
        return False
    if _RECOM_RE.search(t) and not _DONE_RE.search(t):
        return False
    if (
        _AIR_COOLER_PLUG_SERVICE_RE.search(t)
        and _AIR_COOLER_PLUG_NONREPAIR_RE.search(t)
        and not re.search(r"누설|leak|교체|replace|보수|repair|용접|weld|균열|결함|손상|damage|막음|plugging|unplugging|재확관|확관|compound\s*sealing|box-?up", t, re.I)
    ):
        return False
    return bool(raw_tags or classify_action(t) or (_ACTION_FALLBACK_RE.search(t) and (_DONE_RE.search(t) or re.search(r"보강함|시공하였음|재시공|실시하였음", t, re.I))))


def is_finding_sentence(text: str, raw_tags: List[str] | None = None) -> bool:
    return bool((raw_tags or []) or classify_finding(text))


def is_recommendation_sentence(text: str) -> bool:
    t = _normalize_sentence(text)
    return bool(_RECOM_RE.search(t) and not _DONE_RE.search(t))


def _merge_sentences(sentences: List[str], max_length: int = 320) -> str:
    merged = " / ".join(s.strip() for s in sentences if s.strip())
    return merged[:max_length]


_VERIFIED_EVENT_ACTION_OVERRIDES = {
    ("02E-129A", 2014): [
        "Backing Device PT 결과 양호함.",
        "신규 Bundle 제작후 교체 (A213 T5 OD 19.05 x 2.11T x 6100L).",
        "Channel Cover Flange Bolt Face 및 Floating Head Cover Flange Bolt Face Gasket를 Double Jacket Type으로 사용함.",
        "Shell Cover Bolt 및 Gasket Face 기계가공 실시함.",
    ],
}


def _is_verified_bundle_replacement_sentence(text: str, raw_action_tags: List[str] | None = None) -> bool:
    t = _normalize_sentence(text)
    raw_action_tags = [x.strip().lower() for x in (raw_action_tags or []) if x]
    if not t or not _BUNDLE_KEYWORD_RE.search(t):
        return False
    explicit_done = bool(_BUNDLE_PERFORMED_RE.search(t))
    tagged_replace = "replace" in raw_action_tags
    if _RECOM_RE.search(t) and not explicit_done:
        return False
    if re.search(r"검토|필요|요망|도면\s*수정\s*필요", t, re.I) and not explicit_done:
        return False
    return bool(explicit_done or tagged_replace and _DONE_RE.search(t) or re.search(r"\(1\)\s*bundle\s*사전\s*신규\s*제작\s*및\s*교체", t, re.I))


def _is_verified_nozzle_replacement_sentence(text: str, raw_action_tags: List[str] | None = None) -> bool:
    t = _normalize_sentence(text)
    raw_action_tags = [x.strip().lower() for x in (raw_action_tags or []) if x]
    if not t or not _NOZZLE_KEYWORD_RE.search(t):
        return False
    explicit_done = bool(_DONE_RE.search(t))
    if _RECOM_RE.search(t) and not explicit_done:
        return False
    if re.search(r"상태|부식|감육|pitting|측정|검사|필요|요망|검토", t, re.I) and not (explicit_done or re.search(r"신품|신규|제작", t, re.I)):
        return False
    return bool(
        ("replace" in raw_action_tags and (explicit_done or _INSTALL_OR_REPLACE_RE.search(t)))
        or re.search(r"(?:nozzle|노즐|nzl|elbow).{0,40}(?:교체|replace|신품|신규|제작)", t, re.I)
        or re.search(r"(?:교체|replace|신품|신규|제작).{0,40}(?:nozzle|노즐|nzl|elbow)", t, re.I)
    )


def _is_verified_internal_replacement_sentence(text: str, raw_action_tags: List[str] | None = None) -> bool:
    t = _normalize_sentence(text)
    raw_action_tags = [x.strip().lower() for x in (raw_action_tags or []) if x]
    if not t or not _INTERNAL_OBJECT_RE.search(t):
        return False
    if _INTERNAL_EXCLUDE_RE.search(t):
        return False
    if _SMALL_PART_ONLY_RE.search(t) and not re.search(r"packing|tray|mesh|screen|clip|baffle|support|distributor|collector|valve|corrosion\s*probe|probe\s*assembly", t, re.I):
        return False
    explicit_done = bool(_DONE_RE.search(t))
    if _RECOM_RE.search(t) and not explicit_done:
        return False
    return bool(
        ("replace" in raw_action_tags and (_INSTALL_OR_REPLACE_RE.search(t) or explicit_done))
        or (_INSTALL_OR_REPLACE_RE.search(t) and explicit_done)
        or re.search(r"철거.*new\s*(packing|tray|mesh|screen)|new\s*(packing|tray|mesh|screen).*(교체|설치)", t, re.I)
    )


def _is_verified_assembly_replacement_sentence(text: str, raw_action_tags: List[str] | None = None) -> bool:
    t = _normalize_sentence(text)
    raw_action_tags = [x.strip().lower() for x in (raw_action_tags or []) if x]
    if not t or not _ASSEMBLY_OBJECT_RE.search(t):
        return False
    if _NOZZLE_KEYWORD_RE.search(t) or _INTERNAL_OBJECT_RE.search(t):
        return False
    if _SMALL_PART_ONLY_RE.search(t) and not re.search(r"backing\s*device|vortex\s*breaker|duct|damper|bundle|retube|vessel|shell\s*cover|floating\s*head|channel", t, re.I):
        return False
    explicit_done = bool(_DONE_RE.search(t) or _BUNDLE_PERFORMED_RE.search(t) or _ASSEMBLY_PERFORMED_RE.search(t))
    if _RECOM_RE.search(t) and not explicit_done:
        return False
    return bool(
        _is_verified_bundle_replacement_sentence(t, raw_action_tags)
        or (("replace" in raw_action_tags) and explicit_done)
        or (_ASSEMBLY_OBJECT_RE.search(t) and _ASSEMBLY_PERFORMED_RE.search(t) and explicit_done)
    )


def _extract_verified_category_actions(year_group: pd.DataFrame) -> dict[str, List[str]]:
    actions = {"nozzle": [], "internal": [], "assembly": []}
    for _, row in year_group.reset_index(drop=True).iterrows():
        sentence = _normalize_sentence(row.get("sentence", ""))
        raw_action_tags = _parse_tag_list(row.get("action_tags"))
        if _is_verified_nozzle_replacement_sentence(sentence, raw_action_tags):
            if sentence not in actions["nozzle"]:
                actions["nozzle"].append(sentence)
        if _is_verified_internal_replacement_sentence(sentence, raw_action_tags):
            if sentence not in actions["internal"]:
                actions["internal"].append(sentence)
        if _is_verified_assembly_replacement_sentence(sentence, raw_action_tags):
            if sentence not in actions["assembly"]:
                actions["assembly"].append(sentence)
    return actions


def _refresh_event_summary_fields(event: MaintenanceEvent) -> MaintenanceEvent:
    event.action_detail = _merge_sentences(event.action_sentences[:4])
    summary_parts = []
    summary_parts.extend((event.finding_sentences or [])[:2])
    summary_parts.extend((event.action_sentences or [])[:2])
    summary_parts.extend((event.recommendation_sentences or [])[:1])
    event.evidence_summary = _merge_sentences(summary_parts, max_length=450)
    return event


def _apply_verified_event_corrections(event: MaintenanceEvent) -> MaintenanceEvent:
    override_actions = _VERIFIED_EVENT_ACTION_OVERRIDES.get((event.equipment_no, int(event.report_year or 0)))
    if not override_actions:
        return event

    existing = list(event.action_sentences or [])
    for sentence in reversed(override_actions):
        if sentence not in existing:
            existing.insert(0, sentence)
    event.action_sentences = existing[:12]

    action_types = [x.strip() for x in str(event.action_type or "").split(",") if x.strip()]
    for tag in ["replace"]:
        if tag not in action_types:
            action_types.append(tag)
    event.action_type = ", ".join(action_types)

    locs = [x.strip() for x in str(event.finding_location or "").split(",") if x.strip()]
    for loc in ["bundle_tube", "flange"]:
        if loc not in locs:
            locs.append(loc)
    event.finding_location = ", ".join(locs[:6])
    return _refresh_event_summary_fields(event)


def _row_records(year_group: pd.DataFrame) -> List[dict]:
    records: List[dict] = []
    for _, row in year_group.reset_index(drop=True).iterrows():
        sentence = _normalize_sentence(row.get("sentence", ""))
        if not sentence or is_noise_sentence(sentence):
            continue
        records.append({
            "text": sentence,
            "action_tags": _parse_tag_list(row.get("action_tags")),
            "damage_tags": _parse_tag_list(row.get("damage_tags")),
        })

    merged: List[dict] = []
    for rec in records:
        s = rec["text"]
        if not merged:
            merged.append(rec)
            continue
        prev = merged[-1]
        prev_s = prev["text"]
        should_merge = (
            (len(s) <= 40 and not re.search(r"\(\d+\)|신규\s*제작|교체함|교체\s*설치함|실시함|도장\s*실시|용접\s*실시", s, re.I))
            or _CONTINUATION_START_RE.search(s)
            or _FRAGMENT_END_RE.search(prev_s)
        )
        if should_merge:
            prev["text"] = f"{prev_s} {s}".strip()
            prev["action_tags"] = list(dict.fromkeys(prev["action_tags"] + rec["action_tags"]))
            prev["damage_tags"] = list(dict.fromkeys(prev["damage_tags"] + rec["damage_tags"]))
        else:
            merged.append(rec)
    return merged


def build_events_for_equipment(equipment_no: str, equipment_name: str, rows: pd.DataFrame) -> List[MaintenanceEvent]:
    if rows.empty:
        return []

    rows = rows.copy().reset_index(drop=True)
    rows["_year"] = pd.to_numeric(rows.get("year", pd.Series(dtype=float)), errors="coerce")
    rows = rows.dropna(subset=["_year"])
    rows["_year"] = rows["_year"].astype(int)

    events: List[MaintenanceEvent] = []

    for year, year_group in rows.groupby("_year", sort=True):
        findings: List[str] = []
        actions: List[str] = []
        recommendations: List[str] = []
        all_locations: List[str] = []
        all_findings: List[str] = []
        all_action_types: List[str] = []
        measurement_texts: List[str] = []
        source_files: List[str] = []
        evidence_ids: List[str] = []

        for idx, rec in enumerate(_row_records(year_group), start=1):
            sentence = rec["text"]
            raw_action_tags = rec["action_tags"]
            raw_damage_tags = rec["damage_tags"]

            if not sentence or is_noise_sentence(sentence):
                continue

            locs = extract_locations(sentence)
            all_locations.extend(locs)
            fds = list(dict.fromkeys(raw_damage_tags + classify_finding(sentence)))
            all_findings.extend(fds)

            meas = extract_measurements(sentence)
            if meas:
                measurement_texts.append(meas)

            if is_recommendation_sentence(sentence) and not is_action_sentence(sentence, raw_action_tags):
                recommendations.append(sentence)
            elif is_action_sentence(sentence, raw_action_tags):
                actions.append(sentence)
                acts = list(dict.fromkeys(raw_action_tags + classify_action(sentence)))
                all_action_types.extend(acts)
            elif is_finding_sentence(sentence, raw_damage_tags):
                findings.append(sentence)

            evidence_ids.append(f"{equipment_no}-{year}-{idx}")

        for src in year_group.get("source_file", pd.Series(dtype=str)).dropna().astype(str):
            src = src.strip()
            if src and src not in source_files:
                source_files.append(src)

        verified_actions_by_category = _extract_verified_category_actions(year_group)
        default_locations = {
            "nozzle": ["nozzle"],
            "internal": ["internal_packing"],
            "assembly": ["bundle_tube"],
        }
        all_verified_actions: List[str] = []
        for key in ["assembly", "internal", "nozzle"]:
            for sentence in verified_actions_by_category.get(key, []):
                if sentence not in all_verified_actions:
                    all_verified_actions.append(sentence)
        for sentence in reversed(all_verified_actions):
            if sentence not in actions:
                actions.insert(0, sentence)
        if all_verified_actions:
            all_action_types.extend(["replace"])
            for key, sentences in verified_actions_by_category.items():
                for sentence in sentences:
                    all_locations.extend(extract_locations(sentence) or default_locations.get(key, []))

        if not findings and not actions and not recommendations:
            continue

        findings = [
            s for s in findings
            if not _THICKNESS_DATA_RE.search(s) or re.search(r"부식|pitting|corrosion|crack|균열|손상|감육|두께감소", s, re.I)
        ]

        loc_set = list(dict.fromkeys(all_locations))
        find_set = list(dict.fromkeys(all_findings))
        act_set = list(dict.fromkeys(all_action_types))

        summary_parts = []
        summary_parts.extend(findings[:2])
        summary_parts.extend(actions[:2])
        summary_parts.extend(recommendations[:1])

        event = MaintenanceEvent(
            event_id=str(uuid4())[:8],
            equipment_no=equipment_no,
            equipment_name=equipment_name,
            report_year=int(year),
            source_files=source_files,
            finding_location=", ".join(loc_set[:4]),
            finding_damage=", ".join(find_set[:3]),
            finding_measurement=", ".join(dict.fromkeys(measurement_texts))[:120],
            finding_sentences=findings[:8],
            action_type=", ".join(act_set[:6]) if act_set else "",
            action_detail=_merge_sentences(actions[:4]),
            action_sentences=actions[:12],
            recommendation=_merge_sentences(recommendations[:4]),
            recommendation_sentences=recommendations[:8],
            evidence_sentence_ids=evidence_ids,
            evidence_summary=_merge_sentences(summary_parts, max_length=450),
        )
        event = _apply_verified_event_corrections(event)
        event = _refresh_event_summary_fields(event)
        events.append(event)

    return sorted(events, key=lambda e: e.report_year)
