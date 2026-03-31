"""v6 repeat_judger – 반복 정비 판정 모듈

개선 원칙
- '같은 설비에서 같은 손상이 있었다'가 아니라
  '같은 유형의 실제 조치가 서로 다른 연도에 반복 수행되었는가'만 판정한다.
- finding-only 반복은 false positive가 많아 기본 판정에서 제외한다.
- replacement는 generic '교체'만으로 묶지 않고 문장 내 대상(component/object) 유사성까지 본다.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List
from uuid import uuid4

from .schemas import MaintenanceEvent, RepeatCase

ACTION_GROUP_MAP = {
    "weld_repair": "용접/결함보수",
    "replace": "교체",
    "coating_repair": "도장",
    "plugging": "Plugging",
    "temporary_fix": "임시보수",
    "structural_repair": "내부구성품 교체/보수",
}

LOCATION_KR_MAP = {
    "top_head": "Top Head",
    "lower_head": "하부 Head",
    "shell_upper": "Shell 상부",
    "shell_lower": "Shell 하부",
    "nozzle": "Nozzle",
    "internal_tray": "Internal/Tray",
    "internal_packing": "Internal/Packing",
    "entry_horn": "Entry Horn/Wear Pad",
    "bundle_tube": "Bundle/Tube",
    "lining": "Lining",
    "flange": "Flange",
    "pipe_line": "배관 구간",
    "coating_area": "도장 구간",
}

FINDING_KR_MAP = {
    "corrosion": "부식/Pitting",
    "cracking": "균열",
    "leak": "누설",
    "thinning": "감육",
    "damage": "손상/변형",
    "plugging": "막힘",
    "coating_damage": "도장 손상",
}

_COMPONENT_PATTERNS = [
    ("mesh", r"mesh|screen|hold\s*-?down|holdown"),
    ("clip", r"clip|grid\s*clip|saddle\s*clip"),
    ("lining", r"lining|concrete\s*lining"),
    ("vessel", r"new\s*vessel|신규\s*용기|vessel|drum|column|tower"),
    ("shell", r"shell"),
    ("head", r"head"),
    ("nozzle", r"nozzle|노즐"),
    ("spool", r"spool|line"),
    ("bundle", r"bundle|tube"),
    ("coil", r"coil"),
    ("sand", r"diesel\s*sand|\bsand\b"),
    ("filler", r"충진물|filler"),
    ("packing", r"packing|distributor|collector|baffle|tray|bubble\s*cap"),
    ("small_part", r"bolt|nut|gasket|test\s*ring|collar\s*bolt|f/h\s*bolt"),
]

_EXCLUDED_COMPONENTS = {"filler", "small_part"}


def _action_set(event: MaintenanceEvent) -> set:
    return {a.strip() for a in str(event.action_type or "").split(",") if a.strip()}


def _loc_set(event: MaintenanceEvent) -> set:
    return {l.strip() for l in str(event.finding_location or "").split(",") if l.strip()}


def _finding_set(event: MaintenanceEvent) -> set:
    return {f.strip() for f in str(event.finding_damage or "").split(",") if f.strip()}


def _action_text(event: MaintenanceEvent) -> str:
    texts = []
    texts.extend(getattr(event, "action_sentences", []) or [])
    if getattr(event, "action_detail", ""):
        texts.append(event.action_detail)
    return " ".join(str(x) for x in texts if str(x).strip())


def _component_signature(event: MaintenanceEvent) -> str:
    text = _action_text(event)
    hits = []
    for name, pattern in _COMPONENT_PATTERNS:
        if re.search(pattern, text, re.I):
            hits.append(name)
    hits = [x for x in hits if x not in _EXCLUDED_COMPONENTS]
    return ",".join(sorted(dict.fromkeys(hits))) if hits else "generic"


def _build_evidence_line(event: MaintenanceEvent) -> str:
    parts = []
    if event.finding_location:
        locs = [LOCATION_KR_MAP.get(l.strip(), l.strip()) for l in str(event.finding_location).split(",") if l.strip()]
        if locs:
            parts.append(f"[{', '.join(dict.fromkeys(locs))}]")
    if event.action_sentences:
        parts.append(" / ".join(event.action_sentences[:2]))
    elif event.action_detail:
        parts.append(event.action_detail)
    if event.recommendation:
        parts.append(f"[권고] {event.recommendation[:100]}")
    return " ".join(parts).strip()


def judge_repeat_cases(
    equipment_events: Dict[str, List[MaintenanceEvent]],
    equipment_names: Dict[str, str],
    min_years: int = 2,
) -> List[RepeatCase]:
    cases: List[RepeatCase] = []

    for eq_no, events in equipment_events.items():
        eq_name = equipment_names.get(eq_no, "")
        if len({e.report_year for e in events}) < min_years:
            continue

        grouped: Dict[str, List[MaintenanceEvent]] = defaultdict(list)
        for event in events:
            acts = _action_set(event)
            if not acts:
                continue
            comp_sig = _component_signature(event)
            if comp_sig in _EXCLUDED_COMPONENTS:
                continue
            for act in acts:
                key = f"{act}|{comp_sig}"
                grouped[key].append(event)

        for group_key, group_events in grouped.items():
            act, comp_sig = group_key.split("|", 1)
            years = sorted({e.report_year for e in group_events})
            if len(years) < min_years:
                continue

            # replacement는 generic/소부품/권고성 문장은 반복으로 보지 않음
            if act == "replace" and comp_sig == "generic":
                continue
            if any(re.search(r"교체\s*요함|교체\s*필요|교체할\s*경우|적용\s*검토", _action_text(e), re.I) for e in group_events):
                performed = [e for e in group_events if not re.search(r"교체\s*요함|교체\s*필요|교체할\s*경우|적용\s*검토", _action_text(e), re.I)]
                if len({e.report_year for e in performed}) < min_years:
                    continue
                group_events = performed
                years = sorted({e.report_year for e in group_events})

            all_locs = sorted({loc for e in group_events for loc in _loc_set(e)})
            all_findings = sorted({fd for e in group_events for fd in _finding_set(e)})
            history_lines = [f"- {e.report_year}: {_build_evidence_line(e)}" for e in sorted(group_events, key=lambda x: x.report_year)]

            reasons = [
                f"{ACTION_GROUP_MAP.get(act, act)} 조치가 {len(years)}개 연도({', '.join(map(str, years))})에서 반복 수행됨",
            ]
            if comp_sig != "generic":
                reasons.append(f"대상 부품/조치 객체: {comp_sig}")
            if all_locs:
                reasons.append("주요 위치: " + ", ".join(LOCATION_KR_MAP.get(x, x) for x in all_locs))

            confidence = 0.82
            if len(years) >= 3:
                confidence = 0.92
            elif comp_sig != "generic":
                confidence = 0.88

            cases.append(
                RepeatCase(
                    case_id=str(uuid4())[:8],
                    equipment_no=eq_no,
                    equipment_name=eq_name,
                    repeat_key=f"{eq_no}|{group_key}",
                    action_cluster=ACTION_GROUP_MAP.get(act, act),
                    location_cluster=", ".join(LOCATION_KR_MAP.get(l, l) for l in all_locs),
                    damage_cluster=", ".join(FINDING_KR_MAP.get(f, f) for f in all_findings),
                    years=years,
                    events=sorted(group_events, key=lambda x: x.report_year),
                    is_repeat=True,
                    repeat_reason=" / ".join(reasons),
                    confidence=confidence,
                    history_by_year=history_lines,
                )
            )

    return sorted(cases, key=lambda c: (c.equipment_no, c.action_cluster, c.repeat_key))


def merge_cases_per_equipment(cases: List[RepeatCase]) -> List[RepeatCase]:
    """v6 strict 모드에서는 케이스를 설비별로 다시 합치지 않는다.

    과거 방식은 설비 하나에서 서로 다른 1회성 조치가 여러 연도에 존재할 때
    하나의 반복 설비처럼 보이게 만드는 문제가 있었음.
    """
    dedup = {}
    for case in cases:
        dedup[case.repeat_key] = case
    return sorted(dedup.values(), key=lambda c: (c.equipment_no, c.action_cluster, c.repeat_key))
