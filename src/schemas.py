"""v6 schemas – 데이터 구조 정의"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EquipmentAnchor:
    equipment_no: str
    candidate_name: str
    pattern: str
    source_file: str
    source_line: str
    score: float = 0.0
    is_selected: bool = False


@dataclass
class ContextBlock:
    block_id: str
    file_name: str
    page_no: int
    raw_text: str
    prev_text: str = ""
    next_text: str = ""
    section_title: str = ""
    report_year: int = 0
    block_type: str = "paragraph"


@dataclass
class SentenceTag:
    sentence_id: str
    block_id: str
    text: str
    context_text: str
    equipment_no: str = ""
    equipment_name: str = ""
    year: int = 0
    source_file: str = ""
    location_candidates: List[str] = field(default_factory=list)
    damage_candidates: List[str] = field(default_factory=list)
    action_candidates: List[str] = field(default_factory=list)
    measurement_text: str = ""
    recommendation_text: str = ""
    is_action_sentence: bool = False
    is_finding_sentence: bool = False
    is_recommendation: bool = False
    is_thickness_data: bool = False
    is_noise: bool = False


@dataclass
class MaintenanceEvent:
    event_id: str
    equipment_no: str
    equipment_name: str
    report_year: int
    source_files: List[str] = field(default_factory=list)
    finding_location: str = ""
    finding_damage: str = ""
    finding_measurement: str = ""
    finding_sentences: List[str] = field(default_factory=list)
    action_type: str = ""
    action_detail: str = ""
    action_sentences: List[str] = field(default_factory=list)
    recommendation: str = ""
    recommendation_sentences: List[str] = field(default_factory=list)
    evidence_sentence_ids: List[str] = field(default_factory=list)
    evidence_summary: str = ""


@dataclass
class RepeatCase:
    case_id: str
    equipment_no: str
    equipment_name: str
    repeat_key: str
    action_cluster: str = ""
    location_cluster: str = ""
    damage_cluster: str = ""
    years: List[int] = field(default_factory=list)
    events: List[MaintenanceEvent] = field(default_factory=list)
    is_repeat: bool = False
    repeat_reason: str = ""
    confidence: float = 0.0
    title: str = ""
    overview: str = ""
    history_by_year: List[str] = field(default_factory=list)
    long_term_suggestion: str = ""


@dataclass
class TaskRow:
    no: int = 0
    equipment_no: str = ""
    equipment_name: str = ""
    year_count: int = 0
    years_str: str = ""
    repeat_locations: str = ""
    repeat_damages: str = ""
    representative_actions: str = ""
    title: str = ""
    detail: str = ""
    repeat_reason: str = ""
    needs_review: bool = False
    maintenance_categories: str = ""
    ta_actions: str = ""
    followup_recommendations: str = ""
    occurrence_class: str = ""
