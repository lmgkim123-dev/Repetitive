from dataclasses import dataclass, field
from typing import List

@dataclass
class MaintenanceEvent:
    event_id: str
    equipment_no: str
    equipment_name: str
    report_year: int
    source_files: List[str] = field(default_factory=list)
    finding_location: str = ''
    finding_damage: str = ''
    finding_measurement: str = ''
    finding_sentences: List[str] = field(default_factory=list)
    action_type: str = ''
    action_detail: str = ''
    action_sentences: List[str] = field(default_factory=list)
    recommendation: str = ''
    recommendation_sentences: List[str] = field(default_factory=list)
    evidence_sentence_ids: List[str] = field(default_factory=list)
    evidence_summary: str = ''

@dataclass
class RepeatCase:
    equipment_no: str
    equipment_name: str
    years: List[int]
    action_cluster: str = ''
    location_cluster: str = ''
    damage_cluster: str = ''
    repeat_reason: str = ''
    confidence: float = 0.9
    events: List[MaintenanceEvent] = field(default_factory=list)

@dataclass
class TaskRow:
    no: int
    equipment_no: str
    equipment_name: str
    year_count: int
    years_str: str
    title: str
    detail: str
    needs_review: bool
    maintenance_categories: str
    ta_actions: str
    followup_recommendations: str
    occurrence_class: str
    repeat_locations: str = ""
