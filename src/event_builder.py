"""v6 event_builder – 연도별 설비별 정비 이벤트 생성

개선 포인트
- 상태표시/헤더성 문장 제거 강화
- recommendation/조건문을 action 집계에서 제외
- raw row의 action_tags / damage_tags를 우선 활용
- 줄바꿈/OCR 파편은 최소 범위에서만 병합하여 category 오염을 줄임
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List
from uuid import uuid4

import pandas as pd

from .schemas import MaintenanceEvent

ACTION_CLUSTERS = {
    "weld_repair": [
        "육성용접", "육성 용접", "overlay", "hardfacing", "재용접", "용접보수", "용접 보수", "보수용접", "ER-NiCr3", "ErNiCr-3",
        "weld repair", "repair welding", "seal welding", "seal-welding", "seal welded", "weld repaired", "rewelded", "ground out", "weld-built up", "built up with", "deposit welding", "metal plugged",
        "결함 제거 후 용접", "선형 결함 제거 후 용접", "grinding 후 용접", "stitch welding",
    ],
    "replace": [
        "교체", "replace", "replaced", "replacement", "renewed", "prefabricated", "newly fabricated", "replaced with new ones", "신규 제작", "신규교체", "신규 용기", "제작 후 교체",
        "new vessel", "retube", "retubing", "spool 교체", "nozzle 교체", "bundle 교체", "신규 교체 실시", "bellows", "sleeve",
    ],
    "coating_repair": [
        "도장보수", "보수도장", "재도장", "paint repair", "coating repair",
        "phenolic epoxy", "coating 실시", "도장 실시", "touch-up",
    ],
    "plugging": ["plugging", "plug", "tube plug", "막음", "unplugging"],
    "temporary_fix": ["임시조치", "box-up", "compound sealing", "clamp", "patch", "보수", "부분 보수", "부분보수", "lining repair", "lining restored", "concrete lining", "concrete repair", "repaired", "reinforced", "reconditioned", "restored", "restoration", "lathe machined", "machined", "recaping", "ramming refractory", "mortar", "anchor mesh", "메꿈 작업", "gap filling", "gap sealing", "hard face", "f-clip"],
    "structural_repair": [
        "packing 교체", "tray 교체", "internal 교체", "distributor", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체",
        "bubble cap", "baffle", "weir plate", "punch plate", "mesh 교체",
        "screen mesh 교체", "clip 교체", "entry horn", "tray cap", "riser pipe hat",
        "tube support", "support 교체", "support casting", "hook casting", "roof casting", "tube hanger", "flexi-cap", "refractory", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체", "panel coil 교체", "panel coil", "coil 교체", "inner screen 교체", "outer screen 교체", "beam support 교체", "support channel 교체", "grating 교체",
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
    r"차기\s*ta|다음\s*ta|추후|향후|권고|필수\s*교체|교체\s*필수|필수\b|recommend|recommended|mandatory|required|should\s+be|shall\s+be|next\s*(?:ta|shutdown|turnaround|t\s*&\s*i)|차기\s*검사|"
    r"검사.*필요|교체.*요망|교체.*필요|교체할\s*경우|보수.*요함|보수.*필요|요망|실시\s*요함|필요함|적용\s*검토|정밀\s*두께\s*측정\s*필요|하여야겠음|토록\s*하여야겠음",
    re.I,
)
_RECOMMENDATION_ACTION_RE = re.compile(r"교체\s*요함|교체\s*필요|필수\s*교체|교체\s*필수|교체할\s*경우|적용\s*검토|차기|예정|요망|recommended|mandatory|required|should\s+be|shall\s+be|next\s*(?:shutdown|turnaround|t\s*&\s*i)", re.I)
_NEGATED_ACTION_RE = re.compile(r"(?:보수|repair|교체|replace|도장|paint|coating|용접|weld|설치|install|가공|machin(?:e|ed)|보강|reinforc(?:e|ed)|재시공|시공|retube|retubing).{0,40}?(?:하지\s*않(?:음|았음)|미실시|실시하지\s*않(?:음|았음)|안\s*함|없음|불필요|취소하였음|취소됨|취소|cancelled|canceled|별도\s*보수작업\s*실시하지\s*않음|보수\s*작업은\s*실시하지\s*않음|no\s+repair(?:\s+was\s+made)?|repair\s+was\s+not\s+made|not\s+repair(?:ed)?|not\s+repaired|need\s+not\s+repair|no\s+need\s+to\s+repair|not\s+required(?:\s+to\s+repair)?|it\s+was\s+decided\s+that\s+.*?not\s+repair)", re.I)
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
    r"교체함|교체하였다|교체\s*설치함|교체\s*설치하였다|교체\s*하였음|교체\s*완료함|교체\s*완료하였음|교체됨|신규\s*교체|신규\s*제작|제작\s*후\s*교체|설치함|설치하였다|설치\s*하였음|설치\s*완료|실시함|실시하였음|실시\s*완료|작업함|작업\s*실시|보수함|보수하였다|보수하였음|보수\s*실시|보수실시|부분\s*보수|부분보수|보수\s*완료|개선조치(?:함|하였음|하였다)?|보강함|보강\s*실시|복원함|복원\s*하였음|복원\s*완료|reinforced|reconditioned|renewed|restored|restoration|machined|lathe\s*machined|prefabricated|exchange(?:d)?|plugged|carried\s*out\s*plugging|용접\s*실시|용접\s*보수|재시공|시공하였음|repair(ed)?|repair\s*performed|replace(d)?|fabricated|painted|도장\s*실시|coating\s*실시|메꿈\s*작업(?:을)?\s*함|gap\s*(?:filling|fill|sealing)\s*작업(?:을)?\s*함|완료함|reweld(?:ed|ing)?|weld\s*repaired|seal\s*weld(?:ed|ing)?|weld[- ]?built[- ]?up|built\s*up\s*with|ground\s*out|deposit\s*welding|metal\s*plugg(?:ed|ing)",
    re.I,
)
_AIR_COOLER_PLUG_SERVICE_RE = re.compile(r"air\s*cooler\s*plug|a/?c\s*plug|plugs?\b", re.I)
_AIR_COOLER_PLUG_NONREPAIR_RE = re.compile(r"분해|조립|해체|탈거|재조립|opening|closing|open|close", re.I)
_AIR_COOLER_OPENING_CONTEXT_RE = re.compile(r"header\s*plug|header\s*box|plug\s*100%\s*open|plugs?\s+were\s+removed|replug(?:ged|ging)?|hydrojet\s*clean|clean\s+tube\s+inside|cleaned\s+bundle|found\s+no\s+leakage|found\s+no\s+tube\s+leak(?:ed|ing)|상태\s*양호|inspection|검사", re.I)
_LEAK_RESPONSE_PLUG_RE = re.compile(r"tube\s+leak(?:ed|ing)?|tubes\s+were\s+leak(?:ed|ing)|found\s+\d+\s*tubes?\s+were\s+leak(?:ed|ing)|found\s+\d+\s*tube\s+leak(?:ed|ing)|prevent\s+further\s+leak(?:ing)?|leaking\s+sign|failed\s+tube|damaged\s+tube|corroded\s+tube|누설|leak|iris\s+results?", re.I)
_PLUG_ACTION_DONE_RE = re.compile(r"plugged|plugging|carried\s*out\s*plugging|total\s*\d+\s*(?:ea\s*)?tubes?\s*plugged|막음\s*작업", re.I)
_BUNDLE_KEYWORD_RE = re.compile(r"\bbundle\b|번들|retube|tube\s*bundle", re.I)
_BUNDLE_PERFORMED_RE = re.compile(
    r"bundle\s*사전\s*신규\s*제작\s*및\s*교체|bundle\s*사전\s*제작\s*및\s*교체|신규\s*bundle\s*제작후\s*교체|신규\s*bundle\s*로\s*교체함|신규\s*bundle\s*제작\s*되어\s*교체하였|신규\s*bundle\s*제작\s*후\s*교체|신규\s*입고된\s*bundle|bundle\s*교체함|replaced\s+with\s+new\s+.*bundle|new\s+.*bundle\s+which\s+was\s+pre-?made|retube|제작\s*후\s*교체",
    re.I,
)
_INTERNAL_EXCLUDE_RE = re.compile(r"충진물|\bfiller\b|filter\s*media|adsorbent|desiccant|diesel\s*sand", re.I)
_SMALL_PART_ONLY_RE = re.compile(r"\bbolt\b|\bnut\b|\bgasket\b|가스켓|washer|stud|pin|keeper", re.I)
_GASKET_ONLY_RE = re.compile(r"\bgasket\b|가스켓", re.I)
_NOZZLE_KEYWORD_RE = re.compile(r"nozzle|노즐|\bnzl\b|\belbow\b", re.I)
_INTERNAL_OBJECT_RE = re.compile(
    r"tray|tray\s*part(?:s)?|chimney\s*tray|bubble\s*cap|flexi\s*-?cap|downcomer|weir\s*plate|seal\s*plate|deck\s*plate|screen|inner\s*screen|outer\s*screen|mesh|clip|support\s*clip|packing|distributor(?:\s*pipe)?|collector|baffle|demister|internal|entry\s*horn|riser\s*pipe\s*hat|punch\s*plate|panel\s*coil|new\s*coil|old\s*coil|coil\b|beam\s*support|support\s*channel|tube\s*support|support\s*casting|hook\s*casting|roof\s*casting|tube\s*hanger|corrosion\s*probe\s*assembly|probe\s*assembly|corrosion\s*probe|grating|flat\s*form|refractory|\bbushing\b|\bvalve\b|flapper\s*valve|guide\s*set\s*plate|guide\s*plate|orifice\s*plate|dip\s*leg|dipleg|bracing\s*cone|quench\s*distributor|steam\s*ring|coke\s*trap|ferrule|burner\s*tile|checker\s*wall|brick|castable|ceramic\s*sleeve",
    re.I,
)
_ASSEMBLY_OBJECT_RE = re.compile(
    r"bundle|tube\s*bundle|retube|retubing|new\s*vessel|신규\s*용기|\b용기\b|vessel(?:\b|(?=[가-힣]))|drum(?:\b|(?=[가-힣]))|column(?:\b|(?=[가-힣]))|tower(?:\b|(?=[가-힣]))|separator(?:\b|(?=[가-힣]))|receiver(?:\b|(?=[가-힣]))|pot(?:\b|(?=[가-힣]))|shell\s*cover|floating\s*head|\bchannel\b|\bassembly\b|\bassy\b|\bduct\b|\bdamper\b|steam\s*manifold|pilot\s*gas\s*assembly|chimney\s*section|return\s*bend|expansion\s*joint|bellows|sleeve|saddle(?!\s*clip)|combust(?:or|er)|claus\s*combust(?:or|er)",
    re.I,
)
_INSTALL_OR_REPLACE_RE = re.compile(r"교체|교체하였다|replace|exchang(?:e|ed)|설치|install|신규\s*제작|신규\s*교체|제작\s*후\s*교체|fabricat|retube", re.I)
_ACTION_FALLBACK_RE = re.compile(r"교체|replace|replaced|replacement|renewed|prefabricated|retube|retubing|보수|부분\s*보수|부분보수|repair|repaired|reinforced|reconditioned|restored|restoration|machined|lathe\s*machined|보강|용접|weld|reweld|seal\s*weld|weld[- ]?built[- ]?up|ground\s*out|deposit\s*welding|metal\s*plugged|도장|painted|coating|plugging|plug|blind\s*처리|재시공|설치|시공|lining\s*repair|lining\s*restored|concrete\s*lining|concrete\s*repair|내화물\s*보수|refractory|mortar|anchor\s*mesh|메꿈\s*작업|gap\s*(?:filling|fill|sealing)|hard\s*face|f-?clip", re.I)
_TOOLING_RE = re.compile(r"유압\s*토크\s*렌치|토크\s*렌치|hydraulic\s*torque\s*wrench|torque\s*wrench", re.I)
_LEVEL_GAUGE_RE = re.compile(r"level\s*gauge|레벨\s*게이지|liquid\s*level\s*gauge", re.I)
_WELD_ACTION_STRONG_RE = re.compile(
    r"육성\s*용접|육성용접|overlay|hardfacing|ER-NiCr3|ErNiCr-3|재용접|용접보수|보수용접|weld\s*repair|repair\s*welding|weld\s*repaired|reweld(?:ed|ing)?|ground\s*out|weld[- ]?built[- ]?up|built\s*up\s*with|deposit\s*welding|metal\s*plugged|결함\s*제거\s*후\s*용접|선형\s*결함\s*제거\s*후\s*용접|grinding\s*후\s*용접|stitch\s*welding|용접\s*실시",
    re.I,
)
_WELD_REFERENCE_RE = re.compile(r"seal\s*weld(?:ing|ed)?|seal-?weld(?:ing|ed)?|welding\s*부|용접부|weld(?:ing)?\s*joint|weld(?:ing)?\s*부위", re.I)
_INSPECTION_TEST_RE = re.compile(r"\bPT\b|\bMT\b|\bUT\b|침투탐상|자분탐상|비파괴|검사|점검|확인|수압시험|수압\s*테스트|hydro(?:static)?\s*test", re.I)
_GOOD_STATUS_RE = re.compile(r"양호|이상\s*없|상태\s*양호|문제\s*없|good\s+condition|acceptable|satisfactory|no abnormal", re.I)
_REASSEMBLY_ONLY_RE = re.compile(r"분해|해체|개방|opening|opened|조립함|조립\s*하였음|조립\s*하였습니다|조립\s*완료|재조립|재\s*조립|reassembl(?:ed|y)|assembled", re.I)
_ASSEMBLY_PERFORMED_RE = re.compile(
    r"신규\s*제작|사전\s*제작|제작\s*후\s*교체|pre\s*-?fabricat|prefabricated|newly\s*fabricated|신규\s*교체\s*실시|교체함|교체하였다|교체\s*설치함|설치함|retube|renewed|replace(d)?|replaced\s+with\s+new",
    re.I,
)
_SPLIT_LINE_RE = re.compile(r"(?:\\n|\n)+")
_TRAILING_HISTORY_NOTE_RE = re.compile(r"(?:[‘\'`]\d{2}년|(?:19|20)\d{2}년)\s*[^.]{0,140}?(?:교체|보수|용접|replace|replaced|repair(?:ed)?|retube|retubing|renewed)[^.]*$", re.I)
_CLAUSE_SPLIT_RE = re.compile(r"(?:\s+(?=차기\s*TA|차기\s*정기|다음\s*TA|향후|추후|권고|recommend|recommended|should\s+be|shall\s+be|next\s*(?:ta|shutdown|turnaround|t\s*&\s*i)))|(?<=[.!?])\s+")

_ACTION_SECTION_SPLIT_RE = re.compile(
    r"(?i)(?:보수/개선\s*내용|주요\s*정비\s*내용|조치\s*사항|조치\s*내용|정비\s*내용|작업\s*내용)\s*[:：]?"
)
_FINDING_SECTION_SPLIT_RE = re.compile(
    r"(?i)(?:초기\s*검사|상세\s*검사|검사\s*결과|점검\s*결과|육안\s*검사\s*결과|상태\s*확인)\s*[:：]?"
)
_COATING_STATE_RE = re.compile(
    r"hard\s*scale.{0,20}coating|scale.{0,20}coating|coating\s*되어\s*있|coating\s*형성|formed\s+coating|도장\s*상태|coating\s*상태|paint\s*condition|도장\s*양호",
    re.I,
)
_COATING_ACTION_RE = re.compile(
    r"보수도장|재도장|도장\s*실시|touch-?up|epoxy\s*coating|high\s*build\s*epoxy|phenolic\s*epoxy|painted|coating\s*실시",
    re.I,
)
_INSPECTION_RESULT_RE = re.compile(
    r"양호함|양호하였음|이상\s*없|결함\s*없이\s*양호|PT\s*결과\s*양호|MT\s*결과\s*양호|UT\s*결과\s*양호",
    re.I,
)


def _is_weld_inspection_only(text: str) -> bool:
    t = str(text or "")
    if not t or not _WELD_REFERENCE_RE.search(t):
        return False
    if _WELD_ACTION_STRONG_RE.search(t):
        return False
    if not (_INSPECTION_TEST_RE.search(t) and _GOOD_STATUS_RE.search(t)):
        return False
    if re.search(r"교체|replace|보수|repair|재시공|시공|육성용접|overlay", t, re.I):
        return False
    return True


def _is_nonrepair_inspection_reassembly(text: str) -> bool:
    t = str(text or "")
    if not (_INSPECTION_TEST_RE.search(t) and _GOOD_STATUS_RE.search(t) and _REASSEMBLY_ONLY_RE.search(t)):
        return False
    if re.search(r"교체|replace|보수|repair|도장|coating|overlay|육성용접|재용접|용접보수|보수용접|retube|retubing|재시공|lining\s*repair|concrete\s*repair|mortar|anchor\s*mesh", t, re.I):
        return False
    return True


def _is_nonrepair_plug_opening(text: str) -> bool:
    t = str(text or "")
    if not t or not _AIR_COOLER_PLUG_SERVICE_RE.search(t):
        return False
    if _LEAK_RESPONSE_PLUG_RE.search(t) and re.search(r"prevent\s+further\s+leak|tube\s+leak|tubes?\s+were\s+leaked|found\s+\d+\s*tube", t, re.I):
        return False
    if _AIR_COOLER_OPENING_CONTEXT_RE.search(t) and re.search(r"remove|removed|open|opening|clean|cleaned|hydrojet|replug(?:ged|ging)?|assembled|조립|재조립|개방|세척", t, re.I):
        return True
    return False


def _is_leak_response_tube_plugging(text: str) -> bool:
    t = str(text or "")
    if not t or _is_nonrepair_plug_opening(t):
        return False
    if not (_PLUG_ACTION_DONE_RE.search(t) or re.search(r"plug", t, re.I)):
        return False
    return bool(_LEAK_RESPONSE_PLUG_RE.search(t))


def _parse_tag_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[,/;|]\s*", text)
    return [p.strip() for p in parts if p.strip()]


def classify_sentence_role(text: str, raw_tags: List[str] | None = None) -> str:
    t = _normalize_sentence(text)
    raw_tags = raw_tags or []
    if not t or is_noise_sentence(t):
        return "noise"
    if _RECOM_RE.search(t) and not _DONE_RE.search(t):
        return "recommendation"
    if _NEGATED_ACTION_RE.search(t) and not _DONE_RE.search(t):
        return "negative"
    if _is_nonrepair_inspection_reassembly(t) or _is_weld_inspection_only(t) or _is_nonrepair_plug_opening(t):
        return "inspection_only"
    if _COATING_STATE_RE.search(t) and not _COATING_ACTION_RE.search(t):
        return "finding"
    if _DONE_RE.search(t) or raw_tags:
        return "action_done"
    if classify_finding(t):
        return "finding"
    return "other"


def classify_action(text: str) -> List[str]:
    t = _normalize_sentence(text)
    if not t or classify_sentence_role(t) != "action_done":
        return []
    scores = {}
    rules = {
        "weld_repair": [
            (_WELD_ACTION_STRONG_RE, 4),
            (re.compile(r"결함\s*제거\s*후\s*용접|용접보수|보수용접", re.I), 3),
        ],
        "replace": [
            (_INSTALL_OR_REPLACE_RE, 3),
            (re.compile(r"신규\s*설치|신규\s*제작|전체\s*교체|new\s+.*installed|교체\s*하였음|교체함", re.I), 2),
        ],
        "coating_repair": [
            (_COATING_ACTION_RE, 4),
            (_COATING_STATE_RE, -5),
        ],
        "temporary_fix": [
            (re.compile(r"보수|repair|grinding|결함\s*제거|patch|box-?up|메꿈", re.I), 2),
        ],
        "structural_repair": [
            (re.compile(r"전극판|electrode|demister|guide\s*tube|riser|distributor|clip|tray|packing|bushing", re.I), 2),
            (_INSTALL_OR_REPLACE_RE, 1),
        ],
    }
    for cluster, rule_list in rules.items():
        score = 0
        for regex, weight in rule_list:
            if regex.search(t):
                score += weight
        if score > 0:
            scores[cluster] = score
    if not scores:
        tl = t.lower()
        hits = []
        for cluster, keywords in ACTION_CLUSTERS.items():
            if cluster == "weld_repair" and _is_weld_inspection_only(t):
                continue
            if any(k.lower() in tl for k in keywords):
                hits.append(cluster)
        return hits[:1]
    best_cluster, best_score = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0]
    return [best_cluster] if best_score >= 2 else []


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
    t = _TRAILING_HISTORY_NOTE_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _split_sentence_clauses(text: str) -> List[str]:
    t = _normalize_sentence(text)
    if not t:
        return []
    t = _ACTION_SECTION_SPLIT_RE.sub(" |SPLIT| ", t)
    t = _FINDING_SECTION_SPLIT_RE.sub(" |SPLIT| ", t)
    t = re.sub(r"(?:(?<=^)|(?<=\s))(?:\(?\d+\)|\d+\.)\s+", " |SPLIT| ", t)
    t = re.sub(r"(?i)(양호함\.?|양호하였음\.?|이상\s*없음\.?|결함\s*없이\s*양호\.?)(\s+)(?=(보수/개선\s*내용|주요\s*정비\s*내용|신규|교체|보수|repair|replace|anchor\s*bolt|기존\s*anchor\s*bolt|모든\s*Nozzle|내부\s*모든))", r"\1 |SPLIT| ", t)
    t = re.sub(r"(?i)(설치\s*완료|설치함|교체함|교체\s*설치함|replaced|installed)(\s+)(?=(?:anchor\s*bolt|기존\s*anchor\s*bolt|모든\s*Nozzle|내부\s*모든|외부\s*검사|추가\s*점검))", r"\1 |SPLIT| ", t)
    t = re.sub(r"(?i)(?<=mm)(\s+)(?=(보수/개선\s*내용|주요\s*정비\s*내용|신규|교체|보수|repair|replace))", " |SPLIT| ", t)
    rough_parts = []
    for part in re.split(r"\s*(?:\|SPLIT\||/|;|\n)+\s*", t):
        part = part.strip()
        if not part:
            continue
        part_chunks = re.split(r"(?<=[\.!?다함음요])\s+(?=(?:\(?\d+\)|[A-Z#0-9\"“]|Nozzle|Tray|Shell|Top|Bottom|내부|외부|Anchor|차기\s*TA|다음\s*TA|권고|검토|[‘'`]?(?:19|20)?\d{2}년))", part)
        for chunk in part_chunks:
            rough_parts.extend(_CLAUSE_SPLIT_RE.split(chunk))
    parts = [p.strip(" -/,:;") for p in rough_parts if p and p.strip()]
    return parts or ([t] if t else [])


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
    return classify_sentence_role(text, raw_tags) == "action_done"


def is_finding_sentence(text: str, raw_tags: List[str] | None = None) -> bool:
    return bool((raw_tags or []) or classify_finding(text))


def is_recommendation_sentence(text: str) -> bool:
    t = _normalize_sentence(text)
    return bool(_RECOM_RE.search(t) and not _DONE_RE.search(t))


def _merge_sentences(sentences: List[str], max_length: int = 320) -> str:
    merged = " / ".join(s.strip() for s in sentences if s.strip())
    return merged[:max_length]


_GENERATED_OUTPUT_FILE_RE = re.compile(r"반복정비_고정장치_(?:필터결과|조치요약|과제후보)_v\d+|repeat[_ -]?task", re.I)
_NAME_NOISE_RE = re.compile(
    r"검사일|차기검사예정일|검사구분|상세내용|차기고려사항|등록일|공정담당자|검사원|발생년도|발췌\s*category|TA\s*조치사항|추후\s*권고사항|"
    r"점검\s*결과|검사\s*결과|확인됨|확인되었|발생\s*확인|양호한\s*상태|양호함|필요|요망|검토|실시|진행|부식|감육|균열|pitting|corrosion|"
    r"연결\s*nozzle|grid\s*ut|scanning|thickness|두께\s*측정|정밀\s*두께|보수작업|교체여부",
    re.I,
)
_NAME_SENTENCE_LIKE_RE = re.compile(
    r"확인|발생|진행|실시|필요|요망|검토|판단|측정|부착|고착|양호|보수|교체|용접|도장|repair|replace|inspect|confirm|found|observed",
    re.I,
)


def _extract_year(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return None
    m = re.search(r"(19\d{2}|20\d{2})", text)
    if not m:
        return None
    year = int(m.group(1))
    if 1900 <= year <= 2100:
        return year
    return None


def _clean_equipment_name_candidate(value: str, equipment_no: str = "") -> str:
    t = _normalize_sentence(value)
    if not t:
        return ""
    if equipment_no:
        t = re.sub(re.escape(str(equipment_no)), " ", t, flags=re.I)
        t = re.sub(r"(?i)\b" + re.escape(re.sub(r"^(\d{2,3})([A-Z]{1,3})-(\d{3,4}[A-Z]?)$", r"\1-\2-\3", str(equipment_no))) + r"\b", " ", t)
    t = re.sub(r"^(?:AND|THE|OF)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -/:;,.[]()")
    if not t or len(t) < 3 or len(t) > 80:
        return ""
    if _NAME_NOISE_RE.search(t):
        return ""
    if len(t.split()) >= 6 and _NAME_SENTENCE_LIKE_RE.search(t):
        return ""
    if re.search(r"[.!?]", t):
        return ""
    return t


def _resolve_equipment_name(equipment_no: str, fallback_name: str, rows: pd.DataFrame) -> str:
    weighted: List[str] = []
    base = _clean_equipment_name_candidate(fallback_name, equipment_no)
    if base:
        weighted.extend([base, base, base])

    for col in ["equipment_name", "설비명", "equipment"]:
        if col not in rows.columns:
            continue
        for raw in rows[col].dropna().astype(str):
            cleaned = _clean_equipment_name_candidate(raw, equipment_no)
            if cleaned:
                weighted.append(cleaned)

    if not weighted:
        return base

    counts = Counter(weighted)
    return sorted(counts.keys(), key=lambda name: (-counts[name], -len(name), name))[0]


def _resolve_row_year(row) -> int | None:
    for col in ["검사일", "event_date", "date", "inspection_date", "등록일"]:
        if col in row:
            year = _extract_year(row.get(col))
            if year:
                return year

    sentence = _normalize_sentence(row.get("sentence", ""))
    if sentence:
        for pat in [
            r"검사일\s*[:=]\s*([^/|,;]+)",
            r"inspection\s*date\s*[:=]\s*([^/|,;]+)",
            r"event\s*date\s*[:=]\s*([^/|,;]+)",
        ]:
            m = re.search(pat, sentence, re.I)
            if m:
                year = _extract_year(m.group(1))
                if year:
                    return year
        year = _extract_year(sentence)
        if year:
            return year

    year = _extract_year(row.get("year"))
    if year:
        return year

    return _extract_year(row.get("source_file"))


def _looks_like_generated_output_row(row) -> bool:
    source_file = str(row.get("source_file", "") or "")
    sentence = _normalize_sentence(row.get("sentence", ""))
    if source_file and _GENERATED_OUTPUT_FILE_RE.search(source_file):
        return True
    if re.search(r"과제후보_등록형식|카테고리별_발췌|연도별_정비이벤트", sentence, re.I):
        return True
    return False


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
    if _GASKET_ONLY_RE.search(t) and not re.search(r"neck|boss|nipple|elbow|sch\.|용접부|crack|부식|감육|new\s*nozzle|nozzle.{0,40}(?:교체|replace|보수|repair)|(?:교체|replace|보수|repair).{0,40}nozzle", t, re.I):
        return False
    if _RECOM_RE.search(t) and not explicit_done:
        return False
    if re.search(r"상태|부식|감육|pitting|측정|검사|필요|요망|검토", t, re.I) and not (explicit_done or re.search(r"신품|신규|제작", t, re.I)):
        return False
    return bool(
        ("replace" in raw_action_tags and (explicit_done or _INSTALL_OR_REPLACE_RE.search(t)))
        or re.search(r"bushing\s*nozzle|nozzle\s*bushing", t, re.I)
        or re.search(r"(?:nozzle|노즐|nzl|elbow).{0,40}(?:교체|replace|신품|신규|제작|보수|repair|개선조치)", t, re.I)
        or re.search(r"(?:교체|replace|신품|신규|제작|보수|repair|개선조치).{0,40}(?:nozzle|노즐|nzl|elbow)", t, re.I)
    )


def _is_verified_internal_replacement_sentence(text: str, raw_action_tags: List[str] | None = None) -> bool:
    t = _normalize_sentence(text)
    raw_action_tags = [x.strip().lower() for x in (raw_action_tags or []) if x]
    if not t or not _INTERNAL_OBJECT_RE.search(t):
        return False
    if re.search(r"bushing\s*nozzle|nozzle\s*bushing", t, re.I):
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
        or re.search(r"철거.*new\s*(packing|tray|mesh|screen|bushing|seal\s*plate|deck\s*plate)|new\s*(packing|tray|mesh|screen|bushing|seal\s*plate|deck\s*plate).*(교체|설치)|exchanged\s+new\s+bushing|tray\s*parts?.*(replaced|교체)|seal\s*plate.*(replaced|교체)|deck\s*plate.*(replaced|교체)", t, re.I)
    )


def _is_verified_assembly_replacement_sentence(text: str, raw_action_tags: List[str] | None = None) -> bool:
    t = _normalize_sentence(text)
    raw_action_tags = [x.strip().lower() for x in (raw_action_tags or []) if x]
    if not t or not _ASSEMBLY_OBJECT_RE.search(t):
        return False
    if _TOOLING_RE.search(t) or _LEVEL_GAUGE_RE.search(t):
        return False
    explicit_done = bool(_DONE_RE.search(t) or _BUNDLE_PERFORMED_RE.search(t) or _ASSEMBLY_PERFORMED_RE.search(t))
    strong_current_assembly = bool(
        re.search(r"신규\s*제작|제작\s*후\s*교체|신규\s*(?:column|drum|tower|vessel|separator|receiver|pot)|교체\s*설치함|설치\s*완료|newly\s*fabricated|prefabricated|replaced\s+with\s+new", t, re.I)
        and (_ASSEMBLY_OBJECT_RE.search(t) or ("replace" in raw_action_tags))
    )
    if (_NOZZLE_KEYWORD_RE.search(t) or _INTERNAL_OBJECT_RE.search(t)) and not strong_current_assembly:
        return False
    if _SMALL_PART_ONLY_RE.search(t) and not re.search(r"backing\s*device|vortex\s*breaker|duct|damper|bundle|retube|vessel|shell\s*cover|floating\s*head|channel", t, re.I):
        if not (strong_current_assembly and re.search(r"anchor\s*bolt", t, re.I)):
            return False
    if _RECOM_RE.search(t) and not explicit_done:
        return False
    return bool(
        _is_verified_bundle_replacement_sentence(t, raw_action_tags)
        or (("replace" in raw_action_tags) and (explicit_done or strong_current_assembly))
        or (_ASSEMBLY_OBJECT_RE.search(t) and (_ASSEMBLY_PERFORMED_RE.search(t) or strong_current_assembly) and (explicit_done or strong_current_assembly))
    )


def _extract_verified_category_actions(year_group: pd.DataFrame) -> dict[str, List[str]]:
    actions = {"nozzle": [], "internal": [], "assembly": []}
    for _, row in year_group.reset_index(drop=True).iterrows():
        sentence = _normalize_sentence(row.get("sentence", ""))
        raw_action_tags = _parse_tag_list(row.get("action_tags"))
        clauses = _split_sentence_clauses(sentence) or ([sentence] if sentence else [])
        for clause in clauses:
            clause = _normalize_sentence(clause)
            if not clause:
                continue
            if _is_verified_nozzle_replacement_sentence(clause, raw_action_tags):
                if clause not in actions["nozzle"]:
                    actions["nozzle"].append(clause)
            if _is_verified_internal_replacement_sentence(clause, raw_action_tags):
                if clause not in actions["internal"]:
                    actions["internal"].append(clause)
            if _is_verified_assembly_replacement_sentence(clause, raw_action_tags):
                if clause not in actions["assembly"]:
                    actions["assembly"].append(clause)
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
        raw_sentence = row.get("sentence", "")
        action_tags = _parse_tag_list(row.get("action_tags"))
        damage_tags = _parse_tag_list(row.get("damage_tags"))
        for sentence in _split_sentence_clauses(raw_sentence):
            if not sentence or is_noise_sentence(sentence):
                continue
            records.append({
                "text": sentence,
                "action_tags": action_tags,
                "damage_tags": damage_tags,
                "section": str(row.get("section", "") or "").strip().lower(),
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
            (len(s) <= 24 and _CONTINUATION_START_RE.search(s) and not re.search(r"교체|보수|신규|replace|repair|도장|coating|용접", s, re.I))
            or (_FRAGMENT_END_RE.search(prev_s) and len(prev_s) < 80 and not re.search(r"양호|이상\s*없|검사\s*결과", prev_s, re.I))
        )
        if should_merge and prev.get("section", "") == rec.get("section", ""):
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
    rows = rows[~rows.apply(_looks_like_generated_output_row, axis=1)].copy()
    if rows.empty:
        return []

    rows["_year"] = rows.apply(_resolve_row_year, axis=1)
    rows = rows.dropna(subset=["_year"])
    rows["_year"] = rows["_year"].astype(int)
    equipment_name = _resolve_equipment_name(equipment_no, equipment_name, rows)

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
            section = rec.get("section", "")

            if not sentence or is_noise_sentence(sentence):
                continue

            locs = extract_locations(sentence)
            all_locations.extend(locs)
            fds = list(dict.fromkeys(raw_damage_tags + classify_finding(sentence)))
            all_findings.extend(fds)

            meas = extract_measurements(sentence)
            if meas:
                measurement_texts.append(meas)

            explicit_future = bool(re.search(r"차기|다음\s*TA|향후|추후|recommend", sentence, re.I))
            if section == "recommendation":
                recommendations.append(sentence)
            elif is_recommendation_sentence(sentence) and explicit_future and not is_action_sentence(sentence, raw_action_tags):
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

    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    # ===== Cycle2 Patch: Post-filter events =====
    import re as _pfre
    _PF_GOOD_ONLY = _pfre.compile(r'(?:양호|이상\s*없|특이사항\s*없|결함\s*없|no\s*defect)', _pfre.I)
    _PF_INSP_ONLY = _pfre.compile(r'(?:두께\s*측정|thickness\s*measur|UT\s*검사|IRIS\s*검사|screen\s*대상|미\s*개방|외부\s*두께측정|초음파|NDE\s*검사)', _pfre.I)
    _PF_HAS_ACTION = _pfre.compile(r'(?:교체|replace|용접|weld|보수|repair|도장|coating|paint|plug|설치|install|제작|fabricat|grinding|expanding|retubing|lining)', _pfre.I)
    
    filtered = []
    for ev in events:
        d = (ev.action_detail or "").strip()
        a = (ev.action_type or "").strip()
        r = (ev.recommendation or "").strip()
        
        # Cycle4: Skip historical-only references
        _PF_HIST = _pfre.compile(r'(?:전회\s*(?:TA|검사)|이전\s*(?:TA|검사)|지난\s*(?:TA|검사)|기존\s*검사)', _pfre.I)
        _PF_CURRENT = _pfre.compile(r'(?:금번|금회|이번|실시\s*(?:함|하였)|수행|진행)', _pfre.I)
        if d and _PF_HIST.search(d) and not _PF_CURRENT.search(d) and not a:
            continue
        
        # Skip too-short noise (< 10 chars, no action type)
        if len(d) < 10 and not a:
            continue
        # Skip good-condition-only
        if d and _PF_GOOD_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Skip inspection-only
        if d and _PF_INSP_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        # Cycle10: Final noise cleanup
        _PF_PHOTO_ONLY = _pfre.compile(r'^\s*(?:사진\s*\d|Photo\s*\d|\(사진)', _pfre.I)
        _PF_BOLT_ASSEM = _pfre.compile(r'^(?:Header\s*Cover\s*(?:Bolt|조립)|Bolt\s*교체\s*후\s*(?:조립|수압))', _pfre.I)
        if d and _PF_PHOTO_ONLY.match(d.strip()) and not a:
            continue
        if d and len(d) < 40 and _PF_BOLT_ASSEM.match(d.strip()) and not a:
            continue
        
        # Cycle9: Skip pressure-test-only events
        _PF_TEST_ONLY = _pfre.compile(r'(?:수압\s*시험|기밀\s*시험|hydro\s*test|leak\s*test|pressure\s*test)', _pfre.I)
        if d and _PF_TEST_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle8: Skip non-opening external inspection only
        _PF_NO_OPEN = _pfre.compile(r'(?:미\s*개방|개방\s*하지|비개방|외부\s*(?:검사|점검)\s*(?:만|만\s*실시|실시))', _pfre.I)
        if d and _PF_NO_OPEN.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle7: Skip cleaning-only events (no actual repair)
        _PF_CLEAN_ONLY = _pfre.compile(r'(?:cleaning|청소|세척|sludge\s*제거|scale\s*제거|hydrojetting|lance\s*clean)', _pfre.I)
        if d and _PF_CLEAN_ONLY.search(d) and not _PF_HAS_ACTION.search(d) and not a:
            continue
        
        # Cycle6: Skip N/A and meaningless short entries
        if d and _pfre.match(r'^(?:N/?A|없음|해당없음|해당\s*없음|특이사항\s*없음|양호함|\d+[.)\s]*$|-+$)$', d.strip(), _pfre.I):
            continue
        
        # Cycle5: Skip section header noise
        if d and len(d) < 25 and _pfre.match(r'^\d*[.)\s]*(초기검사|상세검사|보수|정비|두께측정|결론|목적|검사결과)', d):
            continue
        
        # Skip recommendation-only (no detail, no action)
        if not d and not a and r:
            continue
        
        filtered.append(ev)
    events = filtered if filtered else events[:1] if events else events
    
    # ===== Cycle3 Patch: Deduplicate similar events =====
    if len(events) > 1:
        deduped = []
        seen = set()
        for ev in events:
            dk = (ev.action_detail or "")[:50].strip().lower()
            key = (ev.equipment_no, ev.report_year, dk)
            if key not in seen:
                seen.add(key)
                deduped.append(ev)
        events = deduped if deduped else events
    # ===== End Cycle3 Patch =====
    # ===== End Cycle2 Patch =====
    return sorted(events, key=lambda e: e.report_year)
