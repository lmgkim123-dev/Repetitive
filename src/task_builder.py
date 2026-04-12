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
    "발췌 Category",
    "TA 조치사항", "추후 권고사항",
    "제목", "상세 내용", "검토필요여부",
]

_INTERNAL_EXCLUDE_RE = re.compile(
    r"충진물|\bfiller\b|filter media|\bmedia\b replacement|adsorbent|desiccant|diesel\s*sand",
    re.I,
)
_SMALL_PART_EXCLUDE_RE = re.compile(
    r"\bbolt\b|\bnut\b|\bgasket\b|가스켓|test\s*ring|collar\s*bolt|floating\s*head\s*bolt|\bf/h\s*bolt\b|keeper|\bpin\b(?!\s*hole)|valve\s*wheel|stud|washer|\banchor\b(?!\s*mesh)",
    re.I,
)
_GASKET_ONLY_RE = re.compile(r"\bgasket\b|가스켓", re.I)
_INTERNAL_PART_RE = re.compile(
    r"mesh|screen|hold\s*-?down|holdown|clip|saddle\s*clip|grid\s*clip|tray(?:\s*part(?:s)?)?|packing|bubble\s*cap|flexi\s*-?cap|tray\s*deck|tray\s*plate|packing\s*clip|baffle|weir\s*plate|demister|internal(?:\s*fittings?)?|seal\s*pan|seal\s*plate|deck\s*plate|entry\s*horn|distributor(?:\s*pipe)?|collector|tray\s*cap|riser\s*pipe\s*hat|punch\s*plate|corrosion\s*probe\s*assembly|probe\s*assembly|corrosion\s*probe|heater\s*tube\s*support|radiant\s*tube\s*support|tube\s*casting\s*support|support\s*casting|hook\s*casting|roof\s*casting|tube\s*hanger|casting\s*support|tube\s*support|vortex\s*breaker|strainer|\bbushing\b|\bvalve\b|flapper\s*valve|guide\s*set\s*plate|guide\s*plate|orifice\s*plate|dip\s*leg|dipleg|bracing\s*cone|quench\s*distributor|steam\s*ring|coke\s*trap|refractory|ferrule|burner\s*tile|checker\s*wall|brick|castable|ceramic\s*sleeve",
    re.I,
)

_COATING_ONLY_SIMPLE_RE = re.compile(
    r"coating\s*repair|coating\s*보수|coating\s*실시|coat(?:ing|ed)|PLISTIX|PRITIX|PLIRAM|PRIRAM|gunning|castable|내화물.{0,20}coating|도장\s*실시|재도장|보수도장",
    re.I,
)
_NONCOATING_SIMPLE_RE = re.compile(
    r"patch-?up|box-?up|compound\s*sealing|hand\s*packing|plug(?:ging|ged)?|막음\s*작업|leak|누설|anchor\s*mesh|용접|weld|grind(?:ing)?|machin(?:e|ed|ing)|메꿈\s*작업|gap\s*(?:filling|fill|sealing)|hard\s*face|복원|복구|교체",
    re.I,
)
_NOZZLE_RE = re.compile(r"nozzle|노즐|\bnzl\b|\belbow\b", re.I)
_ASSEMBLY_OBJ_RE = re.compile(
    r"new\s*vessel|신규\s*용기|\b용기\b|vessel(?:\b|(?=[가-힣]))|drum(?:\b|(?=[가-힣]))|column(?:\b|(?=[가-힣]))|tower(?:\b|(?=[가-힣]))|separator(?:\b|(?=[가-힣]))|receiver(?:\b|(?=[가-힣]))|pot(?:\b|(?=[가-힣]))|\bbundle\b|retube|shell\s*cover|floating\s*head|\bchannel\b|top\s*head|bottom\s*head|\bassembly\b|\bassy\b|\bduct\b|\bdamper\b|steam\s*manifold|pilot\s*gas\s*assembly|chimney\s*section|return\s*bend|expansion\s*joint|bellows|saddle(?!\s*clip)|combust(?:or|er)|claus\s*combust(?:or|er)",
    re.I,
)
_ASSEMBLY_CONTEXT_RE = re.compile(r"신규\s*제작|사전\s*제작|제작\s*후\s*교체|new|fabricat|retube|retubing|전체\s*교체|assy|assembly|신품\s*교체|pre\s*-?fabricat|bellows|sleeve", re.I)
_COATING_RE = re.compile(r"phenolic\s*epoxy|coating(?!\s*상태)|paint(?!\s*상태)|painted|painting|도장(?!상태)|보수도장|재도장|touch-?up", re.I)
_COATING_STATE_RE = re.compile(
    r"hard\s*scale.{0,20}coating|scale.{0,20}coating|coating\s*되어\s*있|coating\s*형성|formed\s+coating|도장\s*상태|coating\s*상태|paint\s*condition|도장\s*양호",
    re.I,
)
_COATING_DONE_RE = re.compile(
    r"보수도장|재도장|도장\s*실시|touch-?up|high\s*build\s*epoxy|phenolic\s*epoxy|painted|coating\s*실시",
    re.I,
)
_PRIMARY_INTERNAL_OBJ_RE = re.compile(
    r"전극판|electrode|demister|guide\s*tube|riser|distributor|clip|tray|packing|bushing|seal\s*plate|deck\s*plate",
    re.I,
)
_BLAST_ONLY_RE = re.compile(r"sand\s*blasting|sandblasting", re.I)
_OVERLAY_RE = re.compile(r"육성\s*용접|육성\s*용접|육성용접|overlay|hardfacing|build[- ]?up\s*weld|erni-?cr-?3|er-?nicr-?3|용접보수|보수용접|erni-?cr-?3|er-?nicr-?3|용접보수|보수용접", re.I)
_WELD_REPAIR_RE = re.compile(
    r"seal\s*weld(?:ing|ed)?|seal-?weld(?:ing|ed)?|stitch\s*weld(?:ing|ed)?|weld\s*repair(?:ed)?|repair\s*weld(?:ing|ed)?|weld\s*repaired|용접보수|보수용접|재\s*용접|재용접|결함\s*제거\s*후\s*용접|선형\s*결함\s*제거\s*후\s*용접|grinding\s*후\s*용접|ground\s*out|grind(?:ing)?\s*out|reweld(?:ed|ing)?|weld[- ]?built[- ]?up|built\s*up\s*with|deposit\s*welding|metal\s*plugg(?:ed|ing)|tig\s*weld(?:ing)?|용접\s*실시|육성\s*용접|용접\s*후\s*나사산|용접\s*후\s*.*가공|용접\s*후\s*.*탐상",
    re.I,
)
_SIMPLE_REPAIR_RE = re.compile(
    r"보수|부분\s*보수|부분보수|repair|repaired|reinforc(?:e|ed|ing)|recondition(?:ed|ing)?|rebuild|rebuilt|restor(?:e|ed|ation)?|machin(?:e|ed|ing)|lathe\s*machin(?:e|ed)|grinding|결함\s*제거|defect\s*remov|patch|patch-?up|보강|anchor\s*mesh|anchor\s*mesh\s*보강|recap(?:ed|ing)?|refractory\s*repair|ramming\s*refractory|mortar|재시공|시공|임시조치|box-?up|compound\s*sealing|lining\s*repair|lining\s*restor(?:e|ed|ation)?|concrete\s*lining|내화물\s*보수|concrete\s*repair|보수\s*완료|plug\b|plugging|plugged|carried\s*out\s*plugging|unplugging|stop\s*hole|막음\s*작업|막음\s*용접|개선조치|복원|복구|복원\s*후\s*조립|메꿈\s*작업|gap\s*(?:filling|fill|sealing)|hard\s*face",
    re.I,
)
_REPAIR_ACTION_RE = re.compile(
    r"grinding|결함\s*제거|부분\s*제거\s*후\s*보수|손상.*보수|defect\s*remov|patch-?up|patch|box-?up|compound\s*sealing|보수함|보수하였음|보수하였습니다|보수하였다|보수\s*실시|보수실시|부분\s*보수|부분보수|보수\s*완료|보강함|보강\s*실시|anchor\s*mesh\s*보강|reinforc(?:e|ed|ing)|recondition(?:ed|ing)?|restor(?:e|ed|ation)?|복원함|복원\s*하였음|복원\s*완료|복구함|복구\s*실시|machin(?:e|ed|ing)|lathe\s*machin(?:e|ed)|recap(?:ed|ing)?|ramming\s*refractory|mortar|재시공|시공하였음|plug\b|plugging|plugged|carried\s*out\s*plugging|unplugging|stop\s*hole|막음\s*작업|막음\s*용접|repair(ed)?|repair\s*performed|개선조치(?:함|하였음|하였습니다|하였다)?|보수\s*작업\s*실시|lining\s*repair|lining\s*restor(?:e|ed|ation)?|concrete\s*repair|concrete\s*lining|메꿈\s*작업(?:을)?\s*함|메꿈\s*작업|gap\s*(?:filling|fill|sealing)|hard\s*face",
    re.I,
)
_REPLACE_RE = re.compile(r"교체|교체하였다|교체하였음|replace|replaced|replacement|exchange|exchanged|renew(?:ed|al)?|newly\s*fabricated|prefabricated|reconditioned\s*with\s*new\s*one|replaced\s*with\s*new\s*ones|신규\s*제작|신규\s*교체|제작\s*후\s*교체|fabricated?.*replace|retube|retubed|retubing|re-?tubing|made\s+new|made\s+a\s+new|newly\s*installed|installed|교체\s*설치함|교체\s*완료|교체\s*완료함|교체\s*완료하였음", re.I)
_ACTION_DONE_RE = re.compile(
    r"교체함|교체하였다|교체\s*설치함|교체\s*설치하였다|교체\s*하였음|교체\s*하였습니다|교체하고|교체하여|교체\s*완료함|교체\s*완료하였음|교체\s*완료하였습니다|교체됨|신규\s*교체|신규\s*제작|제작\s*후\s*교체|설치함|설치하였다|설치\s*하였음|설치\s*하였습니다|설치\s*완료|실시함|실시하였음|실시하였습니다|실시\s*완료|작업함|작업\s*실시|보수함|보수하였다|보수하였음|보수하였습니다|보수\s*실시|보수실시|부분\s*보수|부분보수|보수\s*완료|보강함|보강\s*실시|복원함|복원\s*하였음|복원\s*완료|복구함|재조립|재\s*조립|조립함|조립\s*하였음|조립\s*하였습니다|조립\s*완료|결합\s*작업\s*실시|재축조|재축조하였음|재축조하였습니다|마감\s*처리함|개선조치(?:함|하였음|하였습니다|하였다)|reinforc(?:e|ed|ing)|recondition(?:ed|ing)?|restor(?:e|ed|ation)?|machin(?:e|ed|ing)|lathe\s*machin(?:e|ed)|renew(?:ed|al)?|prefabricated|newly\s*fabricated|replaced\s*with\s*new\s*ones|exchange(?:d)?|reassembl(?:ed|y)|assembled|installed|modified|plugged|carried\s*out\s*plugging|용접\s*실시|blind\s*처리|by-?pass\s*시킴|재시공|재시공\s*하였음|재시공\s*하였습니다|시공하였음|시공하였습니다|repair(ed)?|repair\s*performed|replace(d)?|fabricated|coating\s*실시|painted|도장\s*실시|메꿈\s*작업(?:을)?\s*함|gap\s*(?:filling|fill|sealing)\s*작업(?:을)?\s*함|완료함|reweld(?:ed|ing)?|weld\s*repaired|seal\s*weld(?:ed|ing)?|weld[- ]?built[- ]?up|built\s*up\s*with|ground\s*out|deposit\s*welding|metal\s*plugg(?:ed|ing)",
    re.I,
)
_DONE_FALLBACK_RE = re.compile(r"교체하고|교체하여|교체\s*하였(?:음|습니다)|보수\s*하였(?:음|습니다)|개선조치(?:함|하였(?:음|습니다)|하였다)?|조립\s*하였(?:음|습니다)|설치(?:하였다|\s*하였(?:음|습니다))|시공\s*하였(?:음|습니다)|재시공\s*하였(?:음|습니다)|재축조\s*하였(?:음|습니다)|마감\s*처리함|Close\s*진행|메꿈\s*작업(?:을)?\s*함|gap\s*(?:filling|fill|sealing)\s*작업(?:을)?\s*함", re.I)
_AIR_COOLER_PLUG_SERVICE_RE = re.compile(r"air\s*cooler\s*plug|a/?c\s*plug|plugs?\b", re.I)
_AIR_COOLER_PLUG_NONREPAIR_RE = re.compile(r"분해|조립|해체|탈거|재조립|opening|closing|open|close", re.I)
_AIR_COOLER_OPENING_CONTEXT_RE = re.compile(r"header\s*plug|header\s*box|plug\s*100%\s*open|removed\s+.*plug|replug(?:ged|ging)?|hydrojet\s*clean|clean\s+tube\s+inside|cleaned\s+bundle|found\s+no\s+leakage|found\s+no\s+tube\s+leak(?:ed|ing)|상태\s*양호|no\s+leakage|inspection|검사", re.I)
_LEAK_RESPONSE_PLUG_RE = re.compile(r"tube\s+leak(?:ed|ing)?|tubes\s+were\s+leak(?:ed|ing)|found\s+\d+\s*tubes?\s+were\s+leak(?:ed|ing)|found\s+\d+\s*tube\s+leak(?:ed|ing)|prevent\s+further\s+leak(?:ing)?|leaking\s+sign|failed\s+tube|damaged\s+tube|corroded\s+tube|누설|leak|iris\s+results?", re.I)
_PLUG_ACTION_DONE_RE = re.compile(r"plugged|plugging|carried\s*out\s*plugging|total\s*\d+\s*(?:ea\s*)?tubes?\s*plugged|막음\s*작업", re.I)
_RECOMMEND_ONLY_RE = re.compile(r"요망|요함|필요|권고|필수\s*교체|교체\s*필수|필수\b|차기\s*T/?A|다음\s*T/?A|향후|추후|recommend(?:ed)?|recommended\s+that|it\s+is\s+recommended|strongly\s+recommended|mandatory|required|will\s+be\s+closely\s+inspected|should\s+be|shall\s+be|must\s+be|need(?:s)?\s+(?:to\s+be\s+|to\s+)?(?:repair(?:ed)?|replace(?:d)?|renew(?:ed)?|install(?:ed)?|weld(?:ed)?)|requires?\s+(?:repair|replacement|renewal|installation)|planned\s+for|planned\s+at|scheduled\s+for|scheduled\s+to|next\s*(?:shutdown|turnaround|t\s*&\s*i)|subsequent\s*t\s*&\s*i|검토|적용\s*검토|교체할\s*경우|실시하여야|실시\s*하여야|하여야\s*겠음|해야\s*겠음|토록\s*하여야겠음", re.I)
_FUTURE_SCOPE_RE = re.compile(
    r"차기\s*T/?A|다음\s*T/?A|향후|추후|필수\s*교체|교체\s*필수|필수\b|recommend(?:ed)?|recommended\s+that|it\s+is\s+recommended|strongly\s+recommended|mandatory|required|should\s+be|shall\s+be|must\s+be|need(?:s)?\s+(?:to\s+be\s+|to\s+)?(?:repair(?:ed)?|replace(?:d)?|renew(?:ed)?|install(?:ed)?|weld(?:ed)?)|requires?\s+(?:repair|replacement|renewal|installation)|planned\s+for|planned\s+at|scheduled\s+for|scheduled\s+to|during\s+the\s+next|at\s+next\s*(?:s\.?d\.?|shutdown|turnaround|t\s*&\s*i)|next\s*(?:s\.?d\.?|shutdown|turnaround|t\s*&\s*i)|subsequent\s*t\s*&\s*i|will\s+be\s+closely\s+inspected",
    re.I,
)
_RECOMMEND_CONTEXT_EXEMPT_RE = re.compile(r"as\s+per\s+.*recommendation|according\s+to\s+.*recommendation|권고에\s*따라", re.I)
_NEGATED_ACTION_RE = re.compile(
    r"(?:"
    r"(?:보수|repair|교체|replace|도장|paint|coating|용접|weld|설치|install|가공|machin(?:e|ed)|보강|reinforc(?:e|ed)|재시공|시공).{0,40}?"
    r"(?:하지\s*않(?:음|았음)|미실시|실시하지\s*않(?:음|았음)|안\s*함|없음|불필요)"
    r"|보수\s*작업\s*없[이이]|보수작업\s*없이|보수하지\s*않고|용접\s*보수하지\s*않고|교체하지\s*않고|취소하였음|취소됨|취소|cancelled|canceled"
    r"|without\s+repair|without\s+replacement|not\s+performed"
    r"|별도\s*보수작업\s*실시하지\s*않음"
    r"|보수\s*작업은\s*실시하지\s*않음"
    r"|\bno\s+repair(?:\s+was\s+made)?\b"
    r"|\brepair\s+was\s+not\s+made\b"
    r"|\bnot\s+repair(?:ed)?\b"
    r"|\bneed\s+not\s+repair\b"
    r"|\bno\s+need\s+to\s+repair\b"
    r"|\bnot\s+required(?:\s+to\s+repair)?\b"
    r"|\bit\s+was\s+decided\s+that\s+.*?not\s+repair\b"
    r"|\b(?:repair|renewal|replacement)\s+of\b.{0,80}?\b(?:unnecessary|not\s+necessary|unneeded|unrequired)\b"
    r"|\b(?:repair|replace|renew|install)\b.{0,40}?\b(?:unnecessary|not\s+necessary|unneeded|unrequired)\b"
    r"|\b(?:unnecessary|not\s+necessary|unneeded|unrequired)\s+to\s+(?:repair|replace|renew|install)\b"
    r"|\b(?:were|was|is|are)\s+unnecessary\b"
    r")",
    re.I,
)
_COATING_DAMAGE_ONLY_RE = re.compile(r"도장\s*손상|coating\s*damage|paint\s*damage|도장\s*박리", re.I)
_TOOLING_RE = re.compile(r"유압\s*토크\s*렌치|토크\s*렌치|hydraulic\s*torque\s*wrench|torque\s*wrench", re.I)
_LEVEL_GAUGE_RE = re.compile(r"level\s*gauge|레벨\s*게이지|liquid\s*level\s*gauge", re.I)
_WELD_ACTION_STRONG_RE = re.compile(
    r"육성\s*용접|육성용접|overlay|hardfacing|erni-?cr-?3|er-?nicr-?3|재\s*용접|재용접|용접보수|보수용접|weld\s*repair(?:ed)?|repair\s*weld(?:ing|ed)?|weld\s*repaired|reweld(?:ed|ing)?|weld[- ]?built[- ]?up|built\s*up\s*with|ground\s*out|deposit\s*welding|metal\s*plugg(?:ed|ing)|결함\s*제거\s*후\s*용접|선형\s*결함\s*제거\s*후\s*용접|grinding\s*후\s*용접|용접\s*실시",
    re.I,
)
_WELD_REFERENCE_RE = re.compile(r"seal\s*weld(?:ing|ed)?|seal-?weld(?:ing|ed)?|welding\s*부|용접부|weld(?:ing)?\s*joint|weld(?:ing)?\s*부위", re.I)
_INSPECTION_TEST_RE = re.compile(r"\bPT\b|\bMT\b|\bUT\b|침투탐상|자분탐상|비파괴|검사|점검|확인|수압시험|수압\s*테스트|hydro(?:static)?\s*test", re.I)
_GOOD_STATUS_RE = re.compile(r"양호|이상\s*없|상태\s*양호|문제\s*없|good\s+condition|acceptable|satisfactory|no abnormal", re.I)
_REASSEMBLY_ONLY_RE = re.compile(r"분해|해체|개방|opening|opened|조립함|조립\s*하였음|조립\s*하였습니다|조립\s*완료|재조립|재\s*조립|reassembl(?:ed|y)|assembled", re.I)
_INSPECTION_ONLY_RE = re.compile(r"\bMT\b|\bPT\b|\bUT\b|검사|점검|확인|power\s*brush|power\s*brushing|세척|clean|청소|수압\s*테스트|RT/?수압\s*테스트|액체침투탐상|침투탐상|자분탐상", re.I)
_HISTORY_PAREN_RE = re.compile(r"\([^)]*(20\d{2})년[^)]*\)", re.I)
_TRAILING_HISTORY_NOTE_RE = re.compile(r"(?:[‘'`]?\d{2}년|(?:19|20)\d{2}년).{0,220}$", re.I)
_HEADER_TRASH_RE = re.compile(
    r"검사사항\s*\(초기/상세\)|구분\s*Tube\s*Shell|초기\s*상태|상세\s*검사|표면\s*상태|도장\s*상태|^Line\s*no\.?$|^Nozzle\s*No\.?$",
    re.I,
)
_BULLET_SPLIT_RE = re.compile(
    r"(?:\n+|\\n+|\s+(?=\(\d+\))|\s+(?=-\s)|\s+(?=<[^>]+>)|[,:;]\s*(?=차기\s*T/?A|차기\s*정기|다음\s*T/?A|향후|추후|권고|요망|요함|필요|검토|예정|바람직|recommended|should\s+be|shall\s+be|must\s+be|planned\s+for|planned\s+at|scheduled\s+for|scheduled\s+to|next\s*(?:shutdown|turnaround|t\s*&\s*i))|\s+(?=차기\s*T/?A|차기\s*정기|다음\s*T/?A|향후|추후|권고|요망|요함|필요|검토|예정|바람직|recommended|should\s+be|shall\s+be|must\s+be|planned\s+for|planned\s+at|scheduled\s+for|scheduled\s+to|next\s*(?:shutdown|turnaround|t\s*&\s*i)))"
)
_HISTORICAL_ANCHOR_RE = re.compile(
    r"(?:[‘'`]?(?:\d{2}|(?:19|20)\d{2})년|지난|이전|전회|기존\s*TA|last\s+t\s*&\s*i|last\s+shutdown|previous\s+shutdown|previous\s+t\s*&\s*i|at\s+last\s+t\s*&\s*i|during\s+(?:19|20)\d{2}\s*t\s*&\s*i)",
    re.I,
)
_CURRENT_SCOPE_OVERRIDE_RE = re.compile(
    r"금번|이번\s*TA|금번\s*조치사항|이번\s*조치사항|this\s+turnaround|this\s+shutdown|current\s+shutdown|current\s+turnaround",
    re.I,
)
_FINDING_RE = re.compile(r"양호|이상없|상태\s*양호|확인(?:함|하였음|되었음)?|good\s+condition|found|confirmed|observed|사용\s*가능|usable|acceptable", re.I)
_CURRENT_ACTION_CUE_RE = re.compile(
    r"금번|이번\s*TA|금번\s*조치사항|이번\s*조치사항|this\s+turnaround|this\s+shutdown|current\s+shutdown|current\s+turnaround|교체함|교체\s*하였음|교체\s*하였습니다|교체하고|교체하여|교체\s*설치함|교체\s*실시|설치함|설치하였다|설치\s*하였음|설치\s*하였습니다|설치\s*완료|신규\s*설치|신규설치|보수함|보수\s*하였음|보수\s*하였습니다|보수\s*완료|복원\s*하였음|복원\s*완료|조립함|조립\s*하였음|조립\s*하였습니다|재조립|재\s*조립|재축조|재축조하였음|재축조하였습니다|replaced|installed|repaired|repair\s*performed|modified|reassembl(?:ed|y)|assembled|renewed|retubed",
    re.I,
)
_ACTION_SIGNAL_RE = re.compile(
    r"교체(?:\s*설치)?(?:함|하였다|하였음|하였습니다|완료|실시)?|교체하고|교체하여|설치(?:함|하였음|하였습니다|완료)?|신규\s*설치|신규설치|보수(?:함|하였다|하였음|하였습니다|완료|실시)?|개선조치(?:함|하였다|하였음|하였습니다)?|복원(?:함|하였음|완료)?|복구(?:함|실시)?|조립(?:함|하였음|하였습니다|완료)?|재조립|재\s*조립|결합\s*작업\s*실시|신규\s*시공|재축조|재축조하였음|재축조하였습니다|replaced|exchange(?:d)?|installed|repaired|repair\s*performed|renewed|retubed|retubing|modified|plugged|carried\s*out\s*plugging|reassembl(?:ed|y)|assembled|reweld(?:ed|ing)?|seal\s*weld(?:ed|ing)?|weld\s*repaired|weld[- ]?built[- ]?up|metal\s*plugg(?:ed|ing)|ground\s*out|deposit\s*welding",
    re.I,
)
_RECOMMEND_TAIL_TRIGGER_RE = re.compile(
    r"(?:\(|\[)?(?:차기교체|차기\s*T/?A|다음\s*T/?A|향후|추후|권고|요망|요함|필요|검토|예정|바람직|recommended|recommend(?:\s+that)?|we\s+recommend|should\s+be|shall\s+be|must\s+be|need(?:s)?\s+(?:to\s+be\s+|to\s+)?(?:repair(?:ed)?|replace(?:d)?|renew(?:ed)?|install(?:ed)?|weld(?:ed)?)|requires?\s+(?:repair|replacement|renewal|installation)|planned\s+for|planned\s+at|scheduled\s+for|scheduled\s+to|during\s+the\s+next|at\s+next\s*(?:s\.?d\.?|shutdown|turnaround|t\s*&\s*i)|next\s*(?:s\.?d\.?|shutdown|turnaround|t\s*&\s*i)|subsequent\s*t\s*&\s*i)",
    re.I,
)
_NEGATIVE_TAIL_TRIGGER_RE = re.compile(
    r"(?:실시하지\s*않음|미실시|조치하지\s*않음|교체하지\s*않음|보수하지\s*않음|불필요|without\s+repair|without\s+replacement|not\s+performed|no\s+repair|not\s+required(?:\s+to\s+repair)?|unnecessary|not\s+necessary|unneeded|unrequired)",
    re.I,
)
_FINDING_PREFIX_RE = re.compile(
    r"(?:검사\s*결과|점검\s*결과|상세\s*검사\s*결과|육안검사\s*결과|판단(?:하고|하여)?|판단되며|특이사항\s*없이|이상\s*없이|사용\s*가능(?:한\s*것으로)?\s*판단(?:하고|하여)?|in\s+good\s+condition|generally\s+in\s+good\s+condition)",
    re.I,
)


def _has_explicit_done(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    return bool(re.search(
        r"교체함|교체하였다|교체\s*하였음|교체\s*하였습니다|교체하고|교체하여|교체\s*설치함|교체\s*설치하였음|교체\s*완료|교체\s*실시|설치함|설치\s*하였음|설치\s*하였습니다|설치\s*완료|실시함|완료함|보수\s*완료|보수하였다|보수\s*하였음|보수\s*하였습니다|개선조치(?:함|하였음|하였습니다|하였다)|복원\s*하였음|복원\s*완료|조립함|조립\s*하였음|조립\s*하였습니다|조립\s*완료|재축조|재축조하였음|재축조하였습니다|마감\s*처리함|용접보수|보수용접|재\s*용접|재용접|weld\s*repair|repair\s*weld|weld\s*repaired|reweld(?:ed|ing)?|seal\s*weld(?:ed|ing)?|weld[- ]?built[- ]?up|built\s*up\s*with|ground\s*out|deposit\s*welding|metal\s*plugg(?:ed|ing)|plugged|carried\s*out\s*plugging|replace(d)?|exchange(?:d)?|installed|reassembl(?:ed|y)|assembled|modified|made\s+new|retube|retubed|retubing|re-?tubing|strength\s*welding|확관|bundle\s*사전\s*신규\s*제작\s*및\s*교체|bundle\s*사전\s*제작\s*및\s*교체|신규\s*bundle\s*로\s*교체함|신규\s*bundle\s*제작\s*되어\s*교체|신규\s*용기\s*제작\s*후\s*교체|제작\s*후\s*교체\s*실시"
        , t, re.I) or _DONE_FALLBACK_RE.search(t))


def _is_negative_or_empty(text: str) -> bool:
    return bool(re.fullmatch(r"(?:없음|해당없음|none|n/?a)\.?", _normalize_text(text), re.I))


def _is_weld_inspection_only(text: str) -> bool:
    t = _normalize_text(text)
    if not t or not _WELD_REFERENCE_RE.search(t):
        return False
    if _WELD_ACTION_STRONG_RE.search(t):
        return False
    if not (_INSPECTION_TEST_RE.search(t) and _GOOD_STATUS_RE.search(t)):
        return False
    if re.search(r"교체|replace|보수|repair|재시공|시공|육성용접|overlay", t, re.I):
        return False
    return True


def _is_inspection_reassembly_only(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    if not (_INSPECTION_TEST_RE.search(t) and _GOOD_STATUS_RE.search(t) and _REASSEMBLY_ONLY_RE.search(t)):
        return False
    if _REPLACE_RE.search(t) or _SIMPLE_REPAIR_RE.search(t) or _COATING_RE.search(t) or _WELD_ACTION_STRONG_RE.search(t) or _REPAIR_ACTION_RE.search(t):
        return False
    return True


def _is_nonrepair_plug_opening(text: str) -> bool:
    t = _normalize_text(text)
    if not t or not _AIR_COOLER_PLUG_SERVICE_RE.search(t):
        return False
    if _LEAK_RESPONSE_PLUG_RE.search(t) and re.search(r"prevent\s+further\s+leak|tube\s+leak|tubes?\s+were\s+leaked|found\s+\d+\s*tube", t, re.I):
        return False
    if _AIR_COOLER_OPENING_CONTEXT_RE.search(t) and re.search(r"remove|removed|open|opening|clean|cleaned|hydrojet|replug(?:ged|ging)?|assembled|조립|재조립|개방|세척", t, re.I):
        return True
    return False


def _is_leak_response_tube_plugging(text: str) -> bool:
    t = _normalize_text(text)
    if not t or _is_nonrepair_plug_opening(t):
        return False
    if not (_PLUG_ACTION_DONE_RE.search(t) or re.search(r"plug\b", t, re.I)):
        return False
    return bool(_LEAK_RESPONSE_PLUG_RE.search(t))


def _looks_like_recommendation(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    if _is_negative_or_empty(t):
        return False
    has_done = _has_explicit_done(t)
    has_recommend = bool(_RECOMMEND_ONLY_RE.search(t))
    has_future = bool(_FUTURE_SCOPE_RE.search(t)) and not _RECOMMEND_CONTEXT_EXEMPT_RE.search(t)
    if _RECOMMEND_CONTEXT_EXEMPT_RE.search(t) and has_done and not has_future:
        return False
    if has_future and re.search(r"교체|보수|설치|제작|육성용접|overlay|repair|replace|renew|install|retub|weld", t, re.I):
        if not has_done:
            return True
        if re.search(r"recommend(?:ed)?|should\s+be|shall\s+be|must\s+be|planned\s+for|planned\s+at|scheduled\s+for|scheduled\s+to|next\s*(?:s\.?d\.?|shutdown|turnaround|t\s*&\s*i)|subsequent\s*t\s*&\s*i|차기\s*T/?A|다음\s*T/?A|향후|추후", t, re.I):
            return True
    if has_recommend:
        if not re.search(r"실시함|실시하였음|완료함|완료하였음|보수함|보수\s*완료|복원\s*하였음|복원\s*완료|교체함|교체\s*설치함|설치함|설치\s*완료|repaired|repair\s*performed|replaced|installed|reweld(?:ed|ing)?|seal\s*weld(?:ed|ing)?|weld\s*repaired|metal\s*plugg(?:ed|ing)|reassembl(?:ed|y)", t, re.I):
            return True
        if re.search(r"(교체|보수|설치|제작|육성용접|overlay|repair|replace|renew|install|retub).*(필요|요망|검토|예정)", t, re.I):
            return True
    return False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _has_action_signal(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    return bool(
        _ACTION_SIGNAL_RE.search(t)
        or _REPAIR_ACTION_RE.search(t)
        or _WELD_REPAIR_RE.search(t)
        or _DONE_FALLBACK_RE.search(t)
        or (_REPLACE_RE.search(t) and (_CURRENT_ACTION_CUE_RE.search(t) or _DONE_FALLBACK_RE.search(t)))
    )


def _split_negative_local_scopes(text: str) -> List[str]:
    t = _normalize_text(text)
    if not t:
        return []
    t = re.sub(r"(?i)\b(?:but|however|while|whereas)\b", " ||SPLIT|| ", t)
    t = re.sub(r"다만|반면", " ||SPLIT|| ", t)
    t = re.sub(r"(?i)(?:검사\s*결과에\s*따라|점검\s*결과에\s*따라|판단하고|판단하여)", lambda m: f"{m.group(0)} ||SPLIT||", t)
    parts = re.split(r"\s*(?:\|\|SPLIT\|\||;)\s*", t)
    return [p.strip(" -/,:;") for p in parts if _normalize_text(p)]


def _trim_recommendation_tail(text: str) -> str:
    t = _normalize_text(text)
    if not t:
        return ""
    if _RECOMMEND_CONTEXT_EXEMPT_RE.search(t):
        return t
    t = re.sub(r"\((?:차기교체|차기\s*T/?A[^)]*|다음\s*T/?A[^)]*|향후[^)]*|추후[^)]*|권고[^)]*|필요[^)]*|바람직[^)]*|recommended[^)]*|should\s+be[^)]*|shall\s+be[^)]*|must\s+be[^)]*|planned[^)]*|scheduled[^)]*|next\s*(?:s\.?d\.?|shutdown|turnaround|t\s*&\s*i)[^)]*)\)", " ", t, flags=re.I)
    m = _RECOMMEND_TAIL_TRIGGER_RE.search(t)
    if not m:
        return _normalize_text(t)
    head = _normalize_text(t[:m.start()])
    tail = _normalize_text(t[m.start():])
    if head and (_has_explicit_done(head) or _has_action_signal(head)):
        return head
    if tail and (_has_explicit_done(tail) or _has_action_signal(tail)):
        return _normalize_text(t)
    return _normalize_text(t)


def _trim_noncurrent_tail(text: str) -> str:
    t = _normalize_text(text)
    if not t:
        return ""
    t = _trim_recommendation_tail(t)
    if not t:
        return ""
    if _RECOMMEND_CONTEXT_EXEMPT_RE.search(t):
        return t

    neg_m = _NEGATIVE_TAIL_TRIGGER_RE.search(t)
    hist_m = _HISTORICAL_ANCHOR_RE.search(t)

    if neg_m and neg_m.start() > 0:
        head = _normalize_text(t[:neg_m.start()])
        if head and _has_explicit_done(head):
            return head

    if hist_m and hist_m.start() > 0:
        head = _normalize_text(t[:hist_m.start()])
        if head and (_has_explicit_done(head) or _has_action_signal(head)):
            return head

    return t


def _is_finding_recommendation_only(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return False
    has_finding = bool(_FINDING_RE.search(t))
    has_recommend = bool(_FUTURE_SCOPE_RE.search(t) or re.search(r"바람직|차기교체", t, re.I))
    has_action = bool(_ACTION_DONE_RE.search(t) or _DONE_FALLBACK_RE.search(t) or _REPAIR_ACTION_RE.search(t) or _WELD_REPAIR_RE.search(t) or _REPLACE_RE.search(t))
    return has_finding and has_recommend and not has_action


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
    t = re.sub(r"(?i)(?<=하였음)\s*(?=(?:차기|향후|추후|다음|권고))", ". ", t)
    t = re.sub(r"(?i)(?<=완료함)\s*(?=(?:차기|향후|추후|다음|권고))", ". ", t)
    t = re.sub(r"(?i)(?<=installed)\s*(?=(?:recommended|should\s+be|shall\s+be|next\s*(?:shutdown|turnaround|t\s*&\s*i)))", ". ", t)
    t = re.sub(r"(?i)(?<=repaired)\s*(?=(?:recommended|should\s+be|shall\s+be|next\s*(?:shutdown|turnaround|t\s*&\s*i)))", ". ", t)
    t = _TRAILING_HISTORY_NOTE_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" -/,:;")


def _split_clauses(text: str) -> List[str]:
    if not text:
        return []
    raw = _clean_clause_text(text)
    if not raw:
        return []
    raw = _BULLET_SPLIT_RE.sub(" |SPLIT| ", raw)
    raw = re.sub(r"(?:(?<=^)|(?<=\s))(?:\(?\d+\)|\d+\.)\s+", " |SPLIT| ", raw)
    raw = re.sub(r"(?i)(보수/개선\s*내용|주요\s*정비\s*내용|조치\s*사항|조치\s*내용|초기\s*검사|상세\s*검사(?:\s*\(NDE\s*포함\))?)\s*[:：]?", " |SPLIT| ", raw)
    raw = re.sub(r"(?i)(양호함\.?|양호하였음\.?|이상\s*없음\.?)(\s+)(?=(신규|교체|보수|replace|repair|도장|touch-?up|anchor\s*bolt|기존\s*anchor\s*bolt|모든\s*Nozzle|내부\s*모든))", r"\1 |SPLIT| ", raw)
    raw = re.sub(r"(?i)(설치\s*완료|설치함|교체함|교체\s*설치함|replaced|installed)(\s+)(?=(?:anchor\s*bolt|기존\s*anchor\s*bolt|모든\s*Nozzle|내부\s*모든|외부\s*검사|추가\s*점검))", r"\1 |SPLIT| ", raw)
    first_pass = [p for p in re.split(r"\s*(?:\|SPLIT\||\n|/|;)+\s*", raw) if _normalize_text(p)]
    parts: List[str] = []
    for part in first_pass:
        sub_parts = re.split(r"(?<=[\.!?다함음요])\s+(?=(?:\(?\d+\)|[A-Z#0-9\"“]|Nozzle|Tray|Shell|Top|Bottom|내부|외부|Anchor|차기\s*TA|다음\s*TA|권고|검토|[‘'`]?(?:19|20)?\d{2}년))", part)
        split_more = []
        for sub in sub_parts:
            split_more.extend(re.split(r"\s+(?=(?:but|however|다만|또한|차기\s*T/?A|차기\s*정기|다음\s*T/?A|향후|추후|권고|recommended|recommend\s+that|we\s+recommend|should\s+be|shall\s+be|must\s+be|planned\s+for|planned\s+at|scheduled\s+for|scheduled\s+to|next\s*(?:s\.?d\.?|shutdown|turnaround|t\s*&\s*i)|subsequent\s*t\s*&\s*i))", sub, flags=re.I))
        for sub in split_more:
            sub = _normalize_text(sub)
            if not sub:
                continue
            parts.append(sub)
    return [p for p in parts if _normalize_text(p)]


def _is_recommendation_only(text: str) -> bool:
    t = _normalize_text(text)
    return bool(t and _looks_like_recommendation(t))


def _is_historical_only(text: str, report_year: int) -> bool:
    t = _normalize_text(text)
    years = {int(y) for y in re.findall(r"((?:19|20)\d{2})", t)}
    for y2 in re.findall(r"[‘\'`](\d{2})년", t):
        yy = int(y2)
        years.add(2000 + yy if yy <= 30 else 1900 + yy)
    has_hist_anchor = bool(_HISTORICAL_ANCHOR_RE.search(t))
    has_current_override = bool(_CURRENT_SCOPE_OVERRIDE_RE.search(t) or _CURRENT_ACTION_CUE_RE.search(t))
    if not years and not has_hist_anchor:
        return False
    if any(y == int(report_year) for y in years):
        return False
    if _has_explicit_done(t) and re.search(r"에\s*따라|according\s+to|based\s+on|per\s+.*(?:msr|inspection|result)|사전에\s*준비|재조립|교체하였|설치하였|replaced|installed|modified", t, re.I):
        return False
    if has_hist_anchor and not has_current_override:
        if re.match(r"^(?:\(?\s*)?(?:[‘\'`]?(?:\d{2}|(?:19|20)\d{2})년|지난|이전|전회|기존\s*TA|last\s+t\s*&\s*i|last\s+shutdown|previous\s+shutdown|previous\s+t\s*&\s*i)", t, re.I):
            return True
        if len(t) <= 160 and re.search(r"(?:교체|보수|용접|replace|replaced|repair(?:ed)?|retube|retubing|renewed)", t, re.I):
            return True
    if re.match(r"^(?:[‘\'`]\d{2}년|(?:19|20)\d{2}년)", t) and len(t) <= 120:
        return True
    if re.fullmatch(r".{0,120}?(?:교체|보수|용접|replace|replaced|repair(?:ed)?|retube|retubing|renewed).{0,40}", t, re.I):
        return True
    return False


def _strip_inline_historical_segments(text: str, report_year: int) -> str:
    t = _normalize_text(text)
    if not t:
        return ""
    parts = re.split(r"(?<=[\.;|])\s+", t)
    kept: List[str] = []
    for part in parts:
        p = _normalize_text(part.strip(" -/,:;"))
        if not p:
            continue
        if _is_historical_only(p, report_year):
            continue
        kept.append(p)
    return _normalize_text(" | ".join(kept)) if kept else t


def categorize_text(text: str, action_type: str = "") -> List[str]:
    combined = _normalize_text(f"{action_type} {text}")
    combined = _trim_noncurrent_tail(combined)
    if not combined:
        return []
    if _INTERNAL_EXCLUDE_RE.search(combined) and not (_WELD_REPAIR_RE.search(combined) or _SIMPLE_REPAIR_RE.search(combined) or _REPLACE_RE.search(combined)):
        return []
    if _is_negative_or_empty(combined):
        return []
    if _is_finding_recommendation_only(combined):
        return []
    if _is_recommendation_only(combined):
        return []
    if _NEGATED_ACTION_RE.search(combined) and not _has_explicit_done(combined) and not (re.search(r"bundle\s+.*replaced|bundle\s+.*교체|tube\s+bundle\s+.*교체|신규\s*제작\s*후\s*교체|made\s+new\s+.*nozzle|new\s*nozzle|new\s*nozzles", combined, re.I) and _REPLACE_RE.search(combined)):
        return []
    if _is_nonrepair_plug_opening(combined):
        return []
    if _HEADER_TRASH_RE.search(combined) and not (_REPLACE_RE.search(combined) or _SIMPLE_REPAIR_RE.search(combined) or _COATING_RE.search(combined) or _WELD_REPAIR_RE.search(combined)):
        return []
    if _SMALL_PART_EXCLUDE_RE.search(combined) and not (
        _NOZZLE_RE.search(combined)
        or re.search(r"orifice\s*plate|guide\s*set\s*plate|guide\s*plate|flapper\s*valve|quench\s*distributor|distributor\s*pipe|tray|baffle|deck\s*plate|seal\s*plate|bracing\s*cone|vortex\s*breaker|distributor\b", combined, re.I)
    ):
        return []
    if _BLAST_ONLY_RE.search(combined) and not _COATING_RE.search(combined):
        return []
    if _INSPECTION_ONLY_RE.search(combined) and not (_REPLACE_RE.search(combined) or _COATING_RE.search(combined) or _OVERLAY_RE.search(combined) or _SIMPLE_REPAIR_RE.search(combined) or _WELD_REPAIR_RE.search(combined) or _ACTION_DONE_RE.search(combined) or _DONE_FALLBACK_RE.search(combined)):
        return []
    if _is_inspection_reassembly_only(combined):
        return []

    has_replace = bool(_REPLACE_RE.search(combined)) or "replace" in action_type.lower()
    has_internal = bool(_INTERNAL_PART_RE.search(combined))
    has_nozzle = bool(_NOZZLE_RE.search(combined))
    has_assembly_obj = bool(_ASSEMBLY_OBJ_RE.search(combined))
    has_assembly_ctx = bool(_ASSEMBLY_CONTEXT_RE.search(combined))
    explicit_assembly_obj = bool(re.search(r"combust(?:or|er)|claus\s*combust(?:or|er)|bundle|tube\s*bundle|new\s*vessel|신규\s*용기|\b용기\b|vessel(?:\b|(?=[가-힣]))|drum(?:\b|(?=[가-힣]))|column(?:\b|(?=[가-힣]))|tower(?:\b|(?=[가-힣]))|separator(?:\b|(?=[가-힣]))|receiver(?:\b|(?=[가-힣]))|pot(?:\b|(?=[가-힣]))|shell\s*cover|floating\s*head|\bchannel\b|\bassembly\b|\bassy\b|\bduct\b|\bdamper\b|steam\s*manifold|pilot\s*gas\s*assembly|chimney\s*section|return\s*bend|expansion\s*joint|bellows|saddle(?!\s*clip)", combined, re.I))
    has_small_part = bool(_SMALL_PART_EXCLUDE_RE.search(combined))
    has_tooling = bool(_TOOLING_RE.search(combined))
    has_coating = bool(_COATING_RE.search(combined)) or "coating" in action_type.lower()
    has_overlay = bool(_OVERLAY_RE.search(combined))
    has_weld_repair = bool(_WELD_REPAIR_RE.search(combined)) or "weld_repair" in action_type.lower()
    if _is_weld_inspection_only(combined):
        has_overlay = False
        has_weld_repair = False
    has_simple = bool(_SIMPLE_REPAIR_RE.search(combined)) or any(x in action_type.lower() for x in ["temporary_fix", "plugging"])
    has_done = _has_explicit_done(combined)
    assembly_anchor_aux = bool(
        explicit_assembly_obj
        and has_replace
        and re.search(r"anchor\s*bolt", combined, re.I)
        and re.search(r"\bMT\b|\bPT\b|검사|점검|확인", combined, re.I)
    )
    assembly_strong_done = bool(
        explicit_assembly_obj
        and has_replace
        and (
            has_done
            or re.search(r"신규\s*제작|제작\s*후\s*교체|신규\s*(?:column|drum|tower|vessel|separator|receiver|pot)|교체\s*설치함|설치\s*완료|newly\s*fabricated|prefabricated|replaced\s+with\s+new", combined, re.I)
        )
    )
    categories: List[str] = []

    if has_tooling:
        if has_replace or has_simple or has_done:
            return ["단순 보수"]
        return []

    # Level Gauge Assembly는 설비 본체 Assembly 교체가 아니라 계기/부속 배관 정비로 간주한다.
    if _LEVEL_GAUGE_RE.search(combined):
        if _is_finding_recommendation_only(combined) or (_looks_like_recommendation(combined) and not has_done and not has_simple):
            return []
        if re.search(r"box-?up|배관|pipe|piping|condensate|내부\s*배관|교체\s*작업\s*진행|정비", combined, re.I) or has_replace or has_simple or has_done:
            return ["단순 보수"]
        return []

    coating_done = bool(_COATING_DONE_RE.search(combined) or re.search(r"coat(?:ed|ing)", combined, re.I))
    if _COATING_STATE_RE.search(combined) and not coating_done:
        has_coating = False
    if has_coating and _COATING_DAMAGE_ONLY_RE.search(combined) and not coating_done:
        has_coating = False
    if "coating_repair" in action_type.lower() and re.search(r"touch-?up|epoxy|도장", combined, re.I) and has_done:
        has_coating = True
        coating_done = True

    if has_replace and not (_looks_like_recommendation(combined) and not has_done):
        nozzle_ok = False
        if has_nozzle:
            nozzle_ok = bool(
                has_done
                or re.search(r"made\s+new|newly\s*installed|installed|fabricated|new\s*nozzle|new\s*nozzles|신규\s*설치|신규\s*제작|조립하였|조립함|교체하고|교체하여", combined, re.I)
            )
            if nozzle_ok and re.search(r"mint|ont|pitting|부식|감육|두께감소", combined, re.I) and not re.search(r"신규|제작|설치|제거\s*후|size-?up|기존|조립하였|교체하고|교체하여", combined, re.I):
                nozzle_ok = False
        if nozzle_ok:
            categories.append("Nozzle 교체")

        internal_ok = False
        if has_internal and not has_small_part:
            internal_ok = bool(
                has_done
                or has_assembly_ctx
                or re.search(r"전체\s*교체|교체\s*완료|교체하고|교체하여|조립하였|설치하였|재축조|신규|제작|exchange|exchanged", combined, re.I)
                or (
                    re.search(r"보수/개선\s*내용|보수\s*개선\s*내용", combined, re.I)
                    and not re.search(r"교체\s*예정|교체예정|교체\s*없이|without\s+replacement", combined, re.I)
                )
                or re.search(r"(?:^|[)\]\s])(?:교체|replace)\s*$", combined, re.I)
            )
        if internal_ok:
            categories.append("단순 내부 구성품 교체")

        verified_internal_replace = (
            "replace" in action_type.lower()
            and _PRIMARY_INTERNAL_OBJ_RE.search(combined)
            and not _looks_like_recommendation(combined)
            and (has_done or re.search(r"신규|교체|설치|exchange|exchanged|new", combined, re.I))
        )
        if verified_internal_replace and "단순 내부 구성품 교체" not in categories:
            categories.append("단순 내부 구성품 교체")

        assembly_ok = False
        if (explicit_assembly_obj or (has_assembly_ctx and not has_internal) or assembly_strong_done) and not (has_small_part and not assembly_anchor_aux):
            if not has_nozzle or explicit_assembly_obj or assembly_strong_done:
                assembly_ok = bool(
                    assembly_strong_done
                    or has_done
                    or has_assembly_ctx
                    or re.search(r"retube(?:d|ing)?|re-?tubing|made\s+new|fabricated|installed|reassembl(?:ed|y)|strength\s*welding|튜브\s*교체|번들\s*교체|부분\s*retubing|교체하고|교체하여|조립하였", combined, re.I)
                )
            if assembly_ok and _looks_like_recommendation(combined) and not has_done and not assembly_anchor_aux:
                assembly_ok = False
            if assembly_ok and re.search(r"설계두께|최소허용두께|상태|pitting|general corrosion|부식", combined, re.I) and not has_done and not has_assembly_ctx and not assembly_anchor_aux:
                assembly_ok = False
        if assembly_ok:
            categories.append("Assembly 교체")

    if not categories and has_nozzle and has_done and not _is_weld_inspection_only(combined):
        if re.search(r"보수|repair|개선조치|교체|replace|용접보수|weld\s*repair|crack", combined, re.I):
            categories.append("Nozzle 교체")

    if not categories and has_internal and not has_small_part and has_done and re.search(r"재조립|재\s*조립|조립|결합\s*작업\s*실시|설치|install(?:ed)?|reassembl(?:ed|y)|assembled|modified|재축조|마감\s*처리함|exchange|exchanged|replaced", combined, re.I):
        categories.append("단순 내부 구성품 교체")

    if has_coating and (coating_done or not (has_replace or has_simple or has_overlay)):
        categories.append("도장")
    if has_overlay or has_weld_repair:
        if not (_looks_like_recommendation(combined) and not has_done):
            if has_done or re.search(r"육성용접|overlay|erni-?cr-?3|er-?nicr-?3|strength\s*welding|seal\s*weld(?:ing|ed)?|seal-?weld(?:ing|ed)?|재\s*용접|재용접|용접보수|보수용접|weld\s*repair(?:ed)?|repair\s*weld(?:ing|ed)?|weld\s*repaired|reweld(?:ed|ing)?|weld[- ]?built[- ]?up|built\s*up\s*with|ground\s*out|deposit\s*welding|metal\s*plugg(?:ed|ing)", combined, re.I):
                categories.append("육성용접")
    if has_simple or _is_leak_response_tube_plugging(combined):
        if not (_RECOMMEND_ONLY_RE.search(combined) and not has_done):
            if not re.search(r"보수작업\s*없이|without\s+repair|보수하지\s*않고|용접\s*보수하지\s*않고", combined, re.I):
                if _is_leak_response_tube_plugging(combined) or _REPAIR_ACTION_RE.search(combined) or has_done:
                    categories.append("단순 보수")

    if re.search(r"bushing\s*nozzle|nozzle\s*bushing", combined, re.I):
        if "단순 내부 구성품 교체" in categories:
            categories = [c for c in categories if c != "단순 내부 구성품 교체"]
        if has_replace and has_done and "Nozzle 교체" not in categories:
            categories.append("Nozzle 교체")

    if "Assembly 교체" in categories and re.search(r"seal\s*plate|deck\s*plate|\bbushing\b|tray\s*part", combined, re.I):
        categories = [c for c in categories if c != "Assembly 교체"]

    if "단순 내부 구성품 교체" in categories and re.search(r"flapper\s*valve.*gap|hard\s*face|메꿈\s*작업|gap\s*(?:filling|fill|sealing)", combined, re.I):
        if not re.search(r"교체함|교체\s*완료|신규|new\s+one|replaced\s+with\s+new", combined, re.I):
            categories = [c for c in categories if c != "단순 내부 구성품 교체"]
            if "단순 보수" not in categories:
                categories.append("단순 보수")

    if "Nozzle 교체" in categories and _GASKET_ONLY_RE.search(combined):
        if not re.search(r"neck|boss|nipple|elbow|sch\.|용접부|crack|부식|감육|new\s*nozzle|nozzle.{0,40}(?:교체|replace|보수|repair)|(?:교체|replace|보수|repair).{0,40}nozzle", combined, re.I):
            categories = [c for c in categories if c != "Nozzle 교체"]

    if "단순 보수" in categories and "도장" in categories:
        if _COATING_ONLY_SIMPLE_RE.search(combined) and not _NONCOATING_SIMPLE_RE.search(combined):
            categories = [c for c in categories if c != "단순 보수"]

    primary_object = None
    if "replace" in action_type.lower() and _PRIMARY_INTERNAL_OBJ_RE.search(combined) and (has_done or re.search(r"신규|교체|설치|exchange|exchanged|new", combined, re.I)):
        primary_object = "internal"
    elif has_nozzle and has_replace and has_done:
        primary_object = "nozzle"
    elif (assembly_strong_done or (explicit_assembly_obj or (has_assembly_ctx and not has_internal))) and has_replace and (has_done or assembly_anchor_aux):
        primary_object = "assembly"

    if primary_object == "internal":
        categories = [c for c in categories if c not in ["Nozzle 교체", "Assembly 교체"]]
    elif primary_object == "nozzle":
        categories = [c for c in categories if c not in ["단순 내부 구성품 교체", "Assembly 교체"]]
    elif primary_object == "assembly":
        categories = [c for c in categories if c not in ["단순 내부 구성품 교체", "Nozzle 교체"]]

    if "육성용접" in categories and "단순 보수" in categories:
        categories = [c for c in categories if c != "단순 보수"]

    return _unique_keep_order([c for c in CATEGORY_ORDER if c in categories])


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
        text = _strip_inline_historical_segments(text, year)
        for clause in _split_clauses(text):
            local_scopes = _split_negative_local_scopes(clause) or [clause]
            for local_clause in local_scopes:
                local_clause = _trim_noncurrent_tail(local_clause)
                local_clause = _clean_clause_text(local_clause)
                local_clause = _normalize_text(local_clause)
                if not local_clause or _is_negative_or_empty(local_clause):
                    continue
                if _FINDING_PREFIX_RE.search(local_clause) and _RECOMMEND_TAIL_TRIGGER_RE.search(local_clause) and not _has_action_signal(local_clause):
                    continue
                if _is_finding_recommendation_only(local_clause):
                    continue
                if _NEGATED_ACTION_RE.search(local_clause) and not _has_explicit_done(local_clause):
                    continue
                if _is_historical_only(local_clause, year):
                    continue
                if _is_recommendation_only(local_clause):
                    continue
                if len(local_clause) < 10 and not re.search(r"교체|보수|용접|도장|coating|replace|repair|plug", local_clause, re.I):
                    continue
                if re.fullmatch(r"(?:LL1|LL2|W|D5|C3|C4|M3|T1/T2|Crude|RC Ex\.?|O/H)\.?", local_clause, re.I):
                    continue
                cats = categorize_text(local_clause, "")
                for cat in cats:
                    items.append({
                        "year": year,
                        "category": cat,
                        "text": local_clause,
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
            clause = re.sub(r"^(?:차기\s*Recommendation\s*)?\(?\d+[)\.]\s*", "", clause, flags=re.I)
            clause = re.sub(r"^[\-•*]+\s*", "", clause)
            clause = _normalize_text(clause).strip(" -/:;,.[]()")
            if not clause or _is_negative_or_empty(clause) or _is_historical_only(clause, year):
                continue
            if _has_explicit_done(clause):
                continue
            if re.fullmatch(r"(?:필요|요망|검토|예정)\)?[:：]?\s*\d+\.?", clause, re.I):
                continue
            if len(clause) < 12:
                continue
            if re.search(r"\bMAT\b|최소요구|minimum\s+required", clause, re.I) and not re.search(r"교체|보수|설치|용접|도장|replace|repair|install|weld", clause, re.I):
                continue
            if re.match(r"^(?:필요|요망|검토|예정)\b", clause, re.I) and not re.search(r"교체|보수|설치|도장|용접|replace|repair|install|weld|coating|retub|packing|nozzle|bundle|lining|refractory", clause, re.I):
                continue
            if re.search(r"취소하였음|취소됨|\b취소\b|cancelled|canceled", clause, re.I):
                continue
            if not re.search(r"차기|다음|향후|추후|권고|필요|요망|검토|예정|교체|보수|설치|도장|용접|replace|repair|install|weld|coating|retub|packing|nozzle|bundle|lining|refractory", clause, re.I):
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


def _clean_equipment_name(name: str) -> str:
    t = _normalize_text(name)
    if not t:
        return ""
    t = re.sub(r"^(?:AND|THE)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -/:;,.[]()")
    if len(t) > 80:
        return ""
    if re.search(
        r"검사일|상세내용|차기고려사항|발생년도|발췌\s*category|점검\s*결과|검사\s*결과|확인됨|확인되었|발생\s*확인|양호한\s*상태|양호함|필요|요망|검토|실시|진행|부식|감육|균열|연결\s*nozzle|grid\s*ut|scanning|thickness|두께\s*측정|정밀\s*두께|보수작업|교체여부",
        t,
        re.I,
    ):
        return ""
    if len(t.split()) >= 6 and re.search(r"확인|발생|진행|실시|필요|요망|검토|판단|측정|부착|고착|양호|보수|교체|용접|도장|repair|replace|inspect|confirm|found|observed", t, re.I):
        return ""
    if re.search(r"[.!?]", t):
        return ""
    return t


def _resolve_case_equipment_name(case: RepeatCase) -> str:
    counts: dict[str, int] = {}

    def add(name: str, weight: int = 1):
        cleaned = _clean_equipment_name(name)
        if not cleaned:
            return
        counts[cleaned] = counts.get(cleaned, 0) + weight

    add(getattr(case, "equipment_name", ""), 3)
    for event in getattr(case, "events", []) or []:
        add(getattr(event, "equipment_name", ""), 1)

    if not counts:
        return _clean_equipment_name(getattr(case, "equipment_name", ""))
    return sorted(counts.keys(), key=lambda name: (-counts[name], -len(name), name))[0]


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
        if _TOOLING_RE.search(texts) or _LEVEL_GAUGE_RE.search(texts):
            return False
        return bool(
            re.search(
                r"bundle|tube\s*bundle|new\s*vessel|신규\s*용기|retube|shell\s*cover|floating\s*head|channel\b|backing\s*device|\bassembly\b|\bassy\b|duct|damper|vortex\s*breaker|combust(?:or|er)|claus\s*combust(?:or|er)|신규\s*제작된\s*장치|신규\s*장치|\bdevice\b|\bunit\b",
                texts,
                re.I,
            )
            and (_has_explicit_done(texts) or _ACTION_DONE_RE.search(texts) or _REPLACE_RE.search(texts))
        )
    if category == "Nozzle 교체":
        return bool(re.search(r"nozzle|노즐|\bnzl\b|\belbow\b", texts, re.I))
    if category == "단순 내부 구성품 교체":
        return bool(_INTERNAL_PART_RE.search(texts))
    if category == "도장":
        return bool(_COATING_RE.search(texts))
    if category == "육성용접":
        return bool((_OVERLAY_RE.search(texts) or _WELD_ACTION_STRONG_RE.search(texts)) and not _is_weld_inspection_only(texts))
    if category == "단순 보수":
        return bool(
            _TOOLING_RE.search(texts)
            or _REPAIR_ACTION_RE.search(texts)
            or _is_leak_response_tube_plugging(texts)
            or ((_has_explicit_done(texts) or _ACTION_DONE_RE.search(texts)) and (_SIMPLE_REPAIR_RE.search(texts) or _TOOLING_RE.search(texts)))
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
        eq_name = _resolve_case_equipment_name(case)
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
                repeat_locations="",
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
