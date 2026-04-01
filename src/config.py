from __future__ import annotations

QUALITATIVE_RULES = [
    {
        'label': '반복도장 설비',
        'patterns': [r'도장', r'paint', r'coating', r'touch-?up', r'coating repair'],
        'priority': 20,
    },
    {
        'label': 'Top Head 구간 반복 손상 설비',
        'patterns': [r'top head', r'상부 head', r'head 내부', r'top head.*부식', r'top head.*pitting'],
        'priority': 100,
    },
    {
        'label': 'Shell 하부 반복 손상 설비',
        'patterns': [r'shell 하부', r'하부.*부식', r'하부.*pitting', r'하부.*육성용접', r'하부 방향'],
        'priority': 95,
    },
    {
        'label': 'Top Head/Tray/Internal 반복 손상 설비',
        'patterns': [r'tray', r'internal', r'distributor', r'collector', r'packing', r'baffle', r'wear pad', r'entry horn'],
        'priority': 90,
    },
    {
        'label': '반복 Bundle 교체/검토 설비',
        'patterns': [r'bundle.*교체', r'신규 bundle', r'new bundle', r'번들.*교체', r'bundle 제작', r'bundle 검토'],
        'priority': 92,
    },
    {
        'label': '반복 Plugging 설비',
        'patterns': [r'plugging', r'unplugging', r'plugged', r'coke plugging'],
        'priority': 88,
    },
    {
        'label': '반복 Leak 설비',
        'patterns': [r'leak', r'누설', r'재누설', r'미세 leak'],
        'priority': 87,
    },
    {
        'label': '임시조치 후 본보수 반복 설비',
        'patterns': [r'임시조치', r'box-?up', r'compound sealing', r'clamp', r'patch'],
        'priority': 93,
    },
    {
        'label': '두께감소구간 반복 보수/교체 설비',
        'patterns': [r'두께감소', r'thinning', r'감육', r'교체요청', r'구간 교체', r'corrosion rate'],
        'priority': 91,
    },
    {
        'label': '균열 반복 보수 설비',
        'patterns': [r'균열', r'crack', r'stop hole', r'추가균열'],
        'priority': 89,
    },
    {
        'label': '차기 TA 반복 권고 설비',
        'patterns': [r'차기 ta', r'권고', r'검토 필요', r'교체 필요', r'요망', r'예정 조치'],
        'priority': 70,
    },
    {
        'label': '반복 손상 징후 설비',
        'patterns': [r'corrosion', r'pitting', r'erosion', r'fouling', r'sludge', r'scale'],
        'priority': 50,
    },
]

DAMAGE_KEYWORDS = {
    '부식': [r'corrosion', r'부식', r'pitting', r'pit', r'erosion'],
    '균열': [r'crack', r'균열', r'선형결함'],
    '누설': [r'leak', r'누설'],
    '플러깅': [r'plugging', r'unplugging', r'coke plugging'],
    '도장': [r'도장', r'paint', r'coating'],
    '내부구조물': [r'tray', r'packing', r'distributor', r'collector', r'baffle', r'wear pad', r'entry horn'],
}

ACTION_KEYWORDS = {
    '교체': [r'교체', r'replace', r'신규 제작'],
    '용접보수': [r'용접보수', r'육성용접', r'재 용접', r'weld'],
    '검사/추적': [r'검사', r'점검', r'power brush', r'pt', r'mt', r'iris'],
    '기계가공': [r'기계가공', r'smooth grind', r'grinding'],
    '임시조치': [r'임시조치', r'box-?up', r'compound sealing'],
}

EQUIPMENT_REGEX = r'\b\d{2,3}[A-Z]-\d{3,4}[A-Z]{0,2}\b|\b\d{2,3}-[A-Z]-\d{3,4}[A-Z]{0,2}\b|\b\d{2,3}[A-Z]{1,3}-\d{3,4}[A-Z]{0,2}\b|\b\d{2,3}-[A-Z]-\d{1,4}-[A-Z0-9\-/\"]+\b'

TEXT_EXTENSIONS = {'.txt'}
PDF_EXTENSIONS = {'.pdf'}
EXCEL_EXTENSIONS = {'.xlsx', '.xlsm', '.xls'}
