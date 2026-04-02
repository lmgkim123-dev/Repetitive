from .schemas import RepeatCase

def judge_repeat_cases(equipment_events, equipment_names):
    cases=[]
    for eq, events in equipment_events.items():
        years=sorted({e.report_year for e in events})
        if len(years)>=2:
            cases.append(RepeatCase(equipment_no=eq, equipment_name=equipment_names.get(eq,''), years=years, events=events, confidence=0.9))
    return cases

def merge_cases_per_equipment(cases):
    return cases
