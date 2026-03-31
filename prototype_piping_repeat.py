from pathlib import Path
import re
import pandas as pd

HIST = Path('/home/user/downloads/목록_20260330092940.xlsx')
TRB = Path('/home/user/downloads/목록_20260330092956.xlsx')

POS_DONE = re.compile(r'교체\s*(?:작업\s*)?(?:실시|완료|함|하였|했|예정완료)|동일\s*사양.*교체|신품\s*교체|신규\s*교체|부분\s*교체\s*실시|전구간\s*교체|구간\s*교체|제작하여\s*교체|제작\s*후\s*교체|replace(d)?', re.I)
POS_GENERIC = re.compile(r'교체|replace|신품|신규', re.I)
NEG_ONLY = re.compile(r'교체\s*(?:필요|예정|요망|검토)|차기\s*TA.*교체|추후\s*교체|계획|monitoring', re.I)
LINE_SPLIT = re.compile(r'(?:\\n|\n)+')

def norm_line(x):
    x = str(x or '').upper().strip()
    x = re.sub(r'\s+', '', x)
    return x

def norm_text(x):
    x = str(x or '')
    x = LINE_SPLIT.sub(' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()

def performed(text, status=''):
    t = norm_text(text)
    s = norm_text(status)
    if not t:
        return False
    if POS_DONE.search(t):
        return True
    if POS_GENERIC.search(t) and ('완료' in s):
        return True
    if POS_GENERIC.search(t) and not NEG_ONLY.search(t):
        if re.search(r'보수\s*내용|작업', t):
            return True
    return False

def key_clause(text):
    t = norm_text(text)
    clauses = re.split(r'(?<=[\.。])\s+|\s*\+\s*', t)
    hits = [c for c in clauses if POS_GENERIC.search(c)]
    if not hits:
        hits = [t[:120]]
    out = ' | '.join(hits[:3]).lower()
    out = re.sub(r'[^0-9a-z가-힣]+', '', out)
    return out[:200]

rows = []
df = pd.read_excel(HIST)
for _, r in df.iterrows():
    line = norm_line(r.get('설비번호'))
    date = pd.to_datetime(r.get('검사일'), errors='coerce')
    detail = norm_text(r.get('세부내용'))
    issue = norm_text(r.get('Issue구분'))
    if not line or pd.isna(date):
        continue
    if issue not in {'보수/교체', 'Trouble', '신설/변경'} and '교체' not in detail:
        continue
    if performed(detail, issue):
        rows.append({'source':'hist','line':line,'date':date.date().isoformat(),'year':date.year,'detail':detail,'sig':key_clause(detail)})

df = pd.read_excel(TRB)
for _, r in df.iterrows():
    line = norm_line(r.get('설비번호'))
    date = pd.to_datetime(r.get('발생일자'), errors='coerce')
    title = norm_text(r.get('Trouble 명'))
    detail = norm_text(r.get('세부내용'))
    status = norm_text(r.get('F/U 필요'))
    text = ' '.join([title, detail])
    if not line or pd.isna(date):
        continue
    if performed(text, status):
        rows.append({'source':'trouble','line':line,'date':date.date().isoformat(),'year':date.year,'detail':text,'sig':key_clause(text)})

out = pd.DataFrame(rows)
out = out.drop_duplicates(subset=['line','date','sig']).reset_index(drop=True)
print('events', len(out))
rep = out.groupby('line').agg(count=('line','size'), years=('year', lambda s: ', '.join(map(str, sorted(set(s)))))).reset_index()
rep = rep[rep['count']>=2].sort_values(['count','line'], ascending=[False,True])
print('repeat lines', len(rep))
print(rep.head(30).to_string(index=False))
for ln in rep.head(10)['line']:
    print('\n===', ln, '===')
    print(out[out['line']==ln][['date','source','detail']].sort_values('date').to_string(index=False, max_colwidth=120))
