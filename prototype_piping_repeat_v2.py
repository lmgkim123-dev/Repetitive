from pathlib import Path
import re
import pandas as pd

HIST = Path('/home/user/downloads/목록_20260330092940.xlsx')
TRB = Path('/home/user/downloads/목록_20260330092956.xlsx')

POS_DONE = re.compile(r'교체\s*(?:작업\s*)?(?:실시|완료|함|하였|했)|동일\s*사양.*교체|신품\s*교체|신규\s*교체|부분\s*교체\s*실시|전구간\s*교체|구간\s*교체|제작하여\s*교체|제작\s*후\s*교체|전체\s*교체|spool\s*제작.*교체|replace(d)?', re.I)
POS_GENERIC = re.compile(r'교체|replace|신품|신규', re.I)
NEG_ONLY = re.compile(r'교체\s*(?:필요|예정|요망|검토)|차기\s*TA.*교체|추후\s*교체|계획|monitoring', re.I)
HISTORY_TITLE = re.compile(r'운전\s*중\s*정비이력|TA\s*정비이력|Trouble\s*History\s*자료', re.I)
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
    if POS_GENERIC.search(t) and ('완료' in s or '보수 내용' in t or '작업 실시' in t):
        return True
    return False

def sig(text):
    t = norm_text(text).lower()
    t = re.sub(r'[^0-9a-z가-힣]+', '', t)
    return t[:180]

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
    if performed(detail, issue) and not (NEG_ONLY.search(detail) and not POS_DONE.search(detail)):
        rows.append({'source':'배관 정비이력','line':line,'date':date.date().isoformat(),'year':date.year,'text':detail})

df = pd.read_excel(TRB)
for _, r in df.iterrows():
    line = norm_line(r.get('설비번호'))
    date = pd.to_datetime(r.get('발생일자'), errors='coerce')
    title = norm_text(r.get('Trouble 명'))
    detail = norm_text(r.get('세부내용'))
    status = norm_text(r.get('F/U 필요'))
    if not line or pd.isna(date):
        continue
    if HISTORY_TITLE.search(title):
        continue
    text = ' '.join([title, detail])
    if performed(text, status) and not (NEG_ONLY.search(text) and not POS_DONE.search(text)):
        rows.append({'source':'Trouble List','line':line,'date':date.date().isoformat(),'year':date.year,'text':text})

raw = pd.DataFrame(rows)
raw['text_sig'] = raw['text'].map(sig)
# same line+date on same source is one occurrence
occ = (raw.groupby(['line','date'], as_index=False)
         .agg(year=('year','first'),
              sources=('source', lambda s: ', '.join(sorted(set(s)))),
              details=('text', lambda s: ' | '.join(list(dict.fromkeys(s))[:3]))))
rep = occ.groupby('line', as_index=False).agg(count=('line','size'), years=('year', lambda s: ', '.join(map(str, sorted(set(s))))))
rep = rep[rep['count']>=2].sort_values(['count','line'], ascending=[False,True])
print('occurrences', len(occ))
print('repeat lines', len(rep))
print(rep.head(40).to_string(index=False))
for ln in rep.head(12)['line']:
    print('\n===', ln, '===')
    print(occ[occ['line']==ln][['date','sources','details']].sort_values('date').to_string(index=False, max_colwidth=140))
