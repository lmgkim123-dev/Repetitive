import pandas as pd
from pathlib import Path
p=Path('/mnt/user-data/outputs/repeat_task_v6/반복정비_조치요약_v6_개선판.xlsx')
df=pd.read_excel(p, sheet_name='과제후보_등록형식')
for cat in ['Nozzle 교체','단순 내부 구성품 교체','Assembly 교체','육성용접','도장','단순 보수']:
    sub=df[df['발췌 Category']==cat]
    print(cat, len(sub))
    print(sub[['Equipment No','발생년도','TA 조치사항']].head(10).to_string(index=False))
    print('---')
