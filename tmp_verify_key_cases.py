import pandas as pd
from pathlib import Path
p=Path('/mnt/user-data/outputs/repeat_task_v6/반복정비_조치요약_v6_개선판.xlsx')
df=pd.read_excel(p, sheet_name='과제후보_등록형식')
for eq in ['02E-129A','02E-135','02E-147','02C-102B','02C-107','02E-1007A']:
    sub=df[df['Equipment No'].astype(str)==eq]
    print('\n===', eq, '===')
    if sub.empty:
        print('(no rows)')
        continue
    print(sub[['Equipment No','설비명','발생년도','발췌 Category','TA 조치사항']].to_string(index=False))
print('\nrows=', len(df))
print(df['발췌 Category'].value_counts().to_string())
