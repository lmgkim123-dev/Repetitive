import pandas as pd
from pathlib import Path
pd.set_option('display.max_colwidth', None)
path=Path('/home/user/downloads/목록_20260330092956.xlsx')
df=pd.read_excel(path)
mask=df['세부내용'].astype(str).str.contains('운전 중 정비이력|TA 정비이력|Trouble History', na=False)
sub=df.loc[mask, ['설비번호','발생일자','Trouble 명','세부내용']].head(12)
for i, row in sub.iterrows():
    print('\nIDX', i, 'LINE', row['설비번호'], 'DATE', row['발생일자'])
    print('TITLE:', row['Trouble 명'])
    print('DETAIL:', str(row['세부내용'])[:3000])
    print('-'*120)
