from pathlib import Path
import re
import pandas as pd

files = [
    ('hist', Path('/home/user/downloads/목록_20260330092940.xlsx')),
    ('trouble', Path('/home/user/downloads/목록_20260330092956.xlsx')),
]

for tag, path in files:
    df = pd.read_excel(path)
    print('\nFILE', tag, path.name, 'rows', len(df))
    if tag == 'hist':
        print('Issue구분 counts:\n', df['Issue구분'].astype(str).value_counts().head(20).to_string())
        mask = df['세부내용'].astype(str).str.contains('교체|replace|신규', case=False, na=False)
        print('replace-like rows', mask.sum())
        print(df.loc[mask, ['설비번호','검사일','Issue구분','세부내용']].head(20).to_string(index=False))
    else:
        print('Trouble 구분 counts:\n', df['Trouble 구분'].astype(str).value_counts().head(20).to_string())
        print('F/U 필요 counts:\n', df['F/U 필요'].astype(str).value_counts().head(20).to_string())
        mask = (df['Trouble 명'].astype(str).str.contains('교체|replace|신규', case=False, na=False) |
                df['세부내용'].astype(str).str.contains('교체|replace|신규', case=False, na=False))
        print('replace-like rows', mask.sum())
        print(df.loc[mask, ['설비번호','발생일자','Trouble 구분','Trouble 명','F/U 필요','세부내용']].head(20).to_string(index=False))
