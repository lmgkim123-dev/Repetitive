from pathlib import Path
import pandas as pd

files = [
    Path('/home/user/downloads/목록_20260330092940.xlsx'),
    Path('/home/user/downloads/목록_20260330092956.xlsx'),
]
for path in files:
    print(f'FILE: {path.name}')
    xls = pd.ExcelFile(path)
    print('SHEETS:', xls.sheet_names)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        print(f'\n[{sheet}] shape={df.shape}')
        print('COLUMNS:', list(df.columns))
        print(df.head(5).to_string(index=False))
    print('\n' + '='*120 + '\n')
