"""배관 반복 교체 리포트 생성 스크립트"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.piping_repeat_builder import export_piping_repeat_report

HISTORY_FILE = Path('/home/user/downloads/목록_20260330092940.xlsx')
TROUBLE_FILE = Path('/home/user/downloads/목록_20260330092956.xlsx')
OUT_DIR = Path('/mnt/user-data/outputs/repeat_task_v6')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / '반복정비_배관_LineNumber_반복교체_리포트.xlsx'

output = export_piping_repeat_report(HISTORY_FILE, TROUBLE_FILE, OUT)
print(f'OUTPUT: {output}')
