Windows 실행 순서
1) 압축 해제
2) CMD 또는 PowerShell 열기
3) 폴더 이동: cd 압축푼폴더\v6_code_portable_full
4) 패키지 설치: pip install -r requirements.txt
5) 실행: streamlit run app.py

추가 스크립트
- 설비 반복정비 엑셀 생성: python build_v6_excel.py
- 배관 Line Number 반복교체 리포트 생성: python build_piping_repeat_report.py

streamlit 명령이 안되면:
python -m streamlit run app.py
