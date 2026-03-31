"""배관 반복 교체 탐지 모듈

목적
- 배관 정비이력 / Trouble List 2개 파일을 함께 읽어 Line Number 단위 반복 교체를 찾는다.
- 동일 Line / 동일 일자에 여러 행이 있어도 1개 발생으로 묶는다.
- 배관 전용 규칙: Issue구분이 '보수/교체'이거나, 제목/내용에 '교체' 표현이 있으면 교체 이력로 본다.
- Trouble List 의 History 요약행도 제외하지 않고, 가능한 경우 과거 날짜/연도 단위로 쪼개서 반영한다.

주의
- 이 모듈의 완화 규칙은 배관 전용이며, 고정장치 분류 로직에는 영향을 주지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPLACE_SIGNAL_RE = re.compile(r"교체|replace", re.I)
HISTORY_TITLE_RE = re.compile(r"운전\s*중\s*정비이력|TA\s*정비이력|Trouble\s*History\s*자료", re.I)
LINE_SPLIT_RE = re.compile(r"(?:\\n|\n)+")
DATE_MARKER_RE = re.compile(r"((?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}(?:\s+\d{1,2}:\d{2}:\d{2})?|(?:19|20)\d{2}\s*TA|(?:19|20)\d{2}년)", re.I)
BRACKET_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")

REPORT_COLUMNS = [
    "NO", "Line Number", "발생구분", "발생횟수", "발생년도", "최근발생일",
    "출처", "대표조치", "상세이력", "검토메모",
]


def _norm_line(value) -> str:
    text = str(value or "").upper().strip()
    text = re.sub(r"\s+", "", text)
    return text


def _norm_text(value) -> str:
    text = str(value or "")
    text = LINE_SPLIT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_replace_signal(*texts) -> bool:
    combined = _norm_text(" ".join(str(x or "") for x in texts))
    return bool(combined and REPLACE_SIGNAL_RE.search(combined))


def _condense_text(text: str, limit: int = 220) -> str:
    t = _norm_text(text)
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _action_signature(text: str) -> str:
    t = _norm_text(text).lower()
    t = re.sub(r"[^0-9a-z가-힣]+", "", t)
    return t[:180]


def _parse_marker_to_date(marker: str, fallback_date, seq: int = 0) -> pd.Timestamp:
    marker = _norm_text(marker)
    m = re.search(r"((?:19|20)\d{2})[./-](\d{1,2})[./-](\d{1,2})", marker)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3))).normalize()
        except Exception:
            pass
    m = re.search(r"((?:19|20)\d{2})\s*(?:TA|년)", marker, re.I)
    if m:
        year = int(m.group(1))
        day = max(1, min(28, seq + 1))
        return pd.Timestamp(year=year, month=1, day=day).normalize()
    return pd.Timestamp(fallback_date).normalize()


def _split_history_segments(text: str) -> List[tuple[str, str]]:
    raw = _norm_text(text)
    raw = BRACKET_PREFIX_RE.sub("", raw)
    if not raw:
        return []
    matches = list(DATE_MARKER_RE.finditer(raw))
    if not matches:
        return [("", raw)]
    segments: List[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        marker = m.group(1)
        seg = raw[start:end].strip(" -|/")
        if seg:
            segments.append((marker, seg))
    return segments


def _history_events_from_row(line_no: str, base_date, source_type: str, title: str, detail: str, status_text: str) -> List[dict]:
    combined = _norm_text(" ".join([title, detail]))
    segments = _split_history_segments(combined)
    events: List[dict] = []
    seq = 0
    for marker, seg in segments:
        if not _has_replace_signal(seg):
            continue
        event_date = _parse_marker_to_date(marker, base_date, seq=seq)
        seq += 1
        events.append({
            "source_type": source_type,
            "line_no": line_no,
            "event_date": event_date,
            "event_year": int(event_date.year),
            "status_text": status_text,
            "title": title,
            "detail": seg,
            "raw_text": seg,
        })
    if not events and _has_replace_signal(combined):
        event_date = pd.Timestamp(base_date).normalize()
        events.append({
            "source_type": source_type,
            "line_no": line_no,
            "event_date": event_date,
            "event_year": int(event_date.year),
            "status_text": status_text,
            "title": title,
            "detail": combined,
            "raw_text": combined,
        })
    return events


def _load_history_rows(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(path)
    accepted = []
    excluded = []

    for _, row in df.iterrows():
        line_no = _norm_line(row.get("설비번호"))
        date = pd.to_datetime(row.get("검사일"), errors="coerce")
        issue = _norm_text(row.get("Issue구분"))
        detail = _norm_text(row.get("세부내용"))
        if not line_no or pd.isna(date):
            continue

        base = {
            "source_type": "배관 정비이력",
            "line_no": line_no,
            "event_date": pd.Timestamp(date).normalize(),
            "event_year": int(pd.Timestamp(date).year),
            "status_text": issue,
            "title": issue,
            "detail": detail,
            "raw_text": _norm_text(f"{issue} {detail}"),
        }

        if issue == "보수/교체" or _has_replace_signal(detail):
            accepted.append(base)
        else:
            excluded.append({**base, "exclude_reason": "교체 신호 없음"})

    return pd.DataFrame(accepted), pd.DataFrame(excluded)


def _load_trouble_rows(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(path)
    accepted = []
    excluded = []

    for _, row in df.iterrows():
        line_no = _norm_line(row.get("설비번호"))
        date = pd.to_datetime(row.get("발생일자"), errors="coerce")
        title = _norm_text(row.get("Trouble 명"))
        detail = _norm_text(row.get("세부내용"))
        status = _norm_text(row.get("F/U 필요"))
        trouble_type = _norm_text(row.get("Trouble 구분"))
        if not line_no or pd.isna(date):
            continue

        if HISTORY_TITLE_RE.search(title):
            events = _history_events_from_row(
                line_no=line_no,
                base_date=pd.Timestamp(date).normalize(),
                source_type="Trouble List",
                title=title or trouble_type,
                detail=detail,
                status_text=status,
            )
            if events:
                accepted.extend(events)
            else:
                excluded.append({
                    "source_type": "Trouble List",
                    "line_no": line_no,
                    "event_date": pd.Timestamp(date).normalize(),
                    "event_year": int(pd.Timestamp(date).year),
                    "status_text": status,
                    "title": title or trouble_type,
                    "detail": detail,
                    "raw_text": _norm_text(" ".join([title, detail])),
                    "exclude_reason": "히스토리 행이나 교체 신호 없음",
                })
            continue

        raw_text = _norm_text(" ".join([title, detail]))
        base = {
            "source_type": "Trouble List",
            "line_no": line_no,
            "event_date": pd.Timestamp(date).normalize(),
            "event_year": int(pd.Timestamp(date).year),
            "status_text": status,
            "title": title or trouble_type,
            "detail": detail,
            "raw_text": raw_text,
        }
        if _has_replace_signal(raw_text):
            accepted.append(base)
        else:
            excluded.append({**base, "exclude_reason": "교체 신호 없음"})

    return pd.DataFrame(accepted), pd.DataFrame(excluded)


def load_piping_replacement_occurrences(history_file: str | Path, trouble_file: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hist_ok, hist_ex = _load_history_rows(history_file)
    trb_ok, trb_ex = _load_trouble_rows(trouble_file)

    accepted = pd.concat([hist_ok, trb_ok], ignore_index=True) if (not hist_ok.empty or not trb_ok.empty) else pd.DataFrame()
    excluded = pd.concat([hist_ex, trb_ex], ignore_index=True) if (not hist_ex.empty or not trb_ex.empty) else pd.DataFrame()

    if accepted.empty:
        return pd.DataFrame(), excluded

    accepted = accepted.copy()
    accepted["action_sig"] = accepted["raw_text"].map(_action_signature)
    accepted = accepted.drop_duplicates(subset=["line_no", "event_date", "action_sig"])

    occ = (
        accepted.groupby(["line_no", "event_date"], as_index=False)
        .agg(
            event_year=("event_year", "first"),
            sources=("source_type", lambda s: ", ".join(sorted(set(str(x) for x in s if str(x).strip())))),
            source_count=("source_type", lambda s: len(set(str(x) for x in s if str(x).strip()))),
            titles=("title", lambda s: " | ".join(list(dict.fromkeys(str(x) for x in s if str(x).strip()))[:3])),
            details=("raw_text", lambda s: " | ".join(list(dict.fromkeys(str(x) for x in s if str(x).strip()))[:3])),
        )
        .sort_values(["line_no", "event_date"])
        .reset_index(drop=True)
    )
    return occ, excluded


def build_piping_repeat_report_dataframe(occurrences_df: pd.DataFrame) -> pd.DataFrame:
    if occurrences_df is None or occurrences_df.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    rows: List[dict] = []
    repeat_df = occurrences_df.groupby("line_no", as_index=False).agg(
        occurrence_count=("line_no", "size"),
        years=("event_year", lambda s: sorted(set(int(x) for x in s))),
        recent_date=("event_date", "max"),
    )
    repeat_df = repeat_df[repeat_df["occurrence_count"] >= 2].copy()
    repeat_df = repeat_df.sort_values(["occurrence_count", "recent_date", "line_no"], ascending=[False, False, True]).reset_index(drop=True)

    for idx, row in repeat_df.iterrows():
        line_no = row["line_no"]
        sub = occurrences_df[occurrences_df["line_no"] == line_no].sort_values("event_date")
        detail_lines = []
        representative_actions = []
        all_sources = []
        for _, occ in sub.iterrows():
            all_sources.extend([x.strip() for x in str(occ.get("sources", "")).split(",") if x.strip()])
            representative_actions.append(f"[{occ['event_date'].date()}] {_condense_text(occ.get('titles', '') or occ.get('details', ''), 120)}")
            detail_lines.append(f"- [{occ['event_date'].date()}] {occ.get('sources', '')}: {_condense_text(occ.get('details', ''), 240)}")
        rows.append({
            "NO": idx + 1,
            "Line Number": line_no,
            "발생구분": "2회 이상 반복",
            "발생횟수": int(row["occurrence_count"]),
            "발생년도": ", ".join(map(str, row["years"])),
            "최근발생일": str(pd.Timestamp(row["recent_date"]).date()),
            "출처": ", ".join(sorted(set(all_sources))),
            "대표조치": "\n".join(representative_actions[:6]),
            "상세이력": "\n".join(detail_lines[:12]),
            "검토메모": "배관 전용 규칙: 보수/교체 또는 교체 문구 기반, 동일 Line Number 2건 이상",
        })

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def build_piping_summary_sheet(occurrences_df: pd.DataFrame, excluded_df: pd.DataFrame, repeat_df: pd.DataFrame) -> pd.DataFrame:
    total_occ = 0 if occurrences_df is None else len(occurrences_df)
    total_repeat = 0 if repeat_df is None else len(repeat_df)
    excluded_total = 0 if excluded_df is None else len(excluded_df)
    summary_rows = [
        {"항목": "교체 발생(중복제거 후)", "값": total_occ},
        {"항목": "반복 Line Number(2회 이상)", "값": total_repeat},
        {"항목": "제외 행 수", "값": excluded_total},
    ]
    if excluded_df is not None and not excluded_df.empty:
        for reason, cnt in excluded_df["exclude_reason"].value_counts().items():
            summary_rows.append({"항목": f"제외 - {reason}", "값": int(cnt)})
    return pd.DataFrame(summary_rows)


def export_piping_repeat_report(
    history_file: str | Path,
    trouble_file: str | Path,
    output_path: str | Path,
) -> str:
    occurrences_df, excluded_df = load_piping_replacement_occurrences(history_file, trouble_file)
    repeat_df = build_piping_repeat_report_dataframe(occurrences_df)
    summary_df = build_piping_summary_sheet(occurrences_df, excluded_df, repeat_df)

    output_path = str(output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        repeat_df.to_excel(writer, index=False, sheet_name="repeat_lines")
        (occurrences_df if occurrences_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="occurrences")
        (excluded_df if excluded_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="excluded_review")
    return output_path
