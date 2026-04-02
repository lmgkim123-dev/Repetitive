from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

_EQ_RE = re.compile(r"(?<![A-Z0-9])(\d{2,3})\s*[-_ ]?\s*([A-Z]{1,3})\s*[-_ ]?\s*(\d{3,4}[A-Z]?)(?![A-Z0-9])", re.I)
_NAME_NOISE_RE = re.compile(
    r"검사일|차기검사예정일|검사구분|상세내용|차기고려사항|등록일|공정담당자|검사원|발생년도|발췌\s*category|TA\s*조치사항|추후\s*권고사항|"
    r"점검\s*결과|검사\s*결과|확인됨|확인되었|발생\s*확인|양호한\s*상태|양호함|필요|요망|검토|실시|진행|부식|감육|균열|pitting|corrosion|"
    r"연결\s*nozzle|grid\s*ut|scanning|thickness|두께\s*측정|정밀\s*두께|보수작업|교체여부",
    re.I,
)
_SENTENCE_LIKE_RE = re.compile(
    r"확인|발생|진행|실시|필요|요망|검토|판단|측정|부착|고착|양호|보수|교체|용접|도장|repair|replace|inspect|confirm|found|observed",
    re.I,
)


def normalize_equipment_no(text):
    t = str(text or "").upper().strip()
    if not t:
        return ""
    t = t.replace("–", "-").replace("—", "-").replace("_", "-")
    m = _EQ_RE.search(t)
    if not m:
        return ""
    return f"{m.group(1)}{m.group(2)}-{m.group(3)}"



def _clean_candidate(name, eq_norm=""):
    t = str(name or "").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    if not t or t.lower() in {"nan", "n/a", "none"}:
        return ""
    if eq_norm:
        t = re.sub(re.escape(eq_norm), " ", t, flags=re.I)
        # also drop expanded form 85-C-303 if eq_norm is 85C-303
        t = re.sub(r"(?i)\b" + re.escape(re.sub(r"^(\d{2,3})([A-Z]{1,3})-(\d{3,4}[A-Z]?)$", r"\1-\2-\3", eq_norm)) + r"\b", " ", t)
    t = re.sub(r"^[\[(<\s]*(?:no\.?\s*\d+|\d+[.)]|[-*•])\s*", "", t, flags=re.I)
    t = re.sub(r"^[A-Z]\s*[:：]\s*", "", t)
    t = re.sub(r"\s+", " ", t).strip(" -/:;,.[]()")
    if not t or len(t) < 3 or len(t) > 80:
        return ""
    if _NAME_NOISE_RE.search(t):
        return ""
    words = t.split()
    if len(words) >= 6 and _SENTENCE_LIKE_RE.search(t):
        return ""
    if re.search(r"[.!?]", t):
        return ""
    return t



def _score_candidate(name, source=""):
    t = _clean_candidate(name)
    if not t:
        return 0.0
    score = 1.0
    if source == "column_equipment_name":
        score += 5.0
    elif source == "pair_line":
        score += 3.0
    elif source == "fallback":
        score += 1.0
    if re.fullmatch(r"[A-Z0-9 .&/()'_-]+", t):
        score += 1.5
    if 3 <= len(t.split()) <= 6:
        score += 1.0
    if re.search(r"REACTOR|DRUM|EXCHANGER|COOLER|COLUMN|TOWER|VESSEL|HEATER|FILTER|SCRUBBER|SEPARATOR|ACCUMULATOR", t, re.I):
        score += 2.0
    if re.search(r"NOZZLE|PIPE|배관", t, re.I):
        score -= 2.0
    return score



def build_name_map_from_lines(lines, source_name):
    out: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
    prev = ""
    for raw in lines or []:
        line = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not line:
            continue
        eq_norm = normalize_equipment_no(line)
        if eq_norm:
            tail = line.split(eq_norm, 1)[-1].strip(" -:|")
            cand = _clean_candidate(tail, eq_norm)
            if cand:
                out[eq_norm].append((cand, "pair_line", _score_candidate(cand, "pair_line")))
            elif prev:
                prev_cand = _clean_candidate(prev, eq_norm)
                if prev_cand:
                    out[eq_norm].append((prev_cand, "prev_line", _score_candidate(prev_cand, "pair_line")))
        prev = line
    return out



def extract_names_from_text(eq_norm, text):
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if not t:
        return []
    cands = []
    if eq_norm and eq_norm in t:
        tail = t.split(eq_norm, 1)[-1].strip(" -:|")
        cand = _clean_candidate(tail, eq_norm)
        if cand:
            cands.append((cand, "inline_eq", _score_candidate(cand, "pair_line")))
    return cands



def build_equipment_name_map(d):
    out = {}
    for k, vals in (d or {}).items():
        bucket = Counter()
        score_map = defaultdict(float)
        for item in vals or []:
            if isinstance(item, tuple):
                name = _clean_candidate(item[0], k)
                score = float(item[2]) if len(item) >= 3 else _score_candidate(name, item[1] if len(item) >= 2 else "")
            else:
                name = _clean_candidate(item, k)
                score = _score_candidate(name, "fallback")
            if not name:
                continue
            bucket[name] += 1
            score_map[name] += score
        if not bucket:
            out[k] = ""
            continue
        out[k] = sorted(bucket.keys(), key=lambda n: (-score_map[n], -bucket[n], -len(n), n))[0]
    return out
