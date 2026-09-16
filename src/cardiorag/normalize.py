"""Input normalization: ZH-EN term alignment, patient factor extraction, emergency keyword detection.

Corresponds to block ① in the CardioRAG architecture diagram.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ── Emergency keyword sets (ZH + EN) ──────────────────────────
# Clinical basis (see data/seed/sources.json for full citation chain):
#   * Chest pain + cold sweat / worsening / crushing / radiating pain → ACS red flags
#     (ESC 2023 ACS §5 presentation; ESC 2024 CCS §4 symptom recognition)
#   * Sudden dyspnea / syncope / sudden collapse → hemodynamic instability or
#     cardiac-arrest prodrome requiring immediate routing (ESC 2023 ACS §5.2;
#     AHA 2020 CPR & ECC guidelines basic-emergency triage)
#   * Nitroglycerin-ineffective / persistent pain → ongoing ischemia (ESC 2024 CCS §5)
#   * Unresponsive / not breathing → cardiac arrest (AHA 2020 CPR & ECC §1)
#   * Bleeding red flags on anticoagulants (melena/hematuria/gum bleeding/extensive
#     ecchymosis/hematemesis) → major bleeding risk requiring immediate routing
#     (ESC 2024 AF §10 DOAC bleeding management; ESC 2020 NSTE-ACS §10)
#   * Stroke-like symptoms (one-sided weakness / slurred speech / facial droop)
#     → acute stroke FAST triage (AHA 2019 ischemic stroke guidelines)
#   * Head trauma on anticoagulants → intracranial hemorrhage risk (ESC 2024 AF §10)
# These keywords implement the binary high-risk routing contract declared in
# dataset.meta.json; they are intentionally conservative (high-recall over precision).
EMERGENCY_ZH = [
    "胸痛.*冷汗", "胸痛加重", "胸口.*疼.*冷汗", "胸口.*压迫",
    "突发呼吸困难", "晕厥", "心搏骤停", "心脏骤停",
    "剧烈胸痛", "急性呼吸困难", "突然晕倒", "心跳停止",
    "硝酸甘油.*无效", "持续.*(胸|心).*疼", "压榨.*(胸|心)",
    "叫不醒", "没呼吸", "没有呼吸", "无呼吸",
    "出冷汗.*胸", "大汗.*胸口", "冷汗.*30",
    # ── v2: bleeding red flags (anticoagulant context) ──
    "黑便", "大便.{0,3}(黑|发黑)", "柏油便", "柏油样|柏油一样", "呕血", "吐血",
    "血尿", "尿血", "小便.{0,3}红",
    "牙龈.*(一直|持续|反复|不止).*出血|牙龈.*出血.*(不止|不停)", "出血不止", "大面积瘀斑|瘀斑.*(扩大|加重|蔓延)",
    "咯血", "咳血",
    # ── v2: stroke-like symptoms (FAST) ──
    "单侧.{0,4}无力", "一侧.{0,4}无力", "半身不遂", "偏瘫",
    "说话不清", "言语不清", "吐字不清", "口角歪斜", "嘴歪",
    # ── v2: head trauma (anticoagulant context) ──
    "撞到.{0,4}头", "头撞", "摔.{0,3}(跤|倒).{0,3}头", "头部外伤", "摔.*撞.*头",
]
EMERGENCY_EN = [
    r"chest pain.*radiating", r"sudden dyspnea", r"syncope",
    r"signs of cardiac arrest", r"severe chest pain",
    r"acute shortness of breath", r"sudden collapse",
    # ── v2: bleeding red flags ──
    r"melena", r"black stool", r"hematemesis", r"vomiting blood",
    r"hematuria", r"blood in (my )?urine", r"gum bleeding",
    r"bleeding.*gums", r"uncontrolled bleeding", r"extensive bruis",
    r"ecchymos", r"hemoptysis", r"coughing blood",
    # ── v2: stroke-like symptoms ──
    r"one.sided weakness", r"weakness.*one side", r"facial droop",
    r"slurred speech", r"drooping face",
    # ── v2: head trauma ──
    r"hit.*head", r"head injury", r"fell.*head", r"bump.*head",
]

# ── Out-of-bounds categories ──────────────────────────────────
# Clinical basis (see data/seed/sources.json for full citation chain):
#   * dose_adjustment: medication dose changes require physician evaluation of
#     renal function, interactions, and comorbidity — ESC 2024 AF §11 (DOAC dose
#     by CrCl/HAS-BLED), AHA/ACC/HFSA 2022 HF §7 (GDMT titration)
#   * medication_discontinuation: stopping antiplatelet/anticoagulant/GDMT is a
#     physician decision with thrombotic/rebound risk — ESC 2024 CCS §10,
#     ESC 2023 ACS §10 discharge strategy
#   * procedure_decision: PCI/CABG/ablation suitability is a Heart-Team decision —
#     ESC 2024 CCS §7, ESC 2024 AF §9 (AF-CARE pillar A), ESC 2024 CCS revascularization
# These boundaries implement the patient-safety out-of-scope contract (guardrail
# refuses, never answers a specific dose/stop/procedure recommendation).
# v2.3: bare-substring matching of 停药/停止服用 fired on safe educational
# prose ("擅自停药风险极高" / "关于是否停药，必须由医生决定"), so the verb
# must now be DIRECTLY PRECEDED by a directive modal/verb (建议你/应该/可以…).
# Question markers (是否/能否/可否) and hedged statements are not directives.
OOB_DOSE_ADJUST = r"((?:你|您).*(?:加[到至]|减[到至]|增加[到至]|减少[到至]|改[成为]|调整[成为])\s*\d+(?:\.\d+)?\s*(?:mg|片|粒|次|毫升|ml|克|g))"
OOB_STOP_MED = (
    r"(建议(?:你|您)?.{0,10}(?:停用|停药|停止服用)"
    r"|(?:应该|可以|需要|务必|记得|最好|赶紧|尽快)(?:停用|停药|停止服用))"
)
OOB_SURGERY = r"(你需要做.*(PCI|CABG|消融|手术|支架|搭桥|介入))"
OOB_EN = r"(you should (increase|decrease|adjust|stop|discontinue).*(dose|medication|drug))"

OOB_PATTERNS = [OOB_DOSE_ADJUST, OOB_STOP_MED, OOB_SURGERY, OOB_EN]

# ── v2.2 polarity-aware OOB detection ─────────────────────────
# The v2.1 detector used bare substring matches (停药 / 停止服用) that fired on
# SAFE, prohibitive advice such as "不建议您自行停药" or "请勿减量", causing the
# post-generation guardrail to replace correct educational answers with an empty
# refusal template (false-positive OOB). A negation window before each match
# disambiguates directive advice (OOB) from prohibitive/safe advice (not OOB).
OOB_NEGATION_ZH = (
    "不建议", "请不要", "请勿", "切勿", "千万别", "绝对不要", "不应该", "不应",
    "不要", "不能", "不可以", "不可", "避免", "无需", "勿", "不宜", "最好别",
)
OOB_NEGATION_EN = (
    "do not", "don't", "should not", "shouldn't", "must not", "mustn't",
    "avoid", "never", "without", "not stop", "not discontinue", "not adjust",
)
# How many characters before a match to scan for a negation marker. Chinese
# negation phrases sit immediately before the verb; 14 covers "不建议您自行停药".
_OOB_NEGATION_WINDOW = 14
# v2.3: physician-deferral wording ("无法直接建议" / "必须由医生决定") that
# PRECEDES a matched directive phrase means the answer is DECLINING to give
# the directive — safe educational content, not out-of-bounds advice.
OOB_DECLINE_ZH = (
    "不能直接", "无法直接", "我不能", "无法", "必须由", "由医生", "由您的",
    "主治医生", "是否停药", "能否停药", "可否停药",
)
# Characters that directly precede 停药/停用 in safe contexts (擅自/自行/想/
# 以免/防止/不能/是否…), i.e. the verb is not a directive to the patient.
_OOB_SAFE_PRECEDING = ("不", "别", "勿", "没", "免", "止", "防", "否")


def _is_oob_negated(response: str, match_start: int, match_end: int) -> bool:
    """True if the span [match_start, match_end) is non-directive/safe.

    Negated advice ("请勿停药" / "do not stop"), physician-deferral wording
    ("是否停药必须由医生决定"), and non-directive subjects ("擅自停药风险高")
    are SAFE guidance and must NOT be treated as out-of-bounds directive advice.
    """
    lo = max(0, match_start - _OOB_NEGATION_WINDOW)
    prefix = response[lo:match_start]
    prefix_l = prefix.lower()
    # Bare negation/safety char immediately before the match handles fused forms
    # such as "不应该停用" / "以免停药" / "防止停药".
    if prefix and prefix[-1] in _OOB_SAFE_PRECEDING:
        return True
    # The matched directive phrase itself may sit inside a deferral clause:
    # "关于是否停药…" / "我不能直接给您停药或继续服药的指令".
    inner = response[match_start:match_end]
    if any(d in inner for d in ("是否", "能否", "可否")):
        return True
    # Question-echo spans: "想知道是否可以停药" — the interrogative marker sits
    # just before the captured verb phrase (是否|能否|可否 immediately precedes,
    # or the filler 道/问 introduces an embedded question).
    if inner.startswith(("是否", "能否", "可否")):
        return True
    if prefix.endswith(("道", "问", "咨询", "了解")):
        return True
    # A negation INSIDE the match ("建议不要停药") makes it prohibitive, safe.
    if any(n in inner for n in OOB_NEGATION_ZH):
        return True
    # A wider window catches refusals stated shortly before the match
    # ("我无法直接建议您停药或继续服药").
    wide = response[max(0, match_start - 30):match_start]
    if any(d in wide for d in OOB_DECLINE_ZH):
        return True
    return any(n in prefix for n in OOB_NEGATION_ZH) or any(
        n in prefix_l for n in OOB_NEGATION_EN
    )


def load_term_map(seed_dir: Path) -> dict[str, str]:
    """Load ZH→EN term alignment map from data/seed/term_map.json."""
    map_path = seed_dir / "term_map.json"
    if map_path.exists():
        return json.loads(map_path.read_text(encoding="utf-8"))
    return {}


def normalize_terms(text: str, term_map: dict[str, str]) -> str:
    """Align ZH medical terms to EN equivalents for KG retrieval."""
    for zh_term, en_term in term_map.items():
        text = text.replace(zh_term, f"{zh_term}({en_term})")
    return text


def extract_patient_factors(text: str) -> dict[str, Any]:
    """Extract structured patient factors from free-text input.

    Detects: age, LVEF, CHA₂DS₂-VASc, CrCl, NT-proBNP, cTnI, BP, HR.
    """
    factors: dict[str, Any] = {}

    # Age
    age_match = re.search(r"(?:年龄|age)[:：\s]*(\d+)|(\d+)\s*岁", text, re.IGNORECASE)
    if age_match:
        age_val = age_match.group(1) or age_match.group(2)
        factors["age"] = int(age_val)

    # LVEF
    lvef_match = re.search(r"(?:LVEF|射血分数)[:：\s]*(\d+)[%％]", text, re.IGNORECASE)
    if lvef_match:
        factors["LVEF"] = f"{lvef_match.group(1)}%"

    # CHA₂DS₂-VASc
    chads_match = re.search(
        r"(?:CHA[₂2]DS[₂2][- ]?VASc|房颤卒中评分)\D*(\d+)", text, re.IGNORECASE
    )
    if chads_match:
        factors["CHA2DS2_VASc"] = int(chads_match.group(1))

    # CrCl
    crcl_match = re.search(r"(?:CrCl|肌酐清除率|eGFR)[:：\s]*(\d+)", text, re.IGNORECASE)
    if crcl_match:
        factors["CrCl_or_eGFR"] = int(crcl_match.group(1))

    # NT-proBNP
    ntprobnp_match = re.search(
        r"(?:NT-proBNP|脑钠肽)[:：\s]*(\d+)", text, re.IGNORECASE
    )
    if ntprobnp_match:
        factors["NT_proBNP"] = int(ntprobnp_match.group(1))

    # cTnI
    ctni_match = re.search(r"(?:cTnI|肌钙蛋白)[:：\s]*([\d.]+)", text, re.IGNORECASE)
    if ctni_match:
        factors["cTnI"] = float(ctni_match.group(1))

    return factors


def detect_emergency(text: str) -> tuple[bool, list[str]]:
    """Detect emergency keywords in input text.

    Returns (is_emergency, matched_keywords).
    """
    matched = []
    for pattern in EMERGENCY_ZH + EMERGENCY_EN:
        if re.search(pattern, text, re.IGNORECASE):
            matched.append(pattern)
    return len(matched) > 0, matched


def detect_out_of_bounds(response: str) -> tuple[bool, list[str]]:
    """Detect out-of-bounds content in LLM response (polarity-aware, v2.3).

    A pattern match only counts as out-of-bounds when it is DIRECTIVE advice,
    i.e. NOT preceded by a negation marker, NOT embedded in a physician-
    deferral/question clause, and NOT headed by a non-directive subject. This
    prevents safe educational answers ("不建议您自行停药" / "擅自停药风险极高" /
    "是否停药必须由医生决定") from being misclassified as OOB.

    Returns (is_oob, matched_categories).
    """
    matched = []
    for pattern in OOB_PATTERNS:
        for m in re.finditer(pattern, response, re.IGNORECASE):
            if not _is_oob_negated(response, m.start(), m.end()):
                matched.append(pattern)
                break  # one non-negated match is enough for this pattern
    return len(matched) > 0, matched


def normalize_input(
    text: str, term_map: dict[str, str] | None = None
) -> dict[str, Any]:
    """Full normalization pipeline for a patient/physician query.

    Returns a dict with normalized text, patient factors, and emergency flags.
    """
    term_map = term_map or {}
    normalized = normalize_terms(text, term_map)
    factors = extract_patient_factors(text)
    is_emergency, emergency_kws = detect_emergency(text)

    return {
        "original_text": text,
        "normalized_text": normalized,
        "patient_factors": factors,
        "is_emergency": is_emergency,
        "emergency_keywords": emergency_kws,
        "language": "zh" if _is_chinese(text) else "en",
    }


def _is_chinese(text: str) -> bool:
    """Heuristic: Chinese characters > English alphabet characters → Chinese."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    alpha = sum(1 for c in text if c.isalpha() and c.isascii())
    return cjk > alpha
