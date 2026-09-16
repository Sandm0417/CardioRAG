"""CV-Safety Guardrail: emergency routing + out-of-bounds refusal.

Corresponds to block ⑤ in the CardioRAG architecture diagram.
Implements both input-side (pre-generation) and output-side (post-generation) filtering.
"""

from __future__ import annotations

import re
from typing import Any

from .normalize import detect_emergency, detect_out_of_bounds

# ── Emergency routing templates ───────────────────────────────
EMERGENCY_RESPONSE_ZH = (
    "🚨 **紧急提醒**：根据您的描述（{keywords}），这可能是需要立即就医的紧急情况。"
    "请立即拨打 120 或前往最近急诊科。"
    "心肺复苏（CPR）指引：如患者无意识且无正常呼吸，请立即开始胸外按压（100-120次/分钟，深度5-6cm），"
    "并使用附近 AED。\n\n"
    "⚠️ 本系统为患者沟通辅助工具，不能替代紧急医疗判断。"
)

EMERGENCY_RESPONSE_EN = (
    "🚨 **EMERGENCY ALERT**: Based on your description ({keywords}), "
    "this may be a medical emergency requiring immediate attention. "
    "Please call 911 (or your local emergency number) or go to the nearest ER immediately. "
    "If the person is unconscious and not breathing normally, begin CPR "
    "(100-120 compressions/min, 5-6 cm depth) and use an AED if available.\n\n"
    "⚠️ This is a patient communication aid, not a substitute for emergency medical judgment."
)

# ── Out-of-bounds refusal templates ───────────────────────────
OOB_RESPONSE_ZH = (
    "这类决定需要您的心血管专科医师根据您的完整病情（包括体格检查、化验结果、"
    "既往史和药物相互作用）综合评估后做出。建议尽快预约门诊，与医师当面讨论。\n\n"
    "以下是根据指南提供的背景信息，仅供参考：\n{context}"
)

OOB_RESPONSE_EN = (
    "This type of decision requires comprehensive evaluation by your cardiologist "
    "based on your complete clinical picture (physical exam, lab results, "
    "medical history, and drug interactions). Please schedule an appointment "
    "to discuss with your physician.\n\n"
    "The following background information from guidelines is provided for reference only:\n{context}"
)

# ── Emergency routing phrases in LLM output (answer-side detection) ──
# When the original answer already routes the patient to emergency care, the
# post-generation sanitizer must PRESERVE that routing (v2 policy): replacing the
# whole answer with a refusal template would erase life-saving instructions.
EMERGENCY_ROUTING_ZH = (
    r"拨打\s*120|call\s*911|急救电话|立即拨打|前往.{0,6}急诊|去急诊|到急诊|急诊科|急救中心|emergency|🚨"
)
EMERGENCY_ROUTING_EN = (
    r"call\s*911|emergency room|go to the ER|nearest ER|emergency department|🚨"
)

# ── Strong (unconditional) emergency routing phrases ──
# v2.1 sanitize policy: the answer is preserved ONLY when it contains an
# unconditional, imperative emergency routing ("请立即拨打120"/"立即前往急诊").
# Conditional advice ("如果出现胸痛，请立即就医") or soft suggestions
# ("尽快前往门诊就医") do NOT preserve — the refusal template is used instead,
# because those cases are clinically non-urgent and preserving routing would
# over-triage them as high-risk (false positives in evaluation).
STRONG_ROUTING_ZH = (
    r"(?:立即|马上|立刻|请尽快).{0,8}(?:拨打\s*120|前往.{0,6}急诊|去急诊|到急诊|急诊科)|拨打\s*120"
)
STRONG_ROUTING_EN = (
    r"(?:immediately|right away|right now).{0,24}(?:call|go to|nearest ER|911)|call\s*911"
)

# Conditional routing clauses ("如果…请拨打120") are advisory, not actual
# emergencies; strip them before strong-routing detection so that OOB answers
# with only conditional advice are NOT preserved (v2.1).
_CONDITIONAL_ROUTE_RE = re.compile(
    r"(?:如果|若|如|一旦|when|if)[^。；;\n]{0,40}?(?:拨打\s*120|前往.{0,6}急诊|去急诊|到急诊|急诊科|emergency|call\s*911)",
    re.IGNORECASE,
)

# ── Physician-referral note appended when routing is preserved ──
OOB_NOTE_ZH = (
    "\n\n---\n*注：您的问题还涉及药物剂量调整/停药/手术等专业决策，"
    "具体方案必须由您的医师结合完整病情（肾功能、化验、相互作用等）评估后决定，"
    "切勿自行调整。以上紧急就医提醒依然有效，请优先遵照执行。*"
)

OOB_NOTE_EN = (
    "\n\n---\n*Note: Your question also involves medication dose adjustment / "
    "discontinuation / procedure decisions that must be made by your physician "
    "based on your complete clinical picture (renal function, labs, interactions). "
    "Do not adjust on your own. The emergency guidance above remains in effect; "
    "please follow it first.*"
)

# ── Disclaimer (appended to all responses) ────────────────────
DISCLAIMER_ZH = (
    "\n\n---\n*免责声明：本回答由 AI 根据临床指南生成，仅供患者教育参考。"
    "所有临床决策（包括用药调整、手术选择）请务必咨询执业医师。"
    "如出现胸痛、呼吸困难、晕厥等紧急症状，请立即就医。*"
)

DISCLAIMER_EN = (
    "\n\n---\n*Disclaimer: This response is AI-generated from clinical guidelines "
    "for patient education purposes only. All clinical decisions "
    "(including medication adjustments and procedure choices) must be discussed "
    "with a licensed physician. Seek immediate medical attention for chest pain, "
    "shortness of breath, syncope, or other emergency symptoms.*"
)


class CVGuardrail:
    """Cardiovascular-specific safety guardrail.

    Pre-generation: emergency keyword detection → bypass LLM, route to 120/911.
    Post-generation: out-of-bounds detection → replace with refusal + physician referral.
    """

    def __init__(self):
        self.emergency_triggered = 0
        self.oob_triggered = 0
        self.total_processed = 0

    def pre_generation_check(self, user_input: str) -> dict[str, Any]:
        """Check input BEFORE LLM generation.

        Returns dict with 'block' (bool), 'response' (str|None), 'metadata'.
        """
        is_emergency, keywords = detect_emergency(user_input)
        is_zh = _is_chinese(user_input)

        result: dict[str, Any] = {
            "block": False,
            "response": None,
            "metadata": {
                "emergency_detected": is_emergency,
                "emergency_keywords": keywords,
            },
        }

        if is_emergency:
            self.emergency_triggered += 1
            template = EMERGENCY_RESPONSE_ZH if is_zh else EMERGENCY_RESPONSE_EN
            result["block"] = True
            result["response"] = template.format(keywords=", ".join(keywords))

        return result

    def post_generation_check(
        self, response: str, context: str = ""
    ) -> dict[str, Any]:
        """Check LLM output AFTER generation.

        v2 sanitize policy: if the original answer already contains emergency
        routing phrases (120/911/ER), keep the answer and append a physician-
        referral note instead of replacing the whole text — otherwise the
        life-saving routing would be lost and high-risk cases mis-triaged.

        Returns dict with 'flagged' (bool), 'sanitized_response' (str), 'metadata'.
        """
        is_oob, patterns = detect_out_of_bounds(response)
        is_zh = _is_chinese(response)
        # v2.1: preserve only when an UNCONDITIONAL strong routing is present;
        # conditional clauses ("如果…请拨打120") are stripped first.
        stripped = _CONDITIONAL_ROUTE_RE.sub("", response)
        has_routing = bool(
            re.search(
                STRONG_ROUTING_ZH if is_zh else STRONG_ROUTING_EN,
                stripped,
                re.IGNORECASE,
            )
        )

        result: dict[str, Any] = {
            "flagged": False,
            "sanitized_response": response,
            "metadata": {
                "oob_detected": is_oob,
                "oob_patterns": patterns,
                "oob_categories": _classify_oob(patterns),
                "routing_preserved": has_routing,
            },
        }

        if is_oob:
            self.oob_triggered += 1
            if has_routing:
                note = OOB_NOTE_ZH if is_zh else OOB_NOTE_EN
                result["flagged"] = True
                result["sanitized_response"] = response + note
            else:
                template = OOB_RESPONSE_ZH if is_zh else OOB_RESPONSE_EN
                result["flagged"] = True
                result["sanitized_response"] = template.format(context=context)

        return result

    def append_disclaimer(self, response: str) -> str:
        """Append legal disclaimer to every response."""
        is_zh = _is_chinese(response)
        return response + (DISCLAIMER_ZH if is_zh else DISCLAIMER_EN)

    def full_guardrail(
        self, user_input: str, llm_response: str, context: str = ""
    ) -> dict[str, Any]:
        """Run complete guardrail pipeline.

        Returns the final output with all safety metadata.
        """
        self.total_processed += 1

        # Pre-generation check
        pre = self.pre_generation_check(user_input)
        if pre["block"]:
            return {
                "response": pre["response"],
                "safety_metadata": {
                    **pre["metadata"],
                    "oob_detected": False,
                    "oob_categories": [],
                    "blocked_by_emergency": True,
                },
            }

        # Post-generation check
        post = self.post_generation_check(llm_response, context)
        final_response = (
            post["sanitized_response"] if post["flagged"] else llm_response
        )
        final_response = self.append_disclaimer(final_response)

        return {
            "response": final_response,
            "safety_metadata": {
                **pre["metadata"],
                **post["metadata"],
                "blocked_by_emergency": False,
            },
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Return guardrail statistics."""
        return {
            "total_processed": self.total_processed,
            "emergency_triggered": self.emergency_triggered,
            "oob_triggered": self.oob_triggered,
            "emergency_recall": (
                self.emergency_triggered / max(self.total_processed, 1)
            ),
            "oob_refusal_rate": (
                self.oob_triggered / max(self.total_processed, 1)
            ),
        }


def _classify_oob(patterns: list[str]) -> list[str]:
    """Classify matched OOB patterns into categories."""
    categories = []
    for p in patterns:
        if "剂量" in p or "dose" in p.lower() or "mg" in p:
            categories.append("dose_adjustment")
        elif "停" in p or "stop" in p.lower() or "discontinue" in p.lower():
            categories.append("medication_discontinuation")
        elif "手术" in p or "PCI" in p or "surgery" in p.lower():
            categories.append("procedure_decision")
    return list(set(categories)) if categories else ["unknown"]


def _is_chinese(text: str) -> bool:
    """Heuristic: Chinese characters > English alphabet characters → Chinese."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    alpha = sum(1 for c in text if c.isalpha() and c.isascii())
    return cjk > alpha
