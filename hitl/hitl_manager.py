"""Human-in-the-loop manager for trigger evaluation and decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class HITLDecision:
    required: bool
    reasons: List[str]


class HITLManager:
    """Collect HITL trigger reasons from pipeline stages."""

    def evaluate(
        self,
        ocr_conf: float | None = None,
        asr_conf: float | None = None,
        parser_needs_clarification: bool = False,
        verifier_uncertain: bool = False,
        user_requested_recheck: bool = False,
        threshold: float = 0.72,
    ) -> HITLDecision:
        reasons: List[str] = []
        if ocr_conf is not None and ocr_conf < threshold:
            reasons.append(f"Low OCR confidence ({ocr_conf:.2f})")
        if asr_conf is not None and asr_conf < threshold:
            reasons.append(f"Low ASR confidence ({asr_conf:.2f})")
        if parser_needs_clarification:
            reasons.append("Parser detected ambiguity")
        if verifier_uncertain:
            reasons.append("Verifier uncertainty")
        if user_requested_recheck:
            reasons.append("User explicitly requested re-check")
        return HITLDecision(required=len(reasons) > 0, reasons=reasons)
