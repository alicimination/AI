"""Verifier/critic agent for sanity and domain checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VerificationResult:
    passed: bool
    confidence: float
    checks: List[str]
    needs_hitl: bool


class VerifierAgent:
    """Check solver output for validity and obvious constraints."""

    def run(self, parsed_problem: Dict, solver_answer: str) -> VerificationResult:
        checks: List[str] = []
        confidence = 0.75
        needs_hitl = False

        text = parsed_problem.get("problem_text", "").lower()
        if "probability" in text:
            checks.append("Probability answer should be in [0,1].")
            if any(token in solver_answer for token in ["-", "2", "3", "4", "5", "6", "7", "8", "9"]):
                confidence -= 0.2
                needs_hitl = True

        if "log(" in text:
            checks.append("Ensure log argument is positive.")

        if "/" in text:
            checks.append("Ensure denominator is non-zero.")

        if parsed_problem.get("needs_clarification"):
            checks.append("Parser flagged ambiguity.")
            confidence -= 0.3
            needs_hitl = True

        passed = confidence >= 0.6
        return VerificationResult(passed=passed, confidence=max(0.0, confidence), checks=checks, needs_hitl=needs_hitl)
