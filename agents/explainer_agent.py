"""Explainer/tutor agent for student-friendly JEE style output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExplanationResult:
    explanation: str


class ExplainerAgent:
    """Generate final pedagogical explanation."""

    def run(self, parsed_problem: Dict, plan: List[str], steps: List[str], final_answer: str) -> ExplanationResult:
        formula_hint = {
            "algebra": "Use factorization / quadratic formula where relevant.",
            "probability": "Use P(A)=favorable/total with valid sample space.",
            "calculus": "Use derivative/integral rules and simplify carefully.",
            "linear algebra": "Use matrix identities and determinant properties.",
        }
        topic = parsed_problem.get("topic", "algebra")
        lines = [
            f"Problem: {parsed_problem.get('problem_text', '')}",
            f"Topic detected: {topic}",
            "\nStep-by-step mentor explanation:",
        ]
        for idx, item in enumerate(plan + steps, start=1):
            lines.append(f"{idx}. {item}")
        lines.append(f"\nFormula focus: {formula_hint.get(topic, formula_hint['algebra'])}")
        lines.append(f"Final Answer: {final_answer}")
        lines.append("Reasoning tip: Always check domains, signs, and edge cases before finalizing.")
        return ExplanationResult(explanation="\n".join(lines))
