"""Solver agent that combines RAG context + symbolic tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re

from rag.retriever import RAGRetriever
from tools.python_math_tool import solve_expression, evaluate_expression


@dataclass
class SolverResult:
    plan: List[str]
    steps: List[str]
    final_answer: str
    retrieved_context: List[Dict]


class SolverAgent:
    """Produce plan and solution using RAG + symbolic helper."""

    def __init__(self):
        try:
            self.retriever = RAGRetriever()
            self.retriever_error = None
        except Exception as exc:  # noqa: BLE001
            self.retriever = None
            self.retriever_error = str(exc)

    # -------------------------
    # Expression normalization
    # -------------------------
    def normalize_expression(self, expr: str) -> str:
        """
        Convert human math into SymPy-compatible form.
        Examples:
        x2 -> x**2
        5x -> 5*x
        x(x+1) -> x*(x+1)
        """
        expr = expr.replace("^", "**")
        expr = expr.replace("−", "-")

        # x2 → x**2 (single-letter variables)
        expr = re.sub(r"\b([a-zA-Z])(\d+)\b", r"\1**\2", expr)

        # 5x → 5*x
        expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)

        # x(x+1) → x*(x+1)
        expr = re.sub(r"([a-zA-Z0-9])\(", r"\1*(", expr)
        # )( -> )*(
        expr = re.sub(r"\)\(", ")*(", expr)

        expr = re.sub(r"\s+", "", expr)

        return expr

    # -------------------------
    # Extract equation
    # -------------------------
    def extract_equation(self, question: str) -> str | None:
        """
        Extract equation from question.
        Supports:
        solve for x: x2-5x+6=0
        x^2 - 5x + 6 = 0
        """

        # case 1: explicit solve instruction
        solve_match = re.search(r"solve\s*for\s*x\s*:\s*([^\n]+)", question.lower())
        if solve_match:
            return solve_match.group(1)

        # case 2: parse around '=' and keep only math-like segments.
        if "=" in question:
            left_raw, right_raw = question.rsplit("=", 1)

            left = self._extract_math_side(left_raw, from_end=True)
            right = self._extract_math_side(right_raw, from_end=False)

            equation = f"{left}={right}".strip("=")
            if re.search(r"[A-Za-z]", equation) and re.search(r"\d|\*|\+|\-|\^|/|\(|\)", equation):
                return equation

        # case 3: equation-like inline segment
        eq_match = re.search(r"([A-Za-z0-9\*\+\-\^\(\)\./\s]*[A-Za-z]\d*[A-Za-z0-9\*\+\-\^\(\)\./\s]*=[A-Za-z0-9\*\+\-\^\(\)\./\s]+)", question)
        if eq_match:
            return eq_match.group(1).strip()

        return None

    def _extract_math_side(self, text: str, from_end: bool) -> str:
        """Extract likely math segment from one side of an equation."""
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        # Candidate spans with math-safe chars.
        candidates = re.findall(r"[A-Za-z0-9\(\)\[\]\{\}\*\+\-\^\./\s]+", cleaned)
        candidates = [c.strip() for c in candidates if c.strip()]

        def candidate_score(c: str) -> tuple:
            math_chars = sum(ch.isdigit() or ch in "+-*/^()[]{}" for ch in c)
            has_alpha = any(ch.isalpha() for ch in c)
            has_operator = any(ch in "+-*/^" for ch in c)
            has_paren = any(ch in "()" for ch in c)
            return (math_chars, int(has_alpha), int(has_operator or has_paren), len(c))

        best = ""
        pool = candidates[::-1] if from_end else candidates
        for cand in pool:
            # Remove common non-math prompt words.
            cand = re.sub(
                r"\b(find|roots?|solve|equation|for|the|value|of|what|is|determine|please)\b",
                " ",
                cand,
                flags=re.IGNORECASE,
            )
            cand = re.sub(r"\s+", " ", cand).strip(" :-,")
            if not cand:
                continue

            if candidate_score(cand) > candidate_score(best):
                best = cand

            # Prefer concise trailing math-like tail for left side.
            if from_end:
                tail_match = re.search(r"([A-Za-z0-9\(\)\[\]\{\}\*\+\-\^\./\s]+)$", cand)
                if tail_match:
                    tail = tail_match.group(1).strip()
                    if candidate_score(tail) > candidate_score(best):
                        best = tail

        return best or cleaned

    # -------------------------
    # Main solver
    # -------------------------
    def run(self, parsed_problem: Dict, strategy: str) -> SolverResult:

        question = parsed_problem["problem_text"]

        retrieved = []
        if self.retriever is not None:
            try:
                retrieved = self.retriever.retrieve(question, top_k=4)
            except Exception:  # noqa: BLE001
                retrieved = []

        retrieved_ctx = [
            {
                "source": r.metadata.get("source", "unknown"),
                "score": r.score,
                "content": r.content,
            }
            for r in retrieved
        ]

        plan = [
            f"Use strategy: {strategy}",
            "Retrieve relevant formulas and pitfalls",
            "Apply symbolic manipulation and compute answer",
        ]

        steps: List[str] = []
        answer = "Could not derive a final answer automatically."

        # --------------------------------
        # Extract equation from question
        # --------------------------------
        expr = self.extract_equation(question)

        if expr:

            normalized = self.normalize_expression(expr)
            var_match = re.search(r"[a-zA-Z]", normalized)
            solve_var = var_match.group(0) if var_match else "x"

            steps.append(f"Parsed equation: {normalized}")

            res = solve_expression(normalized, variable=solve_var)

            if res.success:
                steps.append(f"Solved roots using SymPy: {res.output}")
                answer = str(res.output)

        else:

            # --------------------------------
            # Try simplify/evaluate problems
            # --------------------------------
            eval_match = re.search(r"simplify\s*:\s*(.+)$", question.lower())

            if eval_match:
                expr = eval_match.group(1)
                normalized = self.normalize_expression(expr)

                res = evaluate_expression(normalized)

                if res.success:
                    steps.append(f"Simplified expression: {normalized}")
                    answer = str(res.output)

        # --------------------------------
        # fallback explanation
        # --------------------------------
        if not steps:
            steps.append(
                "Used retrieved math context to build a conceptual solution path."
            )
            answer = (
                "Please review the explanation and verify with HITL for final confidence."
            )

        return SolverResult(
            plan=plan,
            steps=steps,
            final_answer=answer,
            retrieved_context=retrieved_ctx,
        )
