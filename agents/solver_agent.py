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
        self.retriever = RAGRetriever()

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

        # x2 → x**2
        expr = re.sub(r"x(\d+)", r"x**\1", expr)

        # 5x → 5*x
        expr = re.sub(r"(\d)x", r"\1*x", expr)

        # x(x+1) → x*(x+1)
        expr = re.sub(r"x\(", r"x*(", expr)

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

        # case 2: direct equation
        eq_match = re.search(r"([\w\*\+\-\^\(\)\s/]+=[\w\*\+\-\^\(\)\s/]+)", question)
        if eq_match:
            return eq_match.group(1)

        return None

    # -------------------------
    # Main solver
    # -------------------------
    def run(self, parsed_problem: Dict, strategy: str) -> SolverResult:

        question = parsed_problem["problem_text"]

        retrieved = self.retriever.retrieve(question, top_k=4)

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

            steps.append(f"Parsed equation: {normalized}")

            res = solve_expression(normalized)

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