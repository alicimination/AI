"""Safe-ish math execution helper using SymPy for symbolic tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import sympy as sp


@dataclass
class MathToolResult:
    success: bool
    output: Any
    error: str = ""


def solve_expression(expression: str, variable: str = "x") -> MathToolResult:
    """Solve equations like x**2 - 5*x + 6 = 0."""
    try:
        sym_var = sp.symbols(variable)
        left, right = expression.split("=")
        eq = sp.Eq(sp.sympify(left), sp.sympify(right))
        roots = sp.solve(eq, sym_var)
        return MathToolResult(success=True, output=roots)
    except Exception as exc:
        return MathToolResult(success=False, output=None, error=str(exc))


def evaluate_expression(expression: str) -> MathToolResult:
    """Evaluate a SymPy-compatible expression."""
    try:
        result = sp.simplify(sp.sympify(expression))
        return MathToolResult(success=True, output=result)
    except Exception as exc:
        return MathToolResult(success=False, output=None, error=str(exc))


def get_allowed_functions() -> Dict[str, Any]:
    """Expose supported operations for UI/tooling transparency."""
    return {
        "solve_expression": "Solve equation in one variable",
        "evaluate_expression": "Simplify/evaluate symbolic expression",
    }
