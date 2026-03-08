"""Intent router agent to choose solver strategy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoutingResult:
    topic: str
    strategy: str


class IntentRouterAgent:
    """Classify/route parsed problem to a strategy bucket."""

    def run(self, topic: str) -> RoutingResult:
        strategy_map = {
            "algebra": "symbolic_solve",
            "probability": "counting_and_axioms",
            "calculus": "differentiate_integrate",
            "linear algebra": "matrix_operations",
        }
        strategy = strategy_map.get(topic, "symbolic_solve")
        return RoutingResult(topic=topic, strategy=strategy)
