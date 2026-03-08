"""Parser agent for cleaning and structuring math problems."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import re


TOPIC_KEYWORDS = {
    "probability": ["probability", "random", "dice", "coin", "event"],
    "calculus": ["derivative", "integral", "limit", "differentiate", "dx"],
    "linear algebra": ["matrix", "determinant", "vector", "eigen"],
    "algebra": ["solve", "equation", "quadratic", "polynomial", "factor"],
}


@dataclass
class ParsedProblem:
    problem_text: str
    topic: str
    variables: List[str]
    constraints: List[str]
    needs_clarification: bool

    def to_dict(self) -> Dict:
        return asdict(self)


class ParserAgent:
    """Parse raw OCR/ASR/text input to structured JSON."""

    def run(self, raw_text: str) -> ParsedProblem:
        cleaned = self._clean_text(raw_text)
        vars_found = sorted(set(re.findall(r"\b[a-zA-Z]\b", cleaned)))
        constraints = self._extract_constraints(cleaned)
        topic = self._detect_topic(cleaned)
        ambiguous = len(cleaned) < 8 or "?" not in cleaned and "solve" not in cleaned.lower()

        return ParsedProblem(
            problem_text=cleaned,
            topic=topic,
            variables=vars_found,
            constraints=constraints,
            needs_clarification=ambiguous,
        )

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.replace("5l", "51")
        return text.strip()

    def _extract_constraints(self, text: str) -> List[str]:
        patterns = [r"[a-zA-Z]\s*[<>]=?\s*-?\d+", r"[a-zA-Z]\s*!=\s*-?\d+"]
        constraints = []
        for pat in patterns:
            constraints.extend(re.findall(pat, text))
        return constraints

    def _detect_topic(self, text: str) -> str:
        lower = text.lower()
        for topic, words in TOPIC_KEYWORDS.items():
            if any(w in lower for w in words):
                return topic
        return "algebra"
