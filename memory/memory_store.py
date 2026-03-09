"""SQLite-backed persistent memory for solved problems and feedback."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

DB_PATH = Path("memory/math_mentor.db")


class MemoryStore:
    """Persist sessions, traces, and user feedback for self-learning."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS solved_problems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_input TEXT,
                    parsed_problem TEXT,
                    retrieved_context TEXT,
                    solution TEXT,
                    verification_result TEXT,
                    user_feedback TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incorrect_text TEXT,
                    corrected_text TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def add_record(
        self,
        original_input: str,
        parsed_problem: Dict[str, Any],
        retrieved_context: List[Dict[str, Any]],
        solution: str,
        verification_result: Dict[str, Any],
        user_feedback: str = "",
    ) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO solved_problems
                (original_input, parsed_problem, retrieved_context, solution, verification_result, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    original_input,
                    json.dumps(parsed_problem),
                    json.dumps(retrieved_context),
                    solution,
                    json.dumps(verification_result),
                    user_feedback,
                ),
            )

    def add_ocr_correction(self, incorrect_text: str, corrected_text: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO ocr_corrections (incorrect_text, corrected_text) VALUES (?, ?)",
                (incorrect_text, corrected_text),
            )


    def apply_correction_rules(self, raw_text: str, limit: int = 100) -> Tuple[str, List[Dict[str, str]]]:
        """Apply known OCR/ASR correction rules at runtime before parsing."""
        corrected = raw_text
        applied: List[Dict[str, str]] = []

        with self._conn() as con:
            rows = con.execute(
                """
                SELECT incorrect_text, corrected_text
                FROM ocr_corrections
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        for incorrect_text, corrected_text in rows:
            if not incorrect_text or not corrected_text:
                continue
            if incorrect_text in corrected:
                corrected = corrected.replace(incorrect_text, corrected_text)
                applied.append(
                    {
                        "incorrect_text": incorrect_text,
                        "corrected_text": corrected_text,
                    }
                )

        return corrected, applied

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, original_input, parsed_problem, solution, verification_result, user_feedback, timestamp "
                "FROM solved_problems ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "id": row[0],
                    "original_input": row[1],
                    "parsed_problem": row[2],
                    "solution": row[3],
                    "verification_result": row[4],
                    "user_feedback": row[5],
                    "timestamp": row[6],
                }
            )
        return out


    # -----------------------------
    # EXACT MATCH SEARCH
    # -----------------------------
    def get_exact_match(self, query: str):

        """Return previously solved problem if exact match exists."""

        with self._conn() as con:

            row = con.execute(
                """
                SELECT
                    original_input,
                    parsed_problem,
                    retrieved_context,
                    solution,
                    verification_result,
                    user_feedback
                FROM solved_problems
                WHERE original_input = ?
                LIMIT 1
                """,
                (query,),
            ).fetchone()

        if not row:
            return None

        try:
            parsed_problem = json.loads(row[1])
        except Exception:
            parsed_problem = row[1]

        try:
            retrieved_context = json.loads(row[2])
        except Exception:
            retrieved_context = row[2]

        try:
            verification_result = json.loads(row[4])
        except Exception:
            verification_result = row[4]

        return {
            "original_input": row[0],
            "parsed_problem": parsed_problem,
            "retrieved_context": retrieved_context,
            "solution": row[3],
            "verification_result": verification_result,
            "user_feedback": row[5],
        }
