"""Solver agent that combines RAG context + symbolic tools."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List

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
            
        # Initialize pipeline placeholder (lazy loaded)
        self.llm_pipeline = None

    def _get_local_llm(self):
        """Lazy loads the LLM to save memory until a word problem requires it."""
        if self.llm_pipeline is not None:
            return self.llm_pipeline

        try:
            from transformers import pipeline
            import torch
            
            # Force downloads strictly to the project folder
            local_dir = os.path.join(os.getcwd(), "local_models")
            os.makedirs(local_dir, exist_ok=True)
            
            # Qwen 2.5 0.5B is tiny (~1GB), highly capable at math, and fast on CPU
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            
            print(f"Loading local LLM ({model_id}) into {local_dir}...")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"cache_dir": local_dir},
                device="cpu", # Safe default. Change to "cuda:0" or "cuda" if you have an Nvidia GPU.
            )
            return self.llm_pipeline
        except ImportError:
            print("WARNING: 'transformers' or 'torch' not installed. LLM fallback disabled.")
            return None

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

        # remove spaces
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

        # case 2: parse around '=' and keep only math-like segments
        if "=" in question:

            left_raw, right_raw = question.rsplit("=", 1)

            left = self._extract_math_side(left_raw, from_end=True)
            right = self._extract_math_side(right_raw, from_end=False)

            left = self._trim_to_math_tail(left)
            right = self._trim_to_math_tail(right)

            equation = f"{left}={right}".strip("=")

            if re.search(r"[A-Za-z]", equation) and re.search(
                r"\d|\*|\+|\-|\^|/|\(|\)", equation
            ):
                return equation

        # case 3: equation-like inline segment
        eq_match = re.search(
            r"([A-Za-z0-9\*\+\-\^\(\)\./\s]*[A-Za-z]\d*[A-Za-z0-9\*\+\-\^\(\)\./\s]*=[A-Za-z0-9\*\+\-\^\(\)\./\s]+)",
            question,
        )

        if eq_match:
            return eq_match.group(1).strip()

        return None

    def _extract_math_side(self, text: str, from_end: bool) -> str:
        """Extract likely math segment from one side of an equation."""

        cleaned = text.strip()

        if not cleaned:
            return cleaned

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

            if from_end:

                tail_match = re.search(
                    r"([A-Za-z0-9\(\)\[\]\{\}\*\+\-\^\./\s]+)$", cand
                )

                if tail_match:
                    tail = tail_match.group(1).strip()

                    if candidate_score(tail) > candidate_score(best):
                        best = tail

        return best or cleaned

    def _trim_to_math_tail(self, text: str) -> str:
        """Trim leading prompt words and keep the equation-like tail."""

        text = text.strip()

        if not text:
            return text

        pattern = r"([A-Za-z]\d+|\d+[A-Za-z]|[A-Za-z0-9\)\]]\s*[\+\-\*\/\^]+\s*[A-Za-z0-9\(\[]|[A-Za-z]\s*\()"

        matches = list(re.finditer(pattern, text))

        if matches:
            return text[matches[0].start() :].strip()

        return text

    # -------------------------
    # Main solver
    # -------------------------
    def run(self, parsed_problem: Dict, strategy: str) -> SolverResult:

        question = parsed_problem["problem_text"]

        retrieved = []

        if self.retriever is not None:
            try:
                retrieved = self.retriever.retrieve(question, top_k=4)
            except Exception:
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
            "Apply symbolic manipulation or local LLM reasoning to compute answer",
        ]

        steps: List[str] = []
        answer = "Could not derive a final answer automatically."

        # --------------------------------
        # 1. Try Symbolic Extraction First
        # --------------------------------

        expr = self.extract_equation(question)

        if expr:

            normalized = self.normalize_expression(expr)

            var_match = re.search(r"[a-zA-Z]", normalized)

            solve_var = var_match.group(0) if var_match else "x"

            steps.append(f"Parsed equation: {normalized}")

            res = solve_expression(normalized, variable=solve_var)

            if not res.success:

                normalized_retry = re.sub(
                    r"[^A-Za-z0-9\*\+\-\^=/\(\)\.\s]", "", normalized
                )

                normalized_retry = self._trim_to_math_tail(normalized_retry)

                normalized_retry = self.normalize_expression(normalized_retry)

                var_match_retry = re.search(r"[a-zA-Z]", normalized_retry)

                solve_var_retry = (
                    var_match_retry.group(0) if var_match_retry else solve_var
                )

                if normalized_retry and normalized_retry != normalized:

                    steps.append(f"Retry parsed equation: {normalized_retry}")

                    res = solve_expression(normalized_retry, variable=solve_var_retry)

            if res.success:

                steps.append(f"Solved roots using SymPy: {res.output}")

                answer = str(res.output)

        else:

            # --------------------------------
            # 2. Try simplify/evaluate problems
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
        # 3. Fallback: Local LLM for Word Problems
        # --------------------------------

        if not steps:
            llm = self._get_local_llm()
            
            if llm:
                steps.append("Equation extraction failed. Initializing local LLM for word problem reasoning...")
                context_text = "\n".join([f"- {c['content']}" for c in retrieved_ctx])
                
                # Format specific to Qwen Instruct models
                prompt = f"""<|im_start|>system
You are a concise math solver. Extract the final answer and provide brief steps separated by '|'.
<|im_end|>
<|im_start|>user
Problem: {question}

Context:
{context_text}

Respond exactly in this format:
STEPS: <step 1> | <step 2> | <step 3>
FINAL_ANSWER: <just the final number or fraction>
<|im_end|>
<|im_start|>assistant
"""
                try:
                    # Generate response
                    output = llm(prompt, max_new_tokens=200, return_full_text=False, temperature=0.1)
                    llm_response = output[0]['generated_text'].strip()
                    
                    if "FINAL_ANSWER:" in llm_response:
                        parts = llm_response.split("FINAL_ANSWER:")
                        steps_raw = parts[0].replace("STEPS:", "").strip()
                        answer_raw = parts[1].strip()
                        
                        parsed_steps = [s.strip() for s in steps_raw.split("|") if s.strip()]
                        steps.extend(parsed_steps)
                        answer = answer_raw
                    else:
                        steps.append(f"LLM Reasoning generated: {llm_response}")
                        answer = "LLM did not format answer properly. Please review."
                        
                except Exception as e:
                    steps.append(f"Local LLM failed: {str(e)}")
                    answer = "Please review the explanation and verify with HITL for final confidence."
            else:
                steps.append(
                    "Used retrieved math context to build a conceptual solution path. (Local LLM unavailable)"
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
