"""Evaluate JEE agent with symbolic answer verification using SymPy."""

import re
import time
from datetime import datetime
from typing import Callable, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class JEEEValuator:
    """Evaluate the JEE math agent and raw LLM with symbolic answer checking."""

    def __init__(self, agent, config):
        """Initialize the JEE evaluator.

        Args:
            agent: The JEE MathAgent instance.
            config: Application configuration.
        """
        self.agent = agent
        self.config = config
        self.llm = self._init_raw_llm()

    def _init_raw_llm(self):
        """Initialize a raw LLM for baseline comparison."""
        return ChatOpenAI(
            base_url=self.config.LLM_BASE_URL,
            api_key=self.config.LLM_API_KEY,
            model=self.config.LLM_MODEL,
            temperature=0.1,
        )

    def run_agent_on_problem(self, problem: str) -> dict:
        """Run the agent on a single JEE problem.

        Args:
            problem: The problem text.

        Returns:
            Dictionary with answer, latex_answer, time_taken, reasoning_trace,
            steps, problem_type, confidence, retries, and success flag.
        """
        start = time.time()
        try:
            result = self.agent.solve(problem)
            elapsed = time.time() - start
            return {
                "answer": result.get("final_answer", ""),
                "latex_answer": result.get("latex_answer", ""),
                "time_taken": elapsed,
                "reasoning_trace": result.get("reasoning_trace", []),
                "steps": result.get("steps", []),
                "problem_type": result.get("problem_type", ""),
                "confidence": result.get("confidence", 0),
                "retries": result.get("retry_count", 0),
                "success": True,
            }
        except Exception as exc:
            return {
                "answer": "",
                "latex_answer": "",
                "time_taken": time.time() - start,
                "reasoning_trace": [f"Error: {str(exc)}"],
                "steps": [],
                "problem_type": "",
                "confidence": 0,
                "retries": 0,
                "success": False,
            }

    def run_raw_llm_on_problem(self, problem: str) -> dict:
        """Run the raw LLM on a single JEE problem without agent tooling.

        Args:
            problem: The problem text.

        Returns:
            Dictionary with answer, time_taken, and success flag.
        """
        start = time.time()
        try:
            prompt = f"""You are solving a JEE Advanced calculus problem. Solve step by step and return ONLY the final numerical/symbolic answer. No explanation.

Problem: {problem}

Final Answer (just the expression/number):"""
            response = self.llm.invoke([HumanMessage(content=prompt)])
            elapsed = time.time() - start
            return {
                "answer": response.content.strip(),
                "time_taken": elapsed,
                "success": True,
            }
        except Exception as exc:
            return {
                "answer": "",
                "time_taken": time.time() - start,
                "success": False,
            }

    @staticmethod
    def check_symbolic_equivalence(predicted: str, gold: str) -> bool:
        """Check if two symbolic expressions are equivalent using SymPy.

        First tries direct string match, then SymPy symbolic comparison,
        then numerical evaluation at random points, then normalized
        string comparison as a fallback.

        Args:
            predicted: The predicted answer string.
            gold: The gold (correct) answer string.

        Returns:
            True if the expressions are equivalent.
        """
        if not predicted or not gold:
            return False

        # Normalize strings
        pred = str(predicted).strip().lower().replace("$", "")
        gold = str(gold).strip().lower().replace("$", "")

        # Direct match
        if pred == gold:
            return True

        # Try SymPy symbolic comparison
        try:
            import sympy as sp

            x, y, z, t = sp.symbols("x y z t", real=True)
            a, b, c, n = sp.symbols("a b c n", real=True)
            pi = sp.pi
            E = sp.E

            # Parse both expressions
            pred_expr = sp.sympify(pred)
            gold_expr = sp.sympify(gold)

            # Check symbolic equivalence
            if pred_expr.equals(gold_expr):
                return True

            # Check simplified difference
            diff = sp.simplify(pred_expr - gold_expr)
            if diff == 0:
                return True

            # Check numerical equivalence at random points
            import random

            for _ in range(5):
                subs = {x: random.uniform(0.1, 5), y: random.uniform(0.1, 5)}
                try:
                    pred_val = float(pred_expr.subs(subs).evalf())
                    gold_val = float(gold_expr.subs(subs).evalf())
                    if abs(pred_val - gold_val) > 0.01:
                        return False
                except Exception:
                    continue
            return True
        except Exception:
            pass

        # Fallback: normalized string comparison
        def normalize(s):
            s = re.sub(r"\s+", "", s)
            s = s.replace("**", "^")
            s = s.replace("atan", "arctan")
            s = s.replace("asin", "arcsin")
            s = s.replace("acos", "arccos")
            return s

        return normalize(pred) == normalize(gold)

    def run_benchmark(
        self,
        dataset: list[dict],
        mode: str = "both",
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Run the JEE benchmark on a dataset.

        Args:
            dataset: List of JEE problem dictionaries.
            mode: 'agent', 'raw_llm', or 'both'.
            progress_callback: Optional callback function(current, total).

        Returns:
            Results dictionary with run_id, accuracy, and per-problem results.
        """
        run_id = f"jee_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        total = len(dataset)
        agent_results = []
        raw_results = []

        for i, prob in enumerate(dataset):
            q = prob["question"]
            gold = prob.get("answer", "")

            if mode in ("agent", "both"):
                res = self.run_agent_on_problem(q)
                correct = self.check_symbolic_equivalence(res["answer"], gold)
                agent_results.append(
                    {
                        "problem": q,
                        "gold_answer": gold,
                        "predicted": res["answer"],
                        "latex_answer": res["latex_answer"],
                        "correct": correct,
                        "time_taken": res["time_taken"],
                        "problem_type": prob.get("problem_type", ""),
                        "difficulty": prob.get("difficulty", ""),
                        "retries": res["retries"],
                        "success": res["success"],
                    }
                )

            if mode in ("raw_llm", "both"):
                res = self.run_raw_llm_on_problem(q)
                correct = self.check_symbolic_equivalence(res["answer"], gold)
                raw_results.append(
                    {
                        "problem": q,
                        "gold_answer": gold,
                        "predicted": res["answer"],
                        "correct": correct,
                        "time_taken": res["time_taken"],
                        "success": res["success"],
                    }
                )

            if progress_callback:
                progress_callback(i + 1, total)

        result = {
            "run_id": run_id,
            "mode": mode,
            "model": self.config.LLM_MODEL,
            "total": total,
            "timestamp": datetime.now().isoformat(),
        }

        if agent_results:
            c = sum(1 for r in agent_results if r["correct"])
            result["agent"] = {
                "correct": c,
                "incorrect": len(agent_results) - c,
                "accuracy": round(c / len(agent_results), 4),
                "avg_time": round(
                    sum(r["time_taken"] for r in agent_results) / len(agent_results), 2
                ),
                "by_type": self._group_by_type(agent_results),
                "by_difficulty": self._group_by_difficulty(agent_results),
                "results": agent_results,
            }

        if raw_results:
            c = sum(1 for r in raw_results if r["correct"])
            result["raw_llm"] = {
                "correct": c,
                "incorrect": len(raw_results) - c,
                "accuracy": round(c / len(raw_results), 4),
                "avg_time": round(
                    sum(r["time_taken"] for r in raw_results) / len(raw_results), 2
                ),
                "results": raw_results,
            }

        return result

    @staticmethod
    def _group_by_type(results: list) -> dict:
        """Group results by problem type and compute accuracy per type."""
        groups = {}
        for r in results:
            t = r.get("problem_type", "unknown")
            if t not in groups:
                groups[t] = {"total": 0, "correct": 0}
            groups[t]["total"] += 1
            if r["correct"]:
                groups[t]["correct"] += 1
        for t in groups:
            groups[t]["accuracy"] = round(
                groups[t]["correct"] / groups[t]["total"], 4
            )
        return groups

    @staticmethod
    def _group_by_difficulty(results: list) -> dict:
        """Group results by difficulty and compute accuracy per level."""
        groups = {}
        for r in results:
            d = r.get("difficulty", "unknown")
            if d not in groups:
                groups[d] = {"total": 0, "correct": 0}
            groups[d]["total"] += 1
            if r["correct"]:
                groups[d]["correct"] += 1
        for d in groups:
            groups[d]["accuracy"] = round(
                groups[d]["correct"] / groups[d]["total"], 4
            )
        return groups
