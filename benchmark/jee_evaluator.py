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

    def run_agent_on_problem(self, problem: str, image_data: str = None) -> dict:
        """Run the agent on a single JEE problem.

        Args:
            problem: The problem text.

        Returns:
            Dictionary with answer, latex_answer, time_taken, reasoning_trace,
            steps, problem_type, confidence, retries, and success flag.
        """
        start = time.time()
        try:
            result = self.agent.solve(problem, image_data=image_data)
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
    def normalize_answer(ans: str) -> any:
        """Normalize an answer string into a comparable Python object.

        Handles symbolic expressions, integers, rationals, coordinates,
        sets, lists, matrices, and booleans.
        """
        if not ans:
            return ""
        s = str(ans).strip().lower().replace("$", "").replace("\\", "")

        # Try coordinate / tuple: (1, 2) or (3, 4, 5)
        tuple_match = re.match(r"^\s*\(([^)]+)\)\s*$", s)
        if tuple_match:
            parts = [p.strip() for p in tuple_match.group(1).split(",")]
            try:
                import sympy as sp
                return tuple(sp.sympify(p) for p in parts)
            except Exception:
                return tuple(parts)

        # Try set: {1, 2, 3}
        set_match = re.match(r"^\s*\{([^}]+)\}\s*$", s)
        if set_match:
            parts = [p.strip() for p in set_match.group(1).split(",")]
            try:
                import sympy as sp
                return frozenset(sp.sympify(p) for p in parts)
            except Exception:
                return frozenset(parts)

        # Try list / matrix: [1, 2, 3] or [[1,2],[3,4]]
        list_match = re.match(r"^\s*\[([\s\S]+)\]\s*$", s)
        if list_match:
            inner = list_match.group(1).strip()
            # Simple heuristic: if it contains '],[' it's a matrix
            if "],[" in inner or "], [" in inner:
                try:
                    import ast
                    return ast.literal_eval(s)
                except Exception:
                    pass
            parts = [p.strip() for p in inner.split(",")]
            try:
                import sympy as sp
                return [sp.sympify(p) for p in parts]
            except Exception:
                return parts

        # Try boolean
        if s in ("true", "yes", "t"):
            return True
        if s in ("false", "no", "f"):
            return False

        # Try integer / rational
        try:
            import sympy as sp
            return sp.Rational(s)
        except Exception:
            pass

        # Fallback: symbolic expression
        try:
            import sympy as sp
            return sp.sympify(s)
        except Exception:
            return s

    @staticmethod
    def check_symbolic_equivalence(predicted: str, gold: str) -> bool:
        """Check if two symbolic expressions are equivalent using SymPy.

        Handles multiple answer types: symbolic, integer, rational,
        coordinate tuples, sets, lists, matrices, and booleans.

        Args:
            predicted: The predicted answer string.
            gold: The gold (correct) answer string.

        Returns:
            True if the expressions are equivalent.
        """
        if not predicted or not gold:
            return False

        # Normalize strings
        pred_raw = str(predicted).strip().lower().replace("$", "")
        gold_raw = str(gold).strip().lower().replace("$", "")

        # Direct match
        if pred_raw == gold_raw:
            return True

        # Normalize to typed objects
        pred_obj = JEEEValuator.normalize_answer(pred_raw)
        gold_obj = JEEEValuator.normalize_answer(gold_raw)

        # Same-type comparison
        if type(pred_obj) is type(gold_obj):
            # Tuples / coordinates
            if isinstance(pred_obj, tuple):
                if len(pred_obj) != len(gold_obj):
                    return False
                import sympy as sp
                for p, g in zip(pred_obj, gold_obj):
                    try:
                        if sp.simplify(p - g) != 0:
                            return False
                    except Exception:
                        if str(p) != str(g):
                            return False
                return True

            # Sets
            if isinstance(pred_obj, frozenset):
                if len(pred_obj) != len(gold_obj):
                    return False
                import sympy as sp
                for p in pred_obj:
                    found = False
                    for g in gold_obj:
                        try:
                            if sp.simplify(p - g) == 0:
                                found = True
                                break
                        except Exception:
                            if str(p) == str(g):
                                found = True
                                break
                    if not found:
                        return False
                return True

            # Lists / matrices
            if isinstance(pred_obj, list):
                if len(pred_obj) != len(gold_obj):
                    return False
                import sympy as sp
                for p, g in zip(pred_obj, gold_obj):
                    try:
                        if sp.simplify(p - g) != 0:
                            return False
                    except Exception:
                        if str(p) != str(g):
                            return False
                return True

            # Booleans
            if isinstance(pred_obj, bool):
                return pred_obj == gold_obj

        # SymPy symbolic comparison (works for Expr, Rational, Integer)
        try:
            import sympy as sp

            x, y, z, t = sp.symbols("x y z t", real=True)
            a, b, c, n = sp.symbols("a b c n", real=True)
            pi = sp.pi
            E = sp.E

            pred_expr = sp.sympify(pred_raw)
            gold_expr = sp.sympify(gold_raw)

            # Direct equals
            if pred_expr.equals(gold_expr):
                return True

            # Simplified difference
            diff = sp.simplify(pred_expr - gold_expr)
            if diff == 0:
                return True

            # Numerical equivalence at random points
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

        # Numerical tolerance for float-like answers
        try:
            pred_f = float(pred_raw)
            gold_f = float(gold_raw)
            if abs(pred_f - gold_f) < 1e-4:
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
            s = s.replace("pi", "π")
            return s

        return normalize(pred_raw) == normalize(gold_raw)

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
