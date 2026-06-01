"""Fast JEE Agent — optimized pipeline with fewer LLM calls.

Pipeline (3-4 LLM calls):
    analyze_solve -> verify_fast -> format_fast -> END
                           |
                    [not done] -> reflect -> analyze_solve

Optimizations:
- analyze + plan + solve combined into ONE node
- Single SymPy code block solves entire problem
- Compact prompts reduce tokens
- Fast path for simple formatting (skips LLM)
- Reduced MAX_RETRIES from 3 to 2
"""

import time

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from agent.jee_state import JEEAgentState
from agent.image_ocr import MathImageOCR
from agent.jee_nodes import (
    analyze_solve_node,
    verify_fast_node,
    reflect_fast_node,
    format_fast_node,
)
from agent.rag_knowledge_base import get_knowledge_base


class JEEAgent:

    def __init__(self, config):
        self.config = config
        self.llm = self._init_llm()
        self._init_rag()
        self.graph = self._build_graph()

    def _init_rag(self):
        """Initialize RAG knowledge base."""
        try:
            from benchmark.jee_loader import JEELoader
            problems = JEELoader.load_problems()
            kb = get_knowledge_base(problems=problems)
            stats = kb.get_stats()
            if stats.get("available"):
                print(f"[JEEFastAgent] RAG ready: {stats['count']} docs")
        except Exception as exc:
            print(f"[JEEFastAgent] RAG init skipped: {exc}")

    def _init_llm(self):
        """Initialize LLM client."""
        kwargs = {
            "base_url": self.config.LLM_BASE_URL,
            "api_key": self.config.LLM_API_KEY,
            "model": self.config.LLM_MODEL,
            "temperature": getattr(self.config, "TEMPERATURE", 0.1),
        }
        if getattr(self.config, "JSON_MODE", True):
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    def _build_graph(self):
        """Build fast linear pipeline."""
        builder = StateGraph(JEEAgentState)

        builder.add_node("analyze_solve", lambda s: analyze_solve_node(s, self.llm, self.config))
        builder.add_node("verify_fast", lambda s: verify_fast_node(s, self.llm, self.config))
        builder.add_node("reflect", lambda s: reflect_fast_node(s, self.llm, self.config))
        builder.add_node("format_fast", lambda s: format_fast_node(s, self.llm, self.config))

        builder.set_entry_point("analyze_solve")
        builder.add_edge("analyze_solve", "verify_fast")

        def _verify_router(state: JEEAgentState):
            if state.get("done", False):
                return "format_fast"
            if state.get("retry_count", 0) >= getattr(self.config, "MAX_RETRIES", 2):
                return "format_fast"
            return "reflect"

        builder.add_conditional_edges(
            "verify_fast",
            _verify_router,
            {"format_fast": "format_fast", "reflect": "reflect"},
        )

        builder.add_edge("reflect", "analyze_solve")
        builder.add_edge("format_fast", END)

        return builder.compile()

    def solve(self, problem: str, image_data: str = None) -> dict:
        """Solve a JEE problem (fast path)."""
        start = time.time()

        extracted_latex = ""
        if image_data:
            try:
                ocr = MathImageOCR()
                extracted_latex = ocr.extract(image_data)
                if extracted_latex:
                    problem = f"{problem}\n\n[Image]: {extracted_latex}"
            except Exception:
                pass

        initial_state: JEEAgentState = {
            "problem": problem,
            "problem_type": None,
            "key_concepts": [],
            "solution_plan": None,
            "thought": "",
            "code": None,
            "sympy_code": None,
            "sympy_result": None,
            "code_result": None,
            "answer": None,
            "steps": [],
            "current_step_index": 0,
            "_all_steps_done": False,
            "symbolic_steps": [],
            "reasoning_trace": [],
            "verification": None,
            "reflection": None,
            "retry_count": 0,
            "image_data": image_data,
            "extracted_latex": extracted_latex if extracted_latex else None,
            "final_answer": None,
            "latex_answer": None,
            "confidence": 0.0,
            "done": False,
        }

        result = self.graph.invoke(initial_state)
        result["_time_taken"] = round(time.time() - start, 2)
        return result
