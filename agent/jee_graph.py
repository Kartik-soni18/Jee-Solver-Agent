"""Enhanced LangGraph for JEE Advanced calculus problem solving.

The graph follows a *decomposed reasoning* topology::

    START -> analyze -> plan -> solve_symbolic -[steps remain]-> solve_symbolic (self-loop)
                                            |
                                     [all steps done]
                                            v
                                    consolidate -> verify -> format -> END
                                                        |
                                                 [not done & retries left]
                                                        v
                                                    reflect -> solve_symbolic

The ``solve_symbolic`` node uses a self-loop to execute planned steps one
at a time.  After all steps complete, the graph proceeds through
consolidation, verification, and formatting.  If verification fails,
``reflect`` produces corrected code and the flow loops back to
``solve_symbolic`` for a retry (up to MAX_RETRIES).
"""

import time

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from agent.jee_state import JEEAgentState
from agent.jee_nodes import (
    analyze_node,
    plan_node,
    solve_symbolic_node,
    consolidate_node,
    verify_symbolic_node,
    reflect_jee_node,
    format_latex_node,
)


class JEEAgent:
    """JEE Advanced calculus problem solver using multi-step symbolic reasoning.

    Usage::

        from config import Config
        from agent.jee_graph import JEEAgent

        cfg = Config(LLM_API_KEY="...")
        agent = JEEAgent(cfg)
        result = agent.solve("Evaluate: lim x->0 sin(3x)/x")
        print(result["final_answer"])   # plain-text answer
        print(result["latex_answer"])   # LaTeX-formatted answer
    """

    def __init__(self, config):
        self.config = config
        self.llm = self._init_llm()
        self.graph = self._build_graph()

    def _init_llm(self):
        """Initialise the LLM client from configuration."""
        return ChatOpenAI(
            base_url=self.config.LLM_BASE_URL,
            api_key=self.config.LLM_API_KEY,
            model=self.config.LLM_MODEL,
            temperature=getattr(self.config, "TEMPERATURE", 0.1),
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Build and compile the JEE agent LangGraph."""
        builder = StateGraph(JEEAgentState)

        # -- Register nodes -------------------------------------------------
        builder.add_node("analyze", lambda s: analyze_node(s, self.llm, self.config))
        builder.add_node("plan", lambda s: plan_node(s, self.llm, self.config))
        builder.add_node("solve_symbolic", lambda s: solve_symbolic_node(s, self.llm, self.config))
        builder.add_node("consolidate", lambda s: consolidate_node(s, self.llm, self.config))
        builder.add_node("verify", lambda s: verify_symbolic_node(s, self.llm, self.config))
        builder.add_node("reflect", lambda s: reflect_jee_node(s, self.llm, self.config))
        builder.add_node("format", lambda s: format_latex_node(s, self.llm, self.config))

        # -- Entry point ----------------------------------------------------
        builder.set_entry_point("analyze")

        # -- Linear edges (analyze -> plan -> solve_symbolic) ---------------
        builder.add_edge("analyze", "plan")
        builder.add_edge("plan", "solve_symbolic")

        # -- solve_symbolic self-loop (while steps remain) ------------------
        def _solve_router(state: JEEAgentState):
            """Route from solve_symbolic: loop if steps remain, else consolidate."""
            if state.get("_all_steps_done", False):
                return "consolidate"
            return "solve_symbolic"

        builder.add_conditional_edges(
            "solve_symbolic",
            _solve_router,
            {"solve_symbolic": "solve_symbolic", "consolidate": "consolidate"},
        )

        # -- Linear edges (consolidate -> verify -> format) -----------------
        builder.add_edge("consolidate", "verify")

        # -- verify conditional (done -> format, else -> reflect) -----------
        def _verify_router(state: JEEAgentState):
            """Route from verify: format if done or max retries, else reflect."""
            if state.get("done", False):
                return "format"
            if state.get("retry_count", 0) >= getattr(self.config, "MAX_RETRIES", 3):
                return "format"
            return "reflect"

        builder.add_conditional_edges(
            "verify",
            _verify_router,
            {"format": "format", "reflect": "reflect"},
        )

        # -- reflect loops back to solve_symbolic for retry -----------------
        builder.add_edge("reflect", "solve_symbolic")

        # -- format -> END --------------------------------------------------
        builder.add_edge("format", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, problem: str) -> dict:
        """Solve a JEE calculus problem.

        Args:
            problem: The JEE problem text (supports LaTeX notation).

        Returns:
            Complete state dictionary including ``final_answer``,
            ``latex_answer``, ``reasoning_trace``, ``steps``,
            ``problem_type``, ``confidence``, and ``_time_taken``.
        """
        start = time.time()

        initial_state: JEEAgentState = {
            # -- Core problem fields ----------------------------------------
            "problem": problem,
            "problem_type": None,
            "key_concepts": [],
            "solution_plan": None,

            # -- Working state ----------------------------------------------
            "thought": "",
            "code": None,
            "sympy_code": None,
            "sympy_result": None,
            "code_result": None,
            "answer": None,

            # -- Step tracking ----------------------------------------------
            "steps": [],
            "current_step_index": 0,
            "_all_steps_done": False,
            "symbolic_steps": [],
            "reasoning_trace": [],

            # -- Verification / reflection ----------------------------------
            "verification": None,
            "reflection": None,
            "retry_count": 0,

            # -- Final output -----------------------------------------------
            "final_answer": None,
            "latex_answer": None,
            "confidence": 0.0,
            "done": False,
        }

        result = self.graph.invoke(initial_state)
        result["_time_taken"] = round(time.time() - start, 2)
        return result
