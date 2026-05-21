"""LangGraph definition for the math reasoning agent."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent.nodes import execute_node, format_node, reflect_node, think_node, verify_node
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState


class MathAgent:
    """LangGraph-based math reasoning agent with code execution and verification."""

    def __init__(self, config):
        """Initialize the math agent with configuration.

        Args:
            config: Application configuration object.
        """
        self.config = config
        self.llm = self._init_llm()
        self.graph = self._build_graph()

    def _init_llm(self):
        """Initialize the LLM client."""
        return ChatOpenAI(
            base_url=self.config.LLM_BASE_URL,
            api_key=self.config.LLM_API_KEY,
            model=self.config.LLM_MODEL,
            temperature=self.config.TEMPERATURE,
        )

    def _build_graph(self):
        """Build the LangGraph state machine.

        The flow is:
        think -> execute -> verify -> [reflect -> execute]* -> format -> END

        Returns:
            Compiled LangGraph.
        """
        builder = StateGraph(AgentState)

        # Add nodes -- use lambda to inject llm and config
        builder.add_node("think", lambda s: think_node(s, self.llm, self.config))
        builder.add_node("execute", lambda s: execute_node(s, self.config))
        builder.add_node("verify", lambda s: verify_node(s, self.llm, self.config))
        builder.add_node("reflect", lambda s: reflect_node(s, self.llm, self.config))
        builder.add_node("format", lambda s: format_node(s, self.llm, self.config))

        builder.set_entry_point("think")
        builder.add_edge("think", "execute")
        builder.add_edge("execute", "verify")

        def router(state):
            """Route from verify to either format or reflect."""
            if state.get("done", False):
                return "format"
            if state.get("retry_count", 0) >= self.config.MAX_RETRIES:
                return "format"
            return "reflect"

        builder.add_conditional_edges(
            "verify",
            router,
            {"format": "format", "reflect": "reflect"},
        )
        builder.add_edge("reflect", "execute")
        builder.add_edge("format", END)

        return builder.compile()

    def solve(self, problem: str) -> dict:
        """Solve a math problem using the agent graph.

        Args:
            problem: The math problem text.

        Returns:
            Final state dictionary with the answer and reasoning trace.
        """
        import time
        start = time.time()
        initial_state = {
            "problem": problem,
            "thought": "",
            "code": None,
            "code_result": None,
            "answer": None,
            "reasoning_trace": [],
            "verification": None,
            "reflection": None,
            "retry_count": 0,
            "final_answer": None,
            "done": False,
        }
        result = self.graph.invoke(initial_state)
        result["_time_taken"] = round(time.time() - start, 2)
        return result
