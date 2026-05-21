"""Streamlit interface for the JEE Advanced Math Agent.

A professional dashboard with LaTeX rendering for JEE calculus problems,
including benchmark evaluation, per-type/per-difficulty breakdowns,
and interactive visualizations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JEE Advanced Math Agent",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color palette for problem types
# ---------------------------------------------------------------------------
PROBLEM_TYPE_COLORS = {
    "limits": "#3498db",
    "differentiation": "#e74c3c",
    "integration": "#2ecc71",
    "definite_integrals": "#9b59b6",
    "differential_equations": "#f39c12",
    "maxima_minima": "#1abc9c",
    "tangent_normal": "#e67e22",
    "area": "#34495e",
}

DIFFICULTY_COLORS = {
    "easy": "#2ecc71",
    "medium": "#f39c12",
    "hard": "#e74c3c",
}

DIFFICULTY_EMOJI = {
    "easy": "🟢",
    "medium": "🟡",
    "hard": "🔴",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar_config():
    """Render the configuration sidebar and return a validated Config."""
    st.sidebar.title("🔷 JEE Advanced Configuration")
    st.sidebar.markdown("---")

    # ── API Configuration ────────────────────────────────────────────────
    st.sidebar.subheader("🤖 Model Settings")

    api_key = st.sidebar.text_input(
        "LLM API Key",
        value="",
        type="password",
        help="Your OpenAI-compatible API key",
    )
    base_url = st.sidebar.text_input(
        "LLM Base URL",
        value="https://api.openai.com/v1",
        help="API base URL (e.g., OpenAI, Groq, Anyscale)",
    )
    model = st.sidebar.text_input(
        "LLM Model",
        value="gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, gpt-4o, llama-3.1-70b)",
    )
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower = more deterministic, Higher = more creative",
    )
    max_retries = st.sidebar.slider(
        "Max Retries",
        min_value=0,
        max_value=5,
        value=3,
        step=1,
        help="Maximum number of reflection retries",
    )

    st.sidebar.markdown("---")

    # ── JEE Dataset Filters ──────────────────────────────────────────────
    st.sidebar.subheader("📚 JEE Dataset Options")

    try:
        from benchmark.jee_loader import JEELoader

        all_types = JEELoader.get_problem_types()
        all_difficulties = ["easy", "medium", "hard"]

        selected_type = st.sidebar.selectbox(
            "Filter by Problem Type",
            options=["All"] + all_types,
            help="Select a specific calculus topic",
        )
        selected_difficulty = st.sidebar.selectbox(
            "Filter by Difficulty",
            options=["All"] + all_difficulties,
            help="Filter problems by difficulty level",
        )
    except Exception:
        selected_type = "All"
        selected_difficulty = "All"
        st.sidebar.warning("Could not load JEE dataset filters.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align:center; color:#7f8c8d; font-size:0.8em;'>"
        "Made with LangGraph + Streamlit + SymPy</div>",
        unsafe_allow_html=True,
    )

    from config import Config

    cfg = Config(
        LLM_API_KEY=api_key,
        LLM_BASE_URL=base_url,
        LLM_MODEL=model,
        TEMPERATURE=temperature,
        MAX_RETRIES=max_retries,
    )

    return cfg, selected_type, selected_difficulty


# ---------------------------------------------------------------------------
# Dataset Overview (inline helper)
# ---------------------------------------------------------------------------
def render_dataset_overview():
    """Render a small overview of the JEE dataset in the sidebar."""
    try:
        from benchmark.jee_loader import JEELoader

        summary = JEELoader.get_dataset_summary()
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Dataset Overview")
        st.sidebar.markdown(f"**Total Problems:** {summary['total_problems']}")

        for diff, count in summary["difficulty_distribution"].items():
            color = DIFFICULTY_COLORS.get(diff, "#7f8c8d")
            emoji = DIFFICULTY_EMOJI.get(diff, "⚪")
            st.sidebar.markdown(
                f"<span style='color:{color};'>{emoji} {diff.capitalize()}: {count}</span>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tab 1: Single Problem Solver
# ---------------------------------------------------------------------------
def render_single_problem_tab(cfg):
    """Render the single JEE problem solver tab with LaTeX support."""
    st.header("🔷 Single Problem Solver")
    st.markdown(
        "Test the JEE agent against a single calculus problem. "
        "Enter any JEE-level problem below — LaTeX expressions are fully supported."
    )

    # Sample problems dropdown
    try:
        from benchmark.jee_loader import JEELoader

        all_problems = JEELoader.load_problems()
        sample_options = {
            "Custom (enter your own)": "",
            **{
                f"[{p['problem_type'].upper()} | {p['difficulty'].upper()}] {p['question'][:80]}...": p[
                    "question"
                ]
                for p in all_problems[:10]
            },
        }
        selected_sample = st.selectbox(
            "Choose a sample problem (optional):",
            options=list(sample_options.keys()),
            index=0,
        )
        default_text = sample_options[selected_sample]
    except Exception:
        default_text = r"Evaluate: $\lim_{x \to 0} \frac{\sin(3x) - 3\sin(x)}{x^3}$"

    problem = st.text_area(
        "Enter a JEE calculus problem:",
        value=default_text,
        height=120,
        help="Supports LaTeX math expressions enclosed in $...$",
    )

    # Render the problem as LaTeX preview
    if problem.strip():
        st.markdown("**Problem Preview:**")
        st.latex(problem.replace("$", "").replace("\\", "\\"))

    col1, col2 = st.columns(2)

    with col1:
        run_agent = st.button(
            "🤖 Run JEE Agent", use_container_width=True, type="primary"
        )
    with col2:
        run_raw = st.button("📝 Run Raw LLM", use_container_width=True)

    # ── Run Agent ──────────────────────────────────────────────────────────
    if run_agent:
        if not cfg.LLM_API_KEY:
            st.error("Please enter your LLM API Key in the sidebar first!")
            return

        try:
            cfg.validate()
            from agent.graph import MathAgent

            agent = MathAgent(cfg)

            with st.spinner("Agent is analyzing, planning, and solving symbolically..."):
                result = agent.solve(problem)

            st.subheader("Agent Result")

            final_answer = result.get("final_answer", "N/A")
            latex_answer = result.get("latex_answer", "")
            retries = result.get("retry_count", 0)
            reasoning_trace = result.get("reasoning_trace", [])
            steps = result.get("steps", [])
            problem_type = result.get("problem_type", "")
            confidence = result.get("confidence", 0)
            time_taken = result.get("_time_taken", "N/A")

            # ── Metrics row ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Final Answer", str(final_answer))
            with m2:
                st.metric("Confidence", f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A")
            with m3:
                st.metric("Time (s)", time_taken)
            with m4:
                st.metric("Retries Used", int(retries))

            # ── Problem type badge ───────────────────────────────────────
            if problem_type:
                badge_color = PROBLEM_TYPE_COLORS.get(problem_type, "#7f8c8d")
                st.markdown(
                    f"<span style='background-color:{badge_color}; color:white; "
                    f"padding:4px 12px; border-radius:12px; font-size:0.85em;'>"
                    f"📐 {problem_type.replace('_', ' ').title()}</span>",
                    unsafe_allow_html=True,
                )

            # ── LaTeX-formatted answer ───────────────────────────────────
            if latex_answer:
                st.markdown("**LaTeX Answer:**")
                st.latex(latex_answer.replace("$", "").replace("\\", "\\"))
            elif final_answer and final_answer != "N/A":
                st.markdown("**Answer:**")
                try:
                    st.latex(str(final_answer).replace("$", "").replace("\\", "\\"))
                except Exception:
                    st.code(str(final_answer))

            # ── Step-by-step derivation (expandable) ─────────────────────
            if steps:
                with st.expander("📋 Step-by-Step Derivation", expanded=False):
                    for i, step in enumerate(steps, 1):
                        if isinstance(step, dict):
                            st.markdown(f"**Step {i}:** {step.get('description', '')}")
                            if "latex" in step and step["latex"]:
                                st.latex(step["latex"].replace("$", "").replace("\\", "\\"))
                            if "result" in step and step["result"]:
                                st.code(f"Result: {step['result']}")
                        else:
                            st.markdown(f"**Step {i}:**")
                            st.code(str(step))
                        st.markdown("---")

            # ── Reasoning trace ──────────────────────────────────────────
            if reasoning_trace:
                with st.expander("🧠 Full Reasoning Trace", expanded=False):
                    for i, entry in enumerate(reasoning_trace, 1):
                        st.markdown(f"**Step {i}:**")
                        st.code(entry, language="text")

            # ── SymPy code (if available) ────────────────────────────────
            if result.get("sympy_code"):
                with st.expander("🔧 SymPy Code Executed", expanded=False):
                    st.code(result["sympy_code"], language="python")
                    if result.get("sympy_result"):
                        st.markdown("**SymPy Output:**")
                        st.code(result["sympy_result"], language="text")

        except Exception as exc:
            st.error(f"Agent error: {exc}")

    # ── Run Raw LLM ────────────────────────────────────────────────────────
    if run_raw:
        if not cfg.LLM_API_KEY:
            st.error("Please enter your LLM API Key in the sidebar first!")
            return

        try:
            cfg.validate()
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                base_url=cfg.LLM_BASE_URL,
                api_key=cfg.LLM_API_KEY,
                model=cfg.LLM_MODEL,
                temperature=cfg.TEMPERATURE,
            )

            with st.spinner("Raw LLM is solving..."):
                import time

                start = time.time()
                prompt = f"""You are solving a JEE Advanced calculus problem. Solve step by step and return ONLY the final numerical/symbolic answer. No explanation.

Problem: {problem}

Final Answer (just the expression/number):"""
                response = llm.invoke([HumanMessage(content=prompt)])
                elapsed = time.time() - start

            raw_answer = response.content.strip()

            st.subheader("Raw LLM Result")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Answer", raw_answer)
            with m2:
                st.metric("Time (s)", f"{elapsed:.2f}")

            # Try to render as LaTeX
            try:
                st.markdown("**Answer Preview:**")
                st.latex(raw_answer.replace("$", "").replace("\\", "\\"))
            except Exception:
                st.info(f"Raw LLM response: {raw_answer}")

        except Exception as exc:
            st.error(f"Raw LLM error: {exc}")


# ---------------------------------------------------------------------------
# Tab 2: JEE Benchmark Runner
# ---------------------------------------------------------------------------
def render_benchmark_tab(cfg, selected_type, selected_difficulty):
    """Render the JEE benchmark testing tab with filtering."""
    st.header("🔷 JEE Benchmark Runner")
    st.markdown(
        "Run the JEE agent and/or raw LLM on the embedded JEE Advanced problem dataset. "
        "Results include symbolic equivalence checking via SymPy."
    )

    # ── Dataset info ─────────────────────────────────────────────────────
    try:
        from benchmark.jee_loader import JEELoader

        summary = JEELoader.get_dataset_summary()

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Total Problems", summary["total_problems"])
        with info_col2:
            st.metric("Problem Types", len(summary["problem_types"]))
        with info_col3:
            st.metric("Topics", len(summary["topics"]))

        # Show filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            st.markdown(
                f"**Problem Type Filter:** `{selected_type}`"
            )
        with filter_col2:
            st.markdown(
                f"**Difficulty Filter:** `{selected_difficulty}`"
            )

        # Count filtered problems
        ptype = None if selected_type == "All" else selected_type
        pdiff = None if selected_difficulty == "All" else selected_difficulty
        filtered = JEELoader.load_problems(problem_type=ptype, difficulty=pdiff)
        st.info(f"📋 **{len(filtered)}** problems match the selected filters.")

    except Exception as exc:
        st.error(f"Could not load dataset info: {exc}")
        filtered = []

    # ── Controls ─────────────────────────────────────────────────────────
    st.markdown("---")
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        problem_count = st.number_input(
            "Number of Problems",
            min_value=1,
            max_value=len(filtered) if filtered else 33,
            value=min(10, len(filtered)) if filtered else 10,
            step=1,
        )
    with control_col2:
        mode = st.selectbox(
            "Mode",
            options=["Agent Only", "Raw LLM Only", "Both (Compare)"],
            index=2,
        )
    with control_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_symbolic_details = st.checkbox(
            "Show Symbolic Details", value=True, help="Show SymPy equivalence checking details"
        )

    mode_map = {
        "Agent Only": "agent",
        "Raw LLM Only": "raw_llm",
        "Both (Compare)": "both",
    }

    if st.button("▶️ Start JEE Benchmark", type="primary", use_container_width=True):
        if not cfg.LLM_API_KEY:
            st.error("Please enter your LLM API Key in the sidebar first!")
            return

        if not filtered:
            st.error("No problems match the selected filters.")
            return

        try:
            cfg.validate()

            # Build filtered dataset
            dataset = filtered[: int(problem_count)]

            st.success(f"Loaded {len(dataset)} JEE problems")

            # Initialize agent and evaluator
            from agent.graph import MathAgent
            from benchmark.jee_evaluator import JEEEValuator

            agent = MathAgent(cfg)
            evaluator = JEEEValuator(agent, cfg)

            # Progress tracking
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_callback(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"Processing JEE problem {current}/{total}...")

            # Run benchmark
            with st.spinner("Running JEE benchmark... This may take several minutes."):
                results = evaluator.run_benchmark(
                    dataset,
                    mode=mode_map[mode],
                    progress_callback=progress_callback,
                )

            progress_bar.empty()
            status_text.empty()

            st.success("✅ JEE Benchmark complete!")

            # ── Results Summary ──────────────────────────────────────────
            st.subheader("📊 Results Summary")

            if "agent" in results:
                agent_data = results["agent"]
                a_col1, a_col2, a_col3, a_col4 = st.columns(4)
                with a_col1:
                    st.metric("Agent Correct", f"{agent_data['correct']}/{results['total']}")
                with a_col2:
                    st.metric("Agent Accuracy", f"{agent_data['accuracy']:.1%}")
                with a_col3:
                    st.metric("Avg Time", f"{agent_data['avg_time']}s")
                with a_col4:
                    avg_retries = sum(
                        r.get("retries", 0) for r in agent_data["results"]
                    ) / max(len(agent_data["results"]), 1)
                    st.metric("Avg Retries", f"{avg_retries:.1f}")

                # Per-type accuracy
                if agent_data.get("by_type"):
                    st.markdown("**Accuracy by Problem Type:**")
                    type_cols = st.columns(len(agent_data["by_type"]))
                    for idx, (ptype, data) in enumerate(agent_data["by_type"].items()):
                        with type_cols[idx]:
                            color = PROBLEM_TYPE_COLORS.get(ptype, "#7f8c8d")
                            st.markdown(
                                f"<div style='text-align:center;'>"
                                f"<span style='color:{color}; font-size:1.2em;'>●</span> "
                                f"<b>{ptype.replace('_', ' ').title()}</b><br/>"
                                f"{data['correct']}/{data['total']} "
                                f"({data['accuracy']:.0%})</div>",
                                unsafe_allow_html=True,
                            )

                # Per-difficulty accuracy
                if agent_data.get("by_difficulty"):
                    st.markdown("**Accuracy by Difficulty:**")
                    diff_cols = st.columns(len(agent_data["by_difficulty"]))
                    for idx, (diff, data) in enumerate(agent_data["by_difficulty"].items()):
                        with diff_cols[idx]:
                            color = DIFFICULTY_COLORS.get(diff, "#7f8c8d")
                            emoji = DIFFICULTY_EMOJI.get(diff, "⚪")
                            st.markdown(
                                f"<div style='text-align:center;'>"
                                f"{emoji} <b>{diff.capitalize()}</b><br/>"
                                f"{data['correct']}/{data['total']} "
                                f"({data['accuracy']:.0%})</div>",
                                unsafe_allow_html=True,
                            )

            if "raw_llm" in results:
                raw_data = results["raw_llm"]
                st.markdown("---")
                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.metric("Raw LLM Correct", f"{raw_data['correct']}/{results['total']}")
                with r_col2:
                    st.metric("Raw LLM Accuracy", f"{raw_data['accuracy']:.1%}")
                with r_col3:
                    st.metric("Avg Time", f"{raw_data['avg_time']}s")

            # Save results
            try:
                from benchmark.results import ResultsStore

                store = ResultsStore(cfg.DATA_DIR)
                filepath = store.save_result(results["run_id"], results)
                st.success(f"💾 Results saved to: `{filepath}`")
            except Exception as save_exc:
                st.warning(f"Could not save results: {save_exc}")

            # ── Per-problem breakdown ────────────────────────────────────
            with st.expander("📋 Per-Problem Breakdown"):
                import pandas as pd

                display_results = []
                if "agent" in results:
                    for r in results["agent"]["results"]:
                        row = {
                            "Problem": r["problem"][:100] + "...",
                            "Gold Answer": r["gold_answer"],
                            "Predicted": r["predicted"],
                            "Correct": "✅" if r["correct"] else "❌",
                            "Type": r.get("problem_type", ""),
                            "Difficulty": r.get("difficulty", ""),
                            "Time(s)": round(r["time_taken"], 2),
                        }
                        if show_symbolic_details:
                            row["Retries"] = r.get("retries", 0)
                        display_results.append(row)

                if display_results:
                    df = pd.DataFrame(display_results)
                    st.dataframe(df, use_container_width=True)

            st.session_state["last_jee_results"] = results

        except Exception as exc:
            st.error(f"Benchmark error: {exc}")


# ---------------------------------------------------------------------------
# Tab 3: Dashboard
# ---------------------------------------------------------------------------
def render_dashboard_tab(cfg):
    """Render the JEE results dashboard with charts and visualizations."""
    st.header("🔷 JEE Results Dashboard")

    # ── Load saved runs ──────────────────────────────────────────────────
    try:
        from benchmark.results import ResultsStore

        store = ResultsStore(cfg.DATA_DIR)
        runs = store.list_runs()
    except Exception:
        runs = []

    if not runs:
        st.info("📭 No benchmark runs found. Run a JEE benchmark first!")
        return

    # Filter JEE runs
    jee_runs = [r for r in runs if r["run_id"].startswith("jee_")]

    if not jee_runs:
        st.info("📭 No JEE benchmark runs found yet. Run the JEE benchmark first!")

    # ── All Runs Table ───────────────────────────────────────────────────
    st.subheader("📊 Saved Benchmark Runs")

    import pandas as pd

    runs_df = pd.DataFrame(runs)
    st.dataframe(runs_df, use_container_width=True)

    # ── JEE-specific Dashboard ───────────────────────────────────────────
    if jee_runs:
        st.markdown("---")
        st.subheader("🔷 JEE-Specific Analytics")

        # Load the most recent JEE run for detailed analysis
        try:
            latest_jee = store.load_result(jee_runs[-1]["run_id"])
        except Exception:
            latest_jee = None

        if latest_jee and "agent" in latest_jee:
            agent_data = latest_jee["agent"]

            # Overall metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Problems", latest_jee["total"])
            with m2:
                st.metric("Accuracy", f"{agent_data['accuracy']:.1%}")
            with m3:
                st.metric("Correct", agent_data["correct"])
            with m4:
                st.metric("Incorrect", agent_data["incorrect"])

            # ── Problem Type Performance Bar Chart ─────────────────────
            if agent_data.get("by_type"):
                st.subheader("📐 Accuracy by Problem Type")

                type_names = list(agent_data["by_type"].keys())
                type_accs = [
                    agent_data["by_type"][t]["accuracy"] for t in type_names
                ]
                type_totals = [
                    agent_data["by_type"][t]["total"] for t in type_names
                ]
                type_colors = [
                    PROBLEM_TYPE_COLORS.get(t, "#7f8c8d") for t in type_names
                ]
                type_labels = [t.replace("_", " ").title() for t in type_names]

                fig_types = go.Figure()
                fig_types.add_trace(
                    go.Bar(
                        x=type_labels,
                        y=type_accs,
                        marker_color=type_colors,
                        text=[f"{a:.0%}" for a in type_accs],
                        textposition="auto",
                        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<br>Problems: %{customdata}<extra></extra>",
                        customdata=type_totals,
                    )
                )
                fig_types.update_layout(
                    title="Agent Accuracy by Problem Type",
                    yaxis=dict(range=[0, 1], tickformat=".0%", title="Accuracy"),
                    xaxis_title="Problem Type",
                    height=450,
                    template="plotly_white",
                )
                st.plotly_chart(fig_types, use_container_width=True)

            # ── Difficulty Performance ─────────────────────────────────
            if agent_data.get("by_difficulty"):
                st.subheader("🎯 Accuracy by Difficulty")

                diff_names = list(agent_data["by_difficulty"].keys())
                diff_accs = [
                    agent_data["by_difficulty"][d]["accuracy"] for d in diff_names
                ]
                diff_colors = [
                    DIFFICULTY_COLORS.get(d, "#7f8c8d") for d in diff_names
                ]
                diff_labels = [d.capitalize() for d in diff_names]

                fig_diff = go.Figure()
                fig_diff.add_trace(
                    go.Bar(
                        x=diff_labels,
                        y=diff_accs,
                        marker_color=diff_colors,
                        text=[f"{a:.0%}" for a in diff_accs],
                        textposition="auto",
                    )
                )
                fig_diff.update_layout(
                    title="Agent Accuracy by Difficulty Level",
                    yaxis=dict(range=[0, 1], tickformat=".0%", title="Accuracy"),
                    xaxis_title="Difficulty",
                    height=400,
                    template="plotly_white",
                )
                st.plotly_chart(fig_diff, use_container_width=True)

            # ── Difficulty Distribution Pie Chart ──────────────────────
            st.subheader("📈 Problem Distribution")

            try:
                from benchmark.jee_loader import JEELoader

                type_dist = JEELoader.get_type_distribution()
                diff_dist = JEELoader.get_difficulty_distribution()

                pie_col1, pie_col2 = st.columns(2)

                with pie_col1:
                    fig_pie_type = px.pie(
                        values=list(type_dist.values()),
                        names=[t.replace("_", " ").title() for t in type_dist.keys()],
                        title="Problems by Type",
                        color_discrete_sequence=list(PROBLEM_TYPE_COLORS.values()),
                    )
                    fig_pie_type.update_traces(
                        textposition="inside", textinfo="percent+label"
                    )
                    fig_pie_type.update_layout(height=400)
                    st.plotly_chart(fig_pie_type, use_container_width=True)

                with pie_col2:
                    fig_pie_diff = px.pie(
                        values=list(diff_dist.values()),
                        names=[d.capitalize() for d in diff_dist.keys()],
                        title="Problems by Difficulty",
                        color_discrete_sequence=list(DIFFICULTY_COLORS.values()),
                    )
                    fig_pie_diff.update_traces(
                        textposition="inside", textinfo="percent+label"
                    )
                    fig_pie_diff.update_layout(height=400)
                    st.plotly_chart(fig_pie_diff, use_container_width=True)

            except Exception as dist_exc:
                st.warning(f"Could not render distribution charts: {dist_exc}")

    # ── Comparison Section ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 Compare Runs")

    run_ids = [r["run_id"] for r in runs]
    if len(run_ids) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            run1_id = st.selectbox("Select Run 1", run_ids, index=0, key="cmp1")
        with c2:
            run2_id = st.selectbox("Select Run 2", run_ids, index=min(1, len(run_ids) - 1), key="cmp2")

        if st.button("Compare Selected Runs"):
            try:
                run1 = store.load_result(run1_id)
                run2 = store.load_result(run2_id)

                # Simple comparison display
                comp_cols = st.columns(2)
                for idx, (run, name) in enumerate([(run1, run1_id), (run2, run2_id)]):
                    with comp_cols[idx]:
                        st.markdown(f"**{name}**")
                        if "agent" in run:
                            st.metric("Agent Accuracy", f"{run['agent']['accuracy']:.1%}")
                            st.metric("Correct", f"{run['agent']['correct']}/{run['total']}")
                        if "raw_llm" in run:
                            st.metric("Raw LLM Accuracy", f"{run['raw_llm']['accuracy']:.1%}")

                # Side-by-side bar chart
                fig_cmp = go.Figure()
                categories = []
                acc1_vals = []
                acc2_vals = []

                for key in ["agent", "raw_llm"]:
                    if key in run1 and key in run2:
                        categories.append(key.replace("_", " ").title())
                        acc1_vals.append(run1[key]["accuracy"])
                        acc2_vals.append(run2[key]["accuracy"])

                if categories:
                    fig_cmp.add_trace(
                        go.Bar(
                            name=run1_id[:20],
                            x=categories,
                            y=acc1_vals,
                            marker_color="#3498db",
                        )
                    )
                    fig_cmp.add_trace(
                        go.Bar(
                            name=run2_id[:20],
                            x=categories,
                            y=acc2_vals,
                            marker_color="#e74c3c",
                        )
                    )
                    fig_cmp.update_layout(
                        title="Accuracy Comparison",
                        yaxis=dict(range=[0, 1], tickformat=".0%"),
                        barmode="group",
                        height=400,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

            except Exception as exc:
                st.error(f"Comparison error: {exc}")
    else:
        st.info("Need at least 2 saved runs to compare. Run more benchmarks!")

    # ── Historical Summary ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Historical Summary")

    if runs:
        agent_accs = [r["agent_accuracy"] for r in runs if r["agent_accuracy"] > 0]
        raw_accs = [r["raw_accuracy"] for r in runs if r["raw_accuracy"] > 0]

        hist_col1, hist_col2, hist_col3 = st.columns(3)
        with hist_col1:
            if agent_accs:
                st.metric("Best Agent Accuracy", f"{max(agent_accs):.1%}")
        with hist_col2:
            if raw_accs:
                st.metric("Best Raw LLM Accuracy", f"{max(raw_accs):.1%}")
        with hist_col3:
            st.metric("Total Runs", len(runs))

        # Time series chart
        if len(runs) > 1:
            ts_df = pd.DataFrame(runs)
            ts_fig = go.Figure()
            if any(ts_df["agent_accuracy"] > 0):
                ts_fig.add_trace(
                    go.Scatter(
                        x=ts_df["timestamp"],
                        y=ts_df["agent_accuracy"],
                        mode="lines+markers",
                        name="Agent Accuracy",
                        line=dict(color="#3498db", width=3),
                        marker=dict(size=8),
                    )
                )
            if any(ts_df["raw_accuracy"] > 0):
                ts_fig.add_trace(
                    go.Scatter(
                        x=ts_df["timestamp"],
                        y=ts_df["raw_accuracy"],
                        mode="lines+markers",
                        name="Raw LLM Accuracy",
                        line=dict(color="#e74c3c", width=3),
                        marker=dict(size=8),
                    )
                )
            ts_fig.update_layout(
                title="Accuracy Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                height=450,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(ts_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Main entry point for the JEE Streamlit app."""
    st.title("🔷 JEE Advanced Math Reasoning Agent")
    st.markdown(
        "A LangGraph-powered agent for solving **JEE Advanced calculus problems** "
        "with symbolic mathematics, LaTeX rendering, and SymPy verification."
    )

    cfg, selected_type, selected_difficulty = sidebar_config()
    render_dataset_overview()

    tab1, tab2, tab3 = st.tabs(
        ["🔷 Single Problem", "📊 JEE Benchmark", "📈 Dashboard"]
    )

    with tab1:
        render_single_problem_tab(cfg)

    with tab2:
        render_benchmark_tab(cfg, selected_type, selected_difficulty)

    with tab3:
        render_dashboard_tab(cfg)


if __name__ == "__main__":
    main()
