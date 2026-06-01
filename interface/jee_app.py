"""Streamlit interface for the JEE Advanced Math Agent.

Photomath-inspired clean, modern UI:
- Light theme with soft shadows and rounded corners
- Large camera/image upload as primary input
- Step-by-step solution timeline
- Clean math rendering with ample whitespace
"""

import base64
import io
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import time

from config import Config
from agent.jee_graph import JEEAgent
from benchmark.jee_evaluator import JEEEValuator
from benchmark.jee_loader import JEELoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Math Solver",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS: Clean Photomath-inspired light theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

    .stApp {
        background: linear-gradient(180deg, #f8f9fb 0%, #ffffff 100%) !important;
    }

    .main .block-container {
        background: transparent !important;
        padding-top: 1rem !important;
        max-width: 1100px !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- Top Navbar --- */
    .navbar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 16px 24px;
        margin: -1rem -1rem 0 -1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    }

    .navbar-icon {
        font-size: 1.4rem;
    }

    .navbar-title {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #1a1a2e !important;
        font-weight: 700 !important;
    }

    p, div, span, label {
        font-family: 'Inter', sans-serif !important;
        color: #4a4a5a !important;
    }

    span[data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
    }

    /* --- Cards --- */
    .math-card {
        background: #ffffff !important;
        border: 1px solid #e8e8f0 !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
    }

    .math-card:hover {
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.07) !important;
        transform: translateY(-1px) !important;
    }

    /* --- Primary Button (Solve) --- */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 16px 32px !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.45) !important;
    }

    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    /* --- Secondary Buttons --- */
    .stButton > button {
        background: #f0f0f5 !important;
        border: 1px solid #e0e0e8 !important;
        border-radius: 12px !important;
        color: #4a4a5a !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #e8e8f0 !important;
        border-color: #d0d0e0 !important;
    }

    /* Example card buttons */
    .math-card + .stButton > button {
        background: #ffffff !important;
        border: 1px solid #6366f1 !important;
        color: #6366f1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    .math-card + .stButton > button:hover {
        background: #f5f5ff !important;
    }

    /* --- Text area --- */
    .stTextArea > div,
    .stTextArea [data-baseweb="textarea"] {
        border: none !important;
        background: transparent !important;
    }

    .stTextArea textarea {
        background: #ffffff !important;
        border: 2px solid #e8e8f0 !important;
        border-radius: 16px !important;
        color: #1a1a2e !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        padding: 16px !important;
        transition: all 0.2s ease !important;
        resize: vertical !important;
    }

    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
        outline: none !important;
    }

    .stTextArea [data-baseweb="base-input"] {
        border: none !important;
        background: transparent !important;
    }

    /* --- File uploader --- */
    .stFileUploader {
        background: #ffffff !important;
        border: 2px dashed #c7c7d5 !important;
        border-radius: 16px !important;
        padding: 8px !important;
        transition: all 0.2s ease !important;
    }

    .stFileUploader:hover {
        border-color: #6366f1 !important;
        background: #fafaff !important;
    }

    .stFileUploader > section,
    .stFileUploader > section > div,
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
    }

    .stFileUploader button,
    .stFileUploader [data-testid="stBaseButton-secondary"] {
        background: #f0f0f5 !important;
        border: 1px solid #d0d0e0 !important;
        border-radius: 10px !important;
        color: #6366f1 !important;
        font-weight: 500 !important;
    }

    .stFileUploader button:hover {
        background: #e8e8f5 !important;
        border-color: #6366f1 !important;
    }

    .stFileUploader p,
    .stFileUploader span,
    .stFileUploader small {
        color: #6b7280 !important;
    }

    .stFileUploader > label {
        display: none !important;
    }

    /* --- Metrics --- */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e8e8f0 !important;
        border-radius: 16px !important;
        padding: 16px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        color: #8a8a9a !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        color: #1a1a2e !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    /* --- Expander --- */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border: 1px solid #e8e8f0 !important;
        border-radius: 14px !important;
        color: #1a1a2e !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 14px 18px !important;
    }

    .streamlit-expanderContent {
        background: #ffffff !important;
        border: 1px solid #e8e8f0 !important;
        border-top: none !important;
        border-radius: 0 0 14px 14px !important;
        padding: 0 18px 18px !important;
    }

    /* --- Code blocks --- */
    .stCode pre {
        background: #f8f9fb !important;
        border: 1px solid #e8e8f0 !important;
        border-radius: 12px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* --- Scrollbar --- */
    ::-webkit-scrollbar { width: 6px !important; }
    ::-webkit-scrollbar-track { background: transparent !important; }
    ::-webkit-scrollbar-thumb { background: #c7c7d5 !important; border-radius: 3px !important; }
    ::-webkit-scrollbar-thumb:hover { background: #a0a0b0 !important; }

    /* --- Divider --- */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #e8e8f0, transparent) !important;
        margin: 24px 0 !important;
    }

    /* --- Step timeline --- */
    .step-card {
        background: #ffffff;
        border: 1px solid #e8e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        position: relative;
        transition: all 0.2s ease;
    }

    .step-card:hover {
        border-color: #c7c7ff;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.08);
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: #fff;
        border-radius: 50%;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 12px;
    }

    .step-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1a1a2e;
        font-size: 0.95rem;
    }

    /* --- Answer display --- */
    .answer-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        margin-bottom: 16px;
    }

    .answer-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.8;
        margin-bottom: 8px;
    }

    .answer-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* --- Badges --- */
    .topic-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 100px;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* --- Spinner --- */
    .stSpinner > div {
        border-color: rgba(99, 102, 241, 0.2) !important;
        border-top-color: #6366f1 !important;
    }

    /* --- Image preview --- */
    .img-preview {
        border-radius: 12px;
        border: 1px solid #e8e8f0;
        overflow: hidden;
    }

    /* --- Processing state --- */
    .processing-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: #f0f0ff;
        border: 1px solid #d0d0ff;
        border-radius: 100px;
        color: #6366f1;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .dot-pulse {
        width: 6px;
        height: 6px;
        background: #6366f1;
        border-radius: 50%;
        animation: dotPulse 1.4s ease-in-out infinite;
    }

    .dot-pulse:nth-child(2) { animation-delay: 0.2s; }
    .dot-pulse:nth-child(3) { animation-delay: 0.4s; }

    @keyframes dotPulse {
        0%, 100% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
TYPE_BADGE_STYLES = {
    "limits": "background:#fef3c7; color:#d97706;",
    "differentiation": "background:#dbeafe; color:#2563eb;",
    "integration": "background:#d1fae5; color:#059669;",
    "definite_integrals": "background:#d1fae5; color:#059669;",
    "differential_equations": "background:#fce7f3; color:#db2777;",
    "maxima_minima": "background:#ffedd5; color:#ea580c;",
    "tangent_normal": "background:#e0e7ff; color:#4f46e5;",
    "area": "background:#ccfbf1; color:#0d9488;",
    "complex_numbers": "background:#f3e8ff; color:#7c3aed;",
    "quadratic": "background:#fee2e2; color:#dc2626;",
    "permutations_combinations": "background:#ecfccb; color:#65a30d;",
    "binomial_theorem": "background:#cffafe; color:#0891b2;",
    "matrices_determinants": "background:#e0f2fe; color:#0284c7;",
    "probability": "background:#fce7f3; color:#db2777;",
    "sequences_series": "background:#fef9c3; color:#ca8a04;",
    "trigonometric_identities": "background:#dbeafe; color:#2563eb;",
    "trigonometric_equations": "background:#dbeafe; color:#2563eb;",
    "solution_of_triangles": "background:#dbeafe; color:#2563eb;",
    "inverse_trig": "background:#dbeafe; color:#2563eb;",
    "straight_lines": "background:#f3e8ff; color:#7c3aed;",
    "circles": "background:#f3e8ff; color:#7c3aed;",
    "parabola": "background:#f3e8ff; color:#7c3aed;",
    "ellipse": "background:#f3e8ff; color:#7c3aed;",
    "hyperbola": "background:#f3e8ff; color:#7c3aed;",
    "vectors": "background:#ffedd5; color:#ea580c;",
    "three_d_geometry": "background:#ffedd5; color:#ea580c;",
    "statistics": "background:#ccfbf1; color:#0d9488;",
    "mathematical_reasoning": "background:#e0e7ff; color:#4f46e5;",
}

DIFFICULTY_BADGE_STYLES = {
    "easy": "background:#d1fae5; color:#059669;",
    "medium": "background:#fef3c7; color:#d97706;",
    "hard": "background:#fee2e2; color:#dc2626;",
}

# ---------------------------------------------------------------------------
# Load 3 curated example problems
# ---------------------------------------------------------------------------
@st.cache_data
def get_example_problems():
    """Return 3 diverse example problems from the dataset."""
    all_problems = JEELoader.load_problems()
    examples = []
    for target_type in ["limits", "differentiation", "integration"]:
        for p in all_problems:
            if p["problem_type"] == target_type:
                examples.append(p)
                break
    return examples


# ---------------------------------------------------------------------------
# Initialize config & agent
# ---------------------------------------------------------------------------
@st.cache_resource
def get_config():
    return Config()


@st.cache_resource
def get_agent(cfg):
    return JEEAgent(cfg)


@st.cache_resource
def get_evaluator(agent, cfg):
    return JEEEValuator(agent, cfg)


# ---------------------------------------------------------------------------
# Render header
# ---------------------------------------------------------------------------
def render_header():
    """Render clean app header with navbar."""
    st.markdown(
        """
        <div class="navbar">
            <span class="navbar-icon">📐</span>
            <span class="navbar-title">Math Solver</span>
        </div>
        <div style="text-align:center; margin: 24px 0 8px 0;">
            <h1 style="font-size: 1.5rem; margin-bottom: 6px; font-weight: 800; color: #1a1a2e;">Solve Any Math Problem</h1>
            <p style="color: #8a8a9a; font-size: 0.9rem; margin-bottom: 0;">
                Type your problem or upload an image for step-by-step solutions
            </p>
        </div>
        <hr style="margin: 20px 0 24px 0;">
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Render example problem cards
# ---------------------------------------------------------------------------
def render_example_cards(examples):
    """Render 3 example problem cards with Solve buttons."""
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 1rem; margin-bottom: 4px;">Example Problems</h3>
            <p style="color: #8a8a9a; font-size: 0.85rem; margin: 0;">
                Try one of these curated problems
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for idx, prob in enumerate(examples):
        with cols[idx]:
            ptype = prob["problem_type"]
            diff = prob["difficulty"]
            topic_style = TYPE_BADGE_STYLES.get(ptype, "background:#e0e7ff; color:#4f46e5;")
            diff_style = DIFFICULTY_BADGE_STYLES.get(diff, "background:#f0f0f5; color:#8a8a9a;")

            st.markdown(
                f"""
                <div class="math-card" style="padding: 16px !important; height: 100%;">
                    <div style="margin-bottom: 10px; display: flex; gap: 6px; flex-wrap: wrap;">
                        <span class="topic-badge" style="{topic_style}">
                            {ptype.replace('_', ' ').title()}
                        </span>
                        <span class="topic-badge" style="{diff_style}">
                            {diff.title()}
                        </span>
                    </div>
                    <div style="font-size: 0.85rem; color: #5a5a6a; min-height: 60px; line-height: 1.5;">
                        {prob['question'][:110]}...
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Solve", key=f"example_{idx}", use_container_width=True):
                st.session_state["problem_input"] = prob["question"]
                st.session_state["run_mode"] = "agent"
                st.rerun()


# ---------------------------------------------------------------------------
# Render input area
# ---------------------------------------------------------------------------
def render_input_area():
    """Render problem input with image upload and solve button."""
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 1rem; margin-bottom: 4px;">Your Problem</h3>
            <p style="color: #8a8a9a; font-size: 0.85rem; margin: 0;">
                Type your math problem or upload an image
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "problem_input" not in st.session_state:
        st.session_state["problem_input"] = ""
    if "image_data" not in st.session_state:
        st.session_state["image_data"] = None

    problem = st.text_area(
        "",
        value=st.session_state["problem_input"],
        height=120,
        key="problem_text_area",
        label_visibility="collapsed",
        placeholder="e.g. Find the derivative of sin(x^2) with respect to x...",
    )

    if problem != st.session_state.get("_last_seen_text", ""):
        st.session_state["_last_seen_text"] = problem

    # Image upload (prominent)
    uploaded_image = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg"],
        key="image_uploader",
        label_visibility="collapsed",
    )
    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        st.session_state["image_data"] = b64
        # Show preview
        st.markdown(
            f"""
            <div class="img-preview" style="margin-bottom: 12px;">
                <img src="data:image/png;base64,{b64}" style="width: 100%; max-height: 200px; object-fit: contain; border-radius: 12px;">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.session_state["image_data"] = None

    # Solve button
    solve_clicked = st.button("Solve", use_container_width=True, type="primary")

    if solve_clicked:
        st.session_state["problem_input"] = problem
        st.session_state["run_mode"] = "agent"
        st.rerun()

    return problem


# ---------------------------------------------------------------------------
# Render solution panel
# ---------------------------------------------------------------------------
def render_solution_panel(result: dict | None):
    """Render the answer and step-by-step solution."""
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 1rem; margin-bottom: 4px;">Solution</h3>
            <p style="color: #8a8a9a; font-size: 0.85rem; margin: 0;">
                Step-by-step breakdown with final answer
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result is None:
        st.markdown(
            """
            <div class="math-card" style="min-height: 200px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 8px; opacity: 0.3;">📝</div>
                    <div style="color: #aaa; font-size: 0.9rem; font-weight: 500;">
                        Enter a problem to see the solution
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    final_answer = result.get("final_answer", "N/A")
    latex_answer = result.get("latex_answer", "")
    retries = result.get("retry_count", 0)
    problem_type = result.get("problem_type", "")
    confidence = result.get("confidence", 0)
    time_taken = result.get("_time_taken", "N/A")
    reasoning_trace = result.get("reasoning_trace", [])
    steps = result.get("steps", [])
    sympy_code = result.get("sympy_code", "")
    sympy_result = result.get("sympy_result", "")
    verification = result.get("verification", "")
    reflection = result.get("reflection", "")

    answer_display = latex_answer if latex_answer else str(final_answer)

    # Final answer box
    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-label">Final Answer</div>
            <div class="answer-value">{answer_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # LaTeX rendering below the answer box
    if latex_answer:
        st.latex(latex_answer.replace("$", ""))
    elif final_answer and final_answer != "N/A":
        try:
            st.latex(str(final_answer).replace("$", ""))
        except Exception:
            st.code(str(final_answer))

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Confidence", f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A")
    with m2:
        st.metric("Time", f"{time_taken}s" if isinstance(time_taken, (int, float)) else str(time_taken))
    with m3:
        st.metric("Retries", int(retries))

    # Topic badge
    if problem_type:
        topic_style = TYPE_BADGE_STYLES.get(problem_type, "background:#e0e7ff; color:#4f46e5;")
        st.markdown(
            f"""
            <div style="margin: 14px 0; text-align: center;">
                <span class="topic-badge" style="{topic_style}">
                    {problem_type.replace('_', ' ').title()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Step-by-step solution
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 1rem; margin-bottom: 4px;">Step-by-Step</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if steps:
        for i, step in enumerate(steps, 1):
            desc = ""
            latex = ""
            step_result = ""
            if isinstance(step, dict):
                desc = step.get("description", "")
                latex = step.get("latex", "")
                step_result = step.get("result", "")
            else:
                desc = str(step)

            st.markdown(
                f"""
                <div class="step-card">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span class="step-number">{i}</span>
                        <span class="step-title">{desc}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if latex:
                st.latex(latex.replace("$", ""))
            if step_result:
                st.markdown(f"<div style='color:#5a5a6a; font-size:0.9rem; margin-left:40px;'>{step_result}</div>", unsafe_allow_html=True)
    elif reasoning_trace:
        for i, item in enumerate(reasoning_trace, 1):
            st.markdown(
                f"""
                <div class="step-card">
                    <div style="display: flex; align-items: center;">
                        <span class="step-number">{i}</span>
                        <span class="step-title">{str(item)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div style="color: #8a8a9a; font-size: 0.9rem; text-align: center; padding: 20px;">
                No step details available for this solution.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Thinking / internals expander
    internals = []
    if sympy_code:
        internals.append(("SymPy Code", sympy_code))
    if sympy_result:
        internals.append(("SymPy Output", sympy_result))
    if verification:
        internals.append(("Verification", verification))
    if reflection:
        internals.append(("Reflection", reflection))

    if internals:
        with st.expander("View Engine Details", expanded=False):
            for name, content in internals:
                st.markdown(
                    f"""
                    <div style="color:#6366f1; font-family:'Inter',sans-serif; font-size:0.8rem; font-weight:600; margin: 14px 0 8px; text-transform: uppercase; letter-spacing: 0.5px;">
                        {name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.code(str(content), language="python" if name == "SymPy Code" else "text")


# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------
def run_agent(problem: str, cfg: Config):
    """Run the JEEAgent on a problem and return result dict."""
    agent = get_agent(cfg)
    image_data = st.session_state.get("image_data")
    result = agent.solve(problem, image_data=image_data)
    return result


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Header
    render_header()

    # Config
    cfg = get_config()
    try:
        cfg.validate()
    except ValueError as exc:
        st.error(f"⚠️ {exc}")
        st.stop()

    # ---- TWO COLUMN LAYOUT ----
    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        # Input area
        problem = render_input_area()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Example problems
        examples = get_example_problems()
        render_example_cards(examples)

    with right_col:
        # Solution panel (always visible, updates when solved)
        run_mode = st.session_state.get("run_mode")
        problem_to_solve = st.session_state.get("problem_input", "")
        image_data = st.session_state.get("image_data")

        if run_mode == "agent" and (problem_to_solve or image_data):
            # Show processing state
            st.markdown(
                """
                <div class="math-card" style="text-align: center; padding: 60px 24px;">
                    <div style="margin-bottom: 16px;">
                        <span class="processing-pill">
                            <span class="dot-pulse"></span>
                            <span class="dot-pulse"></span>
                            <span class="dot-pulse"></span>
                            Solving...
                        </span>
                    </div>
                    <div style="color: #8a8a9a; font-size: 0.85rem;">
                        Running symbolic computation and verification
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner(""):
                try:
                    result = run_agent(problem_to_solve, cfg)
                    st.session_state["last_result"] = result
                    st.session_state["run_mode"] = None
                    st.rerun()
                except Exception as exc:
                    st.session_state["run_mode"] = None
                    st.error(f"Agent error: {exc}")
        else:
            result = st.session_state.get("last_result")
            render_solution_panel(result)


if __name__ == "__main__":
    main()
