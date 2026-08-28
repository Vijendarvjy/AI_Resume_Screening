import os
import re
import json
import time
import tempfile
import hashlib
import copy
from datetime import datetime
from typing import TypedDict, Optional, Tuple, Any

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "fallback_models": [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
    ],
    "max_tokens": 1024,
    "resume_char_limit": 8000,
    "jd_char_limit": 6000,
    "max_retries_per_model": 2,
    "retry_base_delay": 2,
}


# ============================================================
# ENVIRONMENT / SECRETS
# ============================================================

load_dotenv()

GROQ_API_KEY = None

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = None

if not GROQ_API_KEY:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# LLM CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    """
    Create and cache one ChatGroq client per model.
    """

    if not GROQ_API_KEY:
        return None

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model_name,
        max_tokens=CONFIG["max_tokens"],
        temperature=0,
    )


# ============================================================
# LANGGRAPH STATE
# ============================================================

class ResumeState(TypedDict, total=False):
    candidate_name: str
    resume_text: str
    job_description: str
    parsed_resume: dict
    jd_analysis: str
    match_score: str
    recommendation: str
    interview_questions: str
    model_used: str


# ============================================================
# CONSTANTS
# ============================================================

EMPTY_RESUME = {
    "name": "Unknown",
    "email": "",
    "phone": "",
    "skills": [],
    "experience_years": 0,
    "education": "",
    "certifications": [],
    "projects": [],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def trim(text: str, limit: int) -> str:
    """
    Trim text to the specified character limit.
    """

    if text is None:
        return ""

    text = str(text)

    if len(text) <= limit:
        return text

    truncated = text[:limit]

    last_period = truncated.rfind(". ")

    if last_period > limit * 0.8:
        truncated = truncated[:last_period + 1]

    return truncated


def safe_string(value: Any) -> str:
    """
    Convert a value safely to string.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return str(value)


def safe_list(value: Any) -> list:
    """
    Convert a value to a safe list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def safe_invoke(
    prompt: str,
    fallback: str = "Unavailable"
) -> Tuple[str, Optional[str]]:
    """
    Invoke Groq with retry and model fallback.

    Returns:
        (response_text, model_name)
    """

    last_err = None

    for model_name in CONFIG["fallback_models"]:

        llm = get_llm(model_name)

        if llm is None:
            continue

        for attempt in range(CONFIG["max_retries_per_model"]):

            try:
                response = llm.invoke(prompt)

                content = getattr(response, "content", "")

                if isinstance(content, list):
                    content = "".join(
                        str(item.get("text", item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in content
                    )

                content = str(content).strip()

                if content:
                    return content, model_name

                last_err = Exception(
                    f"Empty response received from {model_name}"
                )

            except Exception as e:

                last_err = e

                msg = str(e).lower()

                is_rate_limit = (
                    "429" in msg
                    or "rate_limit" in msg
                    or "rate limit" in msg
                    or "too many requests" in msg
                )

                if is_rate_limit:

                    if attempt < CONFIG["max_retries_per_model"] - 1:

                        wait = CONFIG["retry_base_delay"] * (
                            2 ** attempt
                        )

                        time.sleep(wait)

                        continue

                    # Move to next model
                    break

                # Non-rate-limit errors should not repeatedly
                # hit the same model.
                break

    if last_err:
        return f"{fallback}: {last_err}", None

    return fallback, None


def extract_json(raw: str) -> dict:
    """
    Extract JSON from an LLM response.

    Handles:
    - Plain JSON
    - Markdown JSON fences
    - Additional explanatory text
    """

    if not raw:
        return {}

    cleaned = str(raw).strip()

    # Remove markdown code fences
    cleaned = re.sub(
        r"```(?:json|JSON)?",
        "",
        cleaned
    )

    cleaned = cleaned.replace("```", "").strip()

    # First attempt: entire response
    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Second attempt: locate first JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = cleaned[start:end + 1]

        try:
            result = json.loads(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    return {}


def empty_resume() -> dict:
    """
    Return a completely independent fallback dictionary.
    """

    return copy.deepcopy(EMPTY_RESUME)


def normalize_resume(parsed: dict) -> dict:
    """
    Normalize LLM-generated resume JSON so UI never crashes
    because of missing or incorrectly typed fields.
    """

    if not isinstance(parsed, dict):
        parsed = empty_resume()

    result = empty_resume()

    result["name"] = safe_string(
        parsed.get("name", "Unknown")
    ).strip() or "Unknown"

    result["email"] = safe_string(
        parsed.get("email", "")
    ).strip()

    result["phone"] = safe_string(
        parsed.get("phone", "")
    ).strip()

    result["skills"] = [
        safe_string(x).strip()
        for x in safe_list(parsed.get("skills", []))
        if safe_string(x).strip()
    ]

    experience = parsed.get("experience_years", 0)

    try:
        result["experience_years"] = float(experience)

        if result["experience_years"].is_integer():
            result["experience_years"] = int(
                result["experience_years"]
            )

    except (ValueError, TypeError):
        result["experience_years"] = 0

    result["education"] = safe_string(
        parsed.get("education", "")
    ).strip()

    result["certifications"] = [
        safe_string(x).strip()
        for x in safe_list(
            parsed.get("certifications", [])
        )
        if safe_string(x).strip()
    ]

    result["projects"] = [
        safe_string(x).strip()
        for x in safe_list(
            parsed.get("projects", [])
        )
        if safe_string(x).strip()
    ]

    return result


def content_hash(
    candidate_name: str,
    resume_text: str,
    job_description: str
) -> str:

    raw = (
        candidate_name
        + "||"
        + resume_text
        + "||"
        + job_description
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def pct_from_text(text: str) -> Optional[int]:
    """
    Extract match percentage from LLM output.
    """

    if not text:
        return None

    text = str(text)

    # Preferred pattern
    patterns = [
        r"\*\*Match Percentage:\*\*\s*(\d{1,3})\s*%",
        r"Match Percentage\s*:\s*(\d{1,3})\s*%",
        r"Match\s*:\s*(\d{1,3})\s*%",
        r"(\d{1,3})\s*%",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            try:
                value = int(match.group(1))

                return max(0, min(100, value))

            except ValueError:
                continue

    return None


def decision_from_text(text: str) -> str:
    """
    Extract hiring decision from recommendation.
    """

    if not text:
        return "Unknown"

    patterns = [
        r"\*\*Decision:\*\*\s*(Hire|Reject|Consider)",
        r"Decision\s*:\s*(Hire|Reject|Consider)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip().title()

    # Fallback
    text_lower = text.lower()

    if "hire" in text_lower:
        return "Hire"

    if "reject" in text_lower:
        return "Reject"

    if "consider" in text_lower:
        return "Consider"

    return "Unknown"


# ============================================================
# PDF EXTRACTION
# ============================================================

@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text from a PDF using PyPDFLoader.
    """

    tmp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(file_bytes)
            tmp.flush()

            tmp_path = tmp.name

        pages = PyPDFLoader(tmp_path).load()

        text = "\n".join(
            page.page_content
            for page in pages
        )

        return text.strip()

    except Exception as e:

        raise RuntimeError(
            f"PDF extraction failed: {e}"
        ) from e

    finally:

        if tmp_path and os.path.exists(tmp_path):

            try:
                os.unlink(tmp_path)

            except OSError:
                pass


# ============================================================
# LANGGRAPH PIPELINE
# ============================================================

def make_graph():

    # --------------------------------------------------------
    # NODE 1: PARSE RESUME
    # --------------------------------------------------------

    def parse_resume(state: ResumeState):

        prompt = f"""
You are an expert resume parser.

Extract structured information from the resume below.

Return ONLY a valid JSON object.
Do not return markdown.
Do not return explanations.

Use exactly these fields:

{{
  "name": "string",
  "email": "string",
  "phone": "string",
  "skills": ["skill1", "skill2"],
  "experience_years": 0,
  "education": "string",
  "certifications": ["certification1"],
  "projects": ["project1"]
}}

Rules:
- Do not invent information.
- If information is unavailable, use an empty string or empty list.
- experience_years must be a number.
- Extract technical and professional skills.
- Projects should contain project titles.

Resume:
-------------------------
{trim(
    state.get("resume_text", ""),
    CONFIG["resume_char_limit"]
)}
-------------------------
"""

        raw, model_used = safe_invoke(
            prompt,
            "Resume parsing failed"
        )

        parsed = extract_json(raw)

        if not parsed:

            parsed = empty_resume()

            candidate_name = state.get(
                "candidate_name",
                ""
            )

            if candidate_name:
                parsed["name"] = candidate_name

        parsed = normalize_resume(parsed)

        return {
            "parsed_resume": parsed,
            "model_used": (
                model_used
                or state.get("model_used", "")
            )
        }

    # --------------------------------------------------------
    # NODE 2: ANALYZE JOB DESCRIPTION
    # --------------------------------------------------------

    def analyze_jd(state: ResumeState):

        resume_text = trim(
            state.get("resume_text", ""),
            3000
        )

        job_description = trim(
            state.get("job_description", ""),
            CONFIG["jd_char_limit"]
        )

        prompt = f"""
You are an expert technical recruiter.

Compare the candidate resume with the job description.

Resume:
-------------------------
{resume_text}
-------------------------

Job Description:
-------------------------
{job_description}
-------------------------

Respond with exactly these sections:

**Matching Skills:**
- skill
- skill

**Missing Skills:**
- skill
- skill

**Fit Score:** X/100 — one sentence explaining the score.

Be factual.
Do not invent candidate experience.
"""

        raw, model_used = safe_invoke(
            prompt,
            "JD analysis failed"
        )

        return {
            "jd_analysis": raw,
            "model_used": (
                model_used
                or state.get("model_used", "")
            )
        }

    # --------------------------------------------------------
    # NODE 3: MATCH SCORE
    # --------------------------------------------------------

    def calculate_match(state: ResumeState):

        parsed_resume = state.get(
            "parsed_resume",
            {}
        )

        jd_analysis = state.get(
            "jd_analysis",
            ""
        )

        prompt = f"""
You are an ATS and recruitment scoring expert.

Evaluate the candidate against the job requirements.

Parsed Resume:
-------------------------
{trim(
    json.dumps(
        parsed_resume,
        ensure_ascii=False
    ),
    2000
)}
-------------------------

JD Analysis:
-------------------------
{trim(jd_analysis, 2500)}
-------------------------

Calculate a realistic match percentage from 0 to 100.

Respond exactly in this format:

**Match Percentage:** X%

**Top 3 Strengths:**
1. ...
2. ...
3. ...

**Top 3 Gaps:**
1. ...
2. ...
3. ...

Do not inflate the score.
Base the score only on the available evidence.
"""

        raw, model_used = safe_invoke(
            prompt,
            "Match scoring failed"
        )

        return {
            "match_score": raw,
            "model_used": (
                model_used
                or state.get("model_used", "")
            )
        }

    # --------------------------------------------------------
    # NODE 4: RECOMMENDATION
    # --------------------------------------------------------

    def generate_recommendation(
        state: ResumeState
    ):

        match_score = state.get(
            "match_score",
            ""
        )

        prompt = f"""
You are a senior hiring manager.

Based on the match report below, provide a hiring recommendation.

Match Report:
-------------------------
{trim(match_score, 2500)}
-------------------------

Respond exactly:

**Decision:** Hire / Reject / Consider

**Confidence:** High / Medium / Low

**Reasoning:** One concise paragraph.

**Suggested Next Step:** One sentence.

Important:
- Do not recommend Hire only because the candidate has some matching skills.
- Consider missing critical skills.
- Use Consider when additional assessment is needed.
"""

        raw, model_used = safe_invoke(
            prompt,
            "Recommendation failed"
        )

        return {
            "recommendation": raw,
            "model_used": (
                model_used
                or state.get("model_used", "")
            )
        }

    # --------------------------------------------------------
    # NODE 5: INTERVIEW QUESTIONS
    # --------------------------------------------------------

    def generate_questions(
        state: ResumeState
    ):

        parsed_resume = state.get(
            "parsed_resume",
            {}
        )

        skills = safe_list(
            parsed_resume.get("skills", [])
        )

        skills_str = ", ".join(
            safe_string(skill)
            for skill in skills[:15]
        )

        if not skills_str:
            skills_str = (
                "general software engineering"
            )

        experience = parsed_resume.get(
            "experience_years",
            0
        )

        prompt = f"""
You are a senior technical interviewer.

Generate 10 technical interview questions for this candidate.

Candidate Skills:
{skills_str}

Experience:
Approximately {experience} years.

Requirements:
- Number questions from 1 to 10.
- Mix beginner, intermediate and advanced questions.
- Questions should be practical.
- Focus on the candidate's actual skills.
- Avoid generic HR questions.
- Do not provide answers.
- No preamble.

Return only the questions.
"""

        raw, model_used = safe_invoke(
            prompt,
            "Question generation failed"
        )

        return {
            "interview_questions": raw,
            "model_used": (
                model_used
                or state.get("model_used", "")
            )
        }

    # ========================================================
    # BUILD GRAPH
    # ========================================================

    workflow = StateGraph(ResumeState)

    workflow.add_node(
        "parse_resume",
        parse_resume
    )

    workflow.add_node(
        "analyze_jd",
        analyze_jd
    )

    workflow.add_node(
        "calculate_match",
        calculate_match
    )

    workflow.add_node(
        "generate_recommendation",
        generate_recommendation
    )

    workflow.add_node(
        "generate_questions",
        generate_questions
    )

    workflow.set_entry_point(
        "parse_resume"
    )

    workflow.add_edge(
        "parse_resume",
        "analyze_jd"
    )

    workflow.add_edge(
        "analyze_jd",
        "calculate_match"
    )

    workflow.add_edge(
        "calculate_match",
        "generate_recommendation"
    )

    workflow.add_edge(
        "generate_recommendation",
        "generate_questions"
    )

    workflow.add_edge(
        "generate_questions",
        END
    )

    return workflow.compile()


# ============================================================
# CACHED PIPELINE
# ============================================================

@st.cache_data(
    show_spinner=False,
    ttl=60 * 60 * 6
)
def run_pipeline_cached(
    cache_version: str,
    candidate_name: str,
    resume_text: str,
    job_description: str
) -> dict:

    graph = make_graph()

    result = graph.invoke(
        {
            "candidate_name": candidate_name,
            "resume_text": resume_text,
            "job_description": job_description,
            "parsed_resume": {},
            "jd_analysis": "",
            "match_score": "",
            "recommendation": "",
            "interview_questions": "",
            "model_used": "",
        }
    )

    return dict(result)


# ============================================================
# CHARTS
# ============================================================

def match_gauge(
    pct: int,
    key: str
):

    pct = max(
        0,
        min(100, int(pct))
    )

    if pct >= 70:
        color = "#15803D"
    elif pct >= 45:
        color = "#B45309"
    else:
        color = "#B91C1C"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,

            number={
                "suffix": "%",
                "font": {
                    "family": "IBM Plex Mono, monospace",
                    "color": "#0B1220"
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#E4E7EC"
                },

                "bar": {
                    "color": color
                },

                "bgcolor": "#FFFFFF",

                "bordercolor": "#E4E7EC",

                "steps": [
                    {
                        "range": [0, 45],
                        "color": "#FEE2E2"
                    },
                    {
                        "range": [45, 70],
                        "color": "#FEF3C7"
                    },
                    {
                        "range": [70, 100],
                        "color": "#DCFCE7"
                    }
                ]
            },

            domain={
                "x": [0, 1],
                "y": [0, 1]
            }
        )
    )

    fig.update_layout(
        height=220,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=10
        ),
        paper_bgcolor="#FFFFFF",
        font={
            "family": "Inter, sans-serif",
            "color": "#0B1220"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key
    )


def comparison_bar_chart(
    history: list
):

    if not history:
        return

    names = [
        h["candidate_name"]
        for h in history
    ]

    percentages = [
        h["match_pct"]
        if h["match_pct"] is not None
        else 0
        for h in history
    ]

    colors = [
        "#15803D"
        if p >= 70
        else (
            "#B45309"
            if p >= 45
            else "#B91C1C"
        )
        for p in percentages
    ]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=percentages,
            marker_color=colors,
            text=[
                f"{p}%"
                for p in percentages
            ],
            textposition="auto"
        )
    )

    fig.update_layout(
        height=340,

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),

        yaxis=dict(
            title="Match %",
            range=[0, 100],
            gridcolor="#E4E7EC"
        ),

        xaxis=dict(
            linecolor="#E4E7EC"
        ),

        title=dict(
            text="Candidate comparison",
            font=dict(
                family="Sora, sans-serif",
                size=16,
                color="#0B1220"
            )
        ),

        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",

        font={
            "family": "Inter, sans-serif",
            "color": "#0B1220"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="comparison_chart"
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap'
);

:root {
    --ink: #0B1220;
    --ink-soft: #475467;
    --bg: #FFFFFF;
    --surface: #F7F8FB;
    --border: #E4E7EC;
    --indigo: #4F46E5;
    --indigo-soft: #EEF2FF;
    --teal: #0E7490;
    --teal-soft: #ECFEFF;
    --green: #15803D;
    --green-soft: #DCFCE7;
    --red: #B91C1C;
    --red-soft: #FEE2E2;
    --amber: #B45309;
    --amber-soft: #FEF3C7;
}

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--bg);
}

h1,
h2,
h3,
.stTitle,
[data-testid="stMarkdownContainer"] h1 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.01em;
}

[data-testid="stMarkdownContainer"] h3 {
    font-weight: 600;
}

.app-subtitle {
    font-family: 'Inter', sans-serif;
    color: var(--ink-soft);
    font-size: 0.95rem;
    margin-top: -8px;
    margin-bottom: 1.6rem;
}

.eyebrow {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.eyebrow-indigo {
    color: var(--indigo);
}

.eyebrow-teal {
    color: var(--teal);
}

.panel {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px 6px 22px;
    margin-bottom: 18px;
}

.panel-indigo {
    border-top: 3px solid var(--indigo);
}

.panel-teal {
    border-top: 3px solid var(--teal);
}

.metric-card {
    background: var(--surface);
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid var(--indigo);
    margin-bottom: 12px;
}

.badge-hire,
.badge-reject,
.badge-consider,
.badge-unknown {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 5px 14px;
    border-radius: 20px;
    text-transform: uppercase;
}

.badge-hire {
    background: var(--green-soft);
    color: var(--green);
}

.badge-reject {
    background: var(--red-soft);
    color: var(--red);
}

.badge-consider {
    background: var(--amber-soft);
    color: var(--amber);
}

.badge-unknown {
    background: #F2F4F7;
    color: #475467;
}

.model-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--teal);
    background: var(--teal-soft);
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
    border: 1px solid #CFFAFE;
}

.skill-chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--indigo);
    background: var(--indigo-soft);
    border: 1px solid #E0E7FF;
    padding: 3px 10px;
    border-radius: 6px;
    margin: 0 6px 6px 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding: 0 18px;
    border-radius: 8px;
    font-weight: 600;
    color: var(--ink-soft);
}

.stTabs [aria-selected="true"] {
    background: var(--bg) !important;
    color: var(--indigo) !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}

.block-container {
    padding-top: 2rem;
}

.stButton > button[kind="primary"] {
    background: var(--indigo);
    border-color: var(--indigo);
}

hr {
    border-color: var(--border);
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 AI Resume Screening System"
)

st.markdown(
    """
<div class="app-subtitle">
Enterprise resume intelligence — structured scoring,
fit analysis, and interview preparation in one pass.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# API KEY CHECK
# ============================================================

if not GROQ_API_KEY:

    st.error(
        "❌ GROQ_API_KEY not found. "
        "Add it to `.env` or Streamlit secrets."
    )

    st.code(
        "GROQ_API_KEY=your_groq_api_key",
        language="text"
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state["history"] = []

if "active_result_hash" not in st.session_state:
    st.session_state["active_result_hash"] = None


# ============================================================
# INPUT SECTION
# ============================================================

col_left, col_right = st.columns(
    2,
    gap="large"
)


# ============================================================
# RESUME INPUT
# ============================================================

candidates = []


with col_left:

    st.markdown(
        """
<div class="panel panel-indigo">
<span class="eyebrow eyebrow-indigo">
Input · Candidate
</span>
""",
        unsafe_allow_html=True
    )

    st.subheader("📋 Resume(s)")

    mode = st.radio(
        "Input method",
        [
            "📄 Upload PDF(s)",
            "✏️ Paste Text"
        ],
        horizontal=True
    )

    # --------------------------------------------------------
    # PDF MODE
    # --------------------------------------------------------

    if mode == "📄 Upload PDF(s)":

        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:

            load_errors = []

            for uploaded_file in uploaded_files:

                try:

                    file_bytes = uploaded_file.getvalue()

                    text = extract_pdf_text(
                        file_bytes
                    )

                    if not text.strip():

                        load_errors.append(
                            f"{uploaded_file.name} — "
                            "no extractable text. "
                            "This may be a scanned/image PDF."
                        )

                        continue

                    candidate_name = (
                        os.path.splitext(
                            uploaded_file.name
                        )[0]
                    )

                    candidates.append(
                        (
                            candidate_name,
                            text
                        )
                    )

                except Exception as e:

                    load_errors.append(
                        f"{uploaded_file.name} — "
                        f"failed to read PDF: {e}"
                    )

            if candidates:

                total_chars = sum(
                    len(text)
                    for _, text in candidates
                )

                st.success(
                    f"✅ {len(candidates)} resume(s) loaded "
                    f"— {total_chars:,} total characters"
                )

                with st.expander(
                    f"Preview ({len(candidates)} file(s))"
                ):

                    for name, text in candidates:

                        st.caption(
                            f"**{name}** — "
                            f"{len(text):,} characters"
                        )

                        preview = text[:1000]

                        if preview:
                            st.text(preview)

            for error in load_errors:

                st.warning(
                    f"⚠️ {error}"
                )

    # --------------------------------------------------------
    # TEXT MODE
    # --------------------------------------------------------

    else:

        pasted = st.text_area(
            "Resume text",
            height=300,
            placeholder=(
                "Paste the full resume here…"
            ),
            label_visibility="collapsed"
        ).strip()

        candidate_name = st.text_input(
            "Candidate name (optional)",
            placeholder="e.g. Jordan Lee"
        ).strip()

        if pasted:

            candidates.append(
                (
                    candidate_name or "Candidate",
                    pasted
                )
            )

            st.caption(
                f"{len(pasted):,} characters"
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# JOB DESCRIPTION
# ============================================================

with col_right:

    st.markdown(
        """
<div class="panel panel-teal">
<span class="eyebrow eyebrow-teal">
Input · Role
</span>
""",
        unsafe_allow_html=True
    )

    st.subheader(
        "💼 Job Description"
    )

    job_description = st.text_area(
        "Job description",
        height=350,
        placeholder=(
            "Paste the job description here…"
        ),
        label_visibility="collapsed"
    ).strip()

    if job_description:

        st.caption(
            f"{len(job_description):,} characters"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# ACTION BUTTONS
# ============================================================

st.divider()

run_col, clear_col = st.columns(
    [1, 1]
)


with run_col:

    if len(candidates) == 0:

        button_text = (
            "🔍 Analyze Resume"
        )

    elif len(candidates) == 1:

        button_text = (
            "🔍 Analyze Resume"
        )

    else:

        button_text = (
            f"🔍 Analyze {len(candidates)} Resumes"
        )

    analyze_clicked = st.button(
        button_text,
        type="primary",
        use_container_width=True
    )


with clear_col:

    clear_clicked = st.button(
        "🗑️ Clear history",
        use_container_width=True
    )

    if clear_clicked:

        st.session_state["history"] = []

        st.session_state[
            "active_result_hash"
        ] = None

        st.rerun()


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_clicked:

    if not candidates:

        st.warning(
            "⚠️ Please provide at least one resume "
            "(PDF or text)."
        )

    elif not job_description:

        st.warning(
            "⚠️ Please paste a job description."
        )

    else:

        progress_bar = st.progress(
            0,
            text="Starting analysis…"
        )

        successful_results = 0

        for index, (
            candidate_name,
            resume_text
        ) in enumerate(candidates):

            progress_value = (
                index / len(candidates)
            )

            progress_bar.progress(
                progress_value,
                text=(
                    f"Analyzing {candidate_name} "
                    f"({index + 1}/{len(candidates)})…"
                )
            )

            try:

                result = run_pipeline_cached(
                    "v2",
                    candidate_name,
                    resume_text,
                    job_description
                )

            except Exception as e:

                st.error(
                    f"❌ Failed to analyze "
                    f"{candidate_name}: {e}"
                )

                continue

            if not isinstance(result, dict):

                st.error(
                    f"❌ Invalid result returned "
                    f"for {candidate_name}."
                )

                continue

            match_score = safe_string(
                result.get(
                    "match_score",
                    ""
                )
            )

            recommendation = safe_string(
                result.get(
                    "recommendation",
                    ""
                )
            )

            candidate_hash = content_hash(
                candidate_name,
                resume_text,
                job_description
            )

            entry = {

                "candidate_name":
                    candidate_name,

                "timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M"
                    ),

                "match_pct":
                    pct_from_text(
                        match_score
                    ),

                "decision":
                    decision_from_text(
                        recommendation
                    ),

                "model_used":
                    result.get(
                        "model_used",
                        "unknown"
                    ),

                "result":
                    result,

                "hash":
                    candidate_hash,
            }

            # Remove duplicate analysis
            st.session_state["history"] = [
                h
                for h in st.session_state["history"]
                if h.get("hash") != candidate_hash
            ]

            st.session_state[
                "history"
            ].append(entry)

            successful_results += 1

        progress_bar.progress(
            1.0,
            text="Analysis complete."
        )

        time.sleep(0.3)

        progress_bar.empty()

        if successful_results > 0:

            st.session_state[
                "active_result_hash"
            ] = st.session_state[
                "history"
            ][-1]["hash"]

            st.rerun()


# ============================================================
# RESULTS
# ============================================================

history = st.session_state.get(
    "history",
    []
)


if history:

    st.success(
        f"✅ {len(history)} analysis result(s) available",
        icon="🎉"
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    if len(history) > 1:

        with st.expander(
            "📊 Candidate comparison",
            expanded=True
        ):

            comparison_bar_chart(
                history
            )

            comparison_data = []

            for h in history:

                comparison_data.append(
                    {
                        "Candidate":
                            h.get(
                                "candidate_name",
                                "Unknown"
                            ),

                        "Match %":
                            h.get(
                                "match_pct"
                            ),

                        "Decision":
                            h.get(
                                "decision",
                                "Unknown"
                            ),

                        "Analyzed":
                            h.get(
                                "timestamp",
                                ""
                            ),

                        "Model":
                            h.get(
                                "model_used",
                                "unknown"
                            ),
                    }
                )

            st.dataframe(
                comparison_data,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # CANDIDATE SELECTION
    # ========================================================

    history_hashes = [
        h.get("hash")
        for h in history
    ]

    active_hash = st.session_state.get(
        "active_result_hash"
    )

    if active_hash in history_hashes:

        default_index = (
            history_hashes.index(
                active_hash
            )
        )

    else:

        default_index = (
            len(history) - 1
        )

    candidate_labels = []

    for index, h in enumerate(history):

        name = h.get(
            "candidate_name",
            "Candidate"
        )

        match = h.get(
            "match_pct"
        )

        decision = h.get(
            "decision",
            "Unknown"
        )

        candidate_labels.append(
            f"{name} | "
            f"{match if match is not None else 'N/A'}% | "
            f"{decision}"
        )


    selected_label = st.selectbox(
        "View candidate",
        candidate_labels,
        index=default_index
    )


    selected_index = (
        candidate_labels.index(
            selected_label
        )
    )

    entry = history[selected_index]

    st.session_state[
        "active_result_hash"
    ] = entry.get("hash")


    result = entry.get(
        "result",
        {}
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        f"""
<span class="model-tag">
🧠 {entry.get("model_used", "unknown")}
· analyzed {entry.get("timestamp", "")}
</span>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # TABS
    # ========================================================

    tabs = st.tabs(
        [
            "👤 Parsed Resume",
            "📊 JD Analysis",
            "🎯 Match Score",
            "✅ Recommendation",
            "🎤 Interview Questions",
            "⬇️ Export",
        ]
    )


    # ========================================================
    # TAB 1: PARSED RESUME
    # ========================================================

    with tabs[0]:

        parsed_resume = result.get(
            "parsed_resume",
            empty_resume()
        )

        parsed_resume = normalize_resume(
            parsed_resume
        )

        c1, c2 = st.columns(2)


        with c1:

            st.markdown(
                f"**👤 Name:** "
                f"{parsed_resume.get('name', '—')}"
            )

            st.markdown(
                f"**📧 Email:** "
                f"{parsed_resume.get('email') or '—'}"
            )

            st.markdown(
                f"**📞 Phone:** "
                f"{parsed_resume.get('phone') or '—'}"
            )

            st.markdown(
                f"**🎓 Education:** "
                f"{parsed_resume.get('education') or '—'}"
            )

            st.markdown(
                f"**📅 Experience:** "
                f"{parsed_resume.get('experience_years', 0)} "
                f"year(s)"
            )


        with c2:

            skills = parsed_resume.get(
                "skills",
                []
            )

            st.markdown(
                f"**🛠️ Skills ({len(skills)}):**"
            )

            if skills:

                chips = "".join(
                    f'<span class="skill-chip">'
                    f'{str(skill)}'
                    f'</span>'
                    for skill in skills
                )

                st.markdown(
                    chips,
                    unsafe_allow_html=True
                )

            else:

                st.caption(
                    "No skills extracted."
                )


            certifications = (
                parsed_resume.get(
                    "certifications",
                    []
                )
            )

            if certifications:

                st.markdown(
                    "**🏅 Certifications:**"
                )

                for certification in certifications:

                    st.markdown(
                        f"- {certification}"
                    )


        projects = parsed_resume.get(
            "projects",
            []
        )

        if projects:

            st.markdown(
                "**🚀 Projects:**"
            )

            for project in projects:

                st.markdown(
                    f"- {project}"
                )


        with st.expander(
            "Raw JSON"
        ):

            st.json(
                parsed_resume
            )


    # ========================================================
    # TAB 2: JD ANALYSIS
    # ========================================================

    with tabs[1]:

        jd_analysis = result.get(
            "jd_analysis",
            "JD analysis unavailable."
        )

        st.markdown(
            safe_string(jd_analysis)
        )


    # ========================================================
    # TAB 3: MATCH SCORE
    # ========================================================

    with tabs[2]:

        pct = entry.get(
            "match_pct"
        )

        if pct is not None:

            match_gauge(
                pct,
                key=f"gauge_{entry.get('hash')}"
            )

        else:

            st.warning(
                "Match percentage could not "
                "be extracted from the model response."
            )

        match_score = result.get(
            "match_score",
            "Match score unavailable."
        )

        st.markdown(
            safe_string(match_score)
        )


    # ========================================================
    # TAB 4: RECOMMENDATION
    # ========================================================

    with tabs[3]:

        decision = entry.get(
            "decision",
            "Unknown"
        )

        decision_lower = (
            safe_string(decision)
            .lower()
        )

        if decision_lower == "hire":

            badge_class = "badge-hire"

        elif decision_lower == "reject":

            badge_class = "badge-reject"

        elif decision_lower == "consider":

            badge_class = "badge-consider"

        else:

            badge_class = "badge-unknown"


        st.markdown(
            f"""
<span class="{badge_class}">
{decision.upper()}
</span>
""",
            unsafe_allow_html=True
        )

        st.markdown("")

        recommendation = result.get(
            "recommendation",
            "Recommendation unavailable."
        )

        st.markdown(
            safe_string(recommendation)
        )


    # ========================================================
    # TAB 5: INTERVIEW QUESTIONS
    # ========================================================

    with tabs[4]:

        questions = result.get(
            "interview_questions",
            "Interview questions unavailable."
        )

        st.markdown(
            safe_string(questions)
        )


    # ========================================================
    # TAB 6: EXPORT
    # ========================================================

    with tabs[5]:

        export_data = {

            "candidate_name":
                entry.get(
                    "candidate_name",
                    "Candidate"
                ),

            "analyzed_at":
                entry.get(
                    "timestamp",
                    ""
                ),

            "match_percentage":
                entry.get(
                    "match_pct"
                ),

            "decision":
                entry.get(
                    "decision",
                    "Unknown"
                ),

            "parsed_resume":
                result.get(
                    "parsed_resume",
                    {}
                ),

            "jd_analysis":
                result.get(
                    "jd_analysis",
                    ""
                ),

            "match_score":
                result.get(
                    "match_score",
                    ""
                ),

            "recommendation":
                result.get(
                    "recommendation",
                    ""
                ),

            "interview_questions":
                result.get(
                    "interview_questions",
                    ""
                ),

            "model_used":
                entry.get(
                    "model_used",
                    "unknown"
                ),
        }


        # ----------------------------------------------------
        # JSON EXPORT
        # ----------------------------------------------------

        json_data = json.dumps(
            export_data,
            indent=2,
            ensure_ascii=False
        )

        safe_filename = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            safe_string(
                entry.get(
                    "candidate_name",
                    "candidate"
                )
            )
        ).strip("_") or "candidate"


        st.download_button(
            "⬇️ Download candidate report (JSON)",

            data=json_data,

            file_name=(
                f"{safe_filename}_report.json"
            ),

            mime="application/json",

            use_container_width=True
        )


        # ----------------------------------------------------
        # MARKDOWN EXPORT
        # ----------------------------------------------------

        parsed = normalize_resume(
            result.get(
                "parsed_resume",
                {}
            )
        )

        markdown_lines = [

            f"# AI Resume Screening Report — "
            f"{entry.get('candidate_name', 'Candidate')}",

            "",

            f"**Analyzed:** "
            f"{entry.get('timestamp', '')}",

            f"**Model:** "
            f"{entry.get('model_used', 'unknown')}",

            "",

            "## Parsed Resume",

            "",

            f"**Name:** "
            f"{parsed.get('name', '—')}",

            f"**Email:** "
            f"{parsed.get('email') or '—'}",

            f"**Phone:** "
            f"{parsed.get('phone') or '—'}",

            f"**Education:** "
            f"{parsed.get('education') or '—'}",

            f"**Experience:** "
            f"{parsed.get('experience_years', 0)} years",

            "",

            "### Skills",

            ", ".join(
                parsed.get(
                    "skills",
                    []
                )
            ) or "None",

            "",

            "### Certifications",

            ", ".join(
                parsed.get(
                    "certifications",
                    []
                )
            ) or "None",

            "",

            "### Projects",

            ", ".join(
                parsed.get(
                    "projects",
                    []
                )
            ) or "None",

            "",

            "## JD Analysis",

            "",

            safe_string(
                result.get(
                    "jd_analysis",
                    ""
                )
            ),

            "",

            "## Match Score",

            "",

            safe_string(
                result.get(
                    "match_score",
                    ""
                )
            ),

            "",

            "## Recommendation",

            "",

            safe_string(
                result.get(
                    "recommendation",
                    ""
                )
            ),

            "",

            "## Interview Questions",

            "",

            safe_string(
                result.get(
                    "interview_questions",
                    ""
                )
            ),
        ]


        markdown_data = "\n".join(
            markdown_lines
        )


        st.download_button(
            "⬇️ Download candidate report (Markdown)",

            data=markdown_data,

            file_name=(
                f"{safe_filename}_report.md"
            ),

            mime="text/markdown",

            use_container_width=True
        )


        # ----------------------------------------------------
        # ALL CANDIDATES EXPORT
        # ----------------------------------------------------

        if len(history) > 1:

            st.divider()

            all_candidates = []

            for h in history:

                h_result = h.get(
                    "result",
                    {}
                )

                all_candidates.append(
                    {
                        "candidate_name":
                            h.get(
                                "candidate_name",
                                "Candidate"
                            ),

                        "match_percentage":
                            h.get(
                                "match_pct"
                            ),

                        "decision":
                            h.get(
                                "decision",
                                "Unknown"
                            ),

                        "timestamp":
                            h.get(
                                "timestamp",
                                ""
                            ),

                        "model_used":
                            h.get(
                                "model_used",
                                "unknown"
                            ),

                        "result":
                            h_result,
                    }
                )


            all_json = json.dumps(
                all_candidates,
                indent=2,
                ensure_ascii=False
            )


            st.download_button(
                "⬇️ Download all candidates (JSON)",

                data=all_json,

                file_name=(
                    "all_candidates_report.json"
                ),

                mime="application/json",

                use_container_width=True
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload or paste resume(s) and a job description, "
        "then click Analyze."
    )
