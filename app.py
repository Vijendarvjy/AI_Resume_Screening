import os
import re
import json
import time
import tempfile
import hashlib
import copy
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Tuple

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CONFIG = {
    # Tried in order; on rate-limit/errors we fall through to the next one.
    "fallback_models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
    "max_tokens": 1024,
    "resume_char_limit": 8000,
    "jd_char_limit": 6000,
    "max_retries_per_model": 2,
    "retry_base_delay": 2,  # seconds, doubles each retry
}

# ─────────────────────────────────────────
# ENV / SECRETS
# ─────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = None
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─────────────────────────────────────────
# LLM CLIENTS (cached per model name, so switching models doesn't
# require an app restart)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    if not GROQ_API_KEY:
        return None
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model_name,
        max_tokens=CONFIG["max_tokens"],
    )

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
class ResumeState(TypedDict):
    candidate_name:      str
    resume_text:         str
    job_description:     str
    parsed_resume:        dict
    jd_analysis:         str
    match_score:         str
    recommendation:      str
    interview_questions: str
    model_used:          str

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def trim(text: str, limit: int) -> str:
    """Trim to a character limit, preferring to cut on a sentence boundary
    so we don't waste tokens on a half-sentence and don't cut mid-thought."""
    s = str(text)
    if len(s) <= limit:
        return s
    truncated = s[:limit]
    last_period = truncated.rfind(". ")
    if last_period > limit * 0.8:
        truncated = truncated[: last_period + 1]
    st.toast(f"✂️ Trimmed input to ~{len(truncated):,} chars for LLM context.", icon="✂️")
    return truncated

def safe_invoke(prompt: str, fallback: str = "Unavailable") -> Tuple[str, Optional[str]]:
    """Invoke the LLM with retry + exponential backoff on rate limits,
    and fall through to the next configured model if one is exhausted."""
    last_err = None
    for model_name in CONFIG["fallback_models"]:
        llm = get_llm(model_name)
        if llm is None:
            continue
        for attempt in range(CONFIG["max_retries_per_model"]):
            try:
                result = llm.invoke(prompt).content
                return result, model_name
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_rate_limit = "429" in msg or "rate_limit" in msg or "rate limit" in msg
                if is_rate_limit:
                    if attempt < CONFIG["max_retries_per_model"] - 1:
                        wait = CONFIG["retry_base_delay"] * (2 ** attempt)
                        time.sleep(wait)
                        continue
                    else:
                        # exhausted retries on this model — try the next one
                        break
                else:
                    # non-rate-limit error — no point retrying same model
                    break
    return f"{fallback}: {last_err}", None

def extract_json(raw: str) -> dict:
    """Robustly extract JSON even when the model wraps it in markdown fences."""
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}

EMPTY_RESUME = {
    "name": "Unknown", "email": "", "phone": "",
    "skills": [], "experience_years": 0,
    "education": "", "certifications": [], "projects": [],
}

def empty_resume() -> dict:
    """Fresh copy of the fallback resume shape. Must be a deep copy — EMPTY_RESUME's
    list fields (skills/certifications/projects) would otherwise be shared mutable
    state across every candidate that hits the parse-failure fallback."""
    return copy.deepcopy(EMPTY_RESUME)

def content_hash(candidate_name: str, resume_text: str, job_description: str) -> str:
    return hashlib.sha256(
        (candidate_name + "||" + resume_text + "||" + job_description).encode("utf-8")
    ).hexdigest()[:16]

def pct_from_text(text: str) -> Optional[int]:
    """Pull the match percentage. Anchor to the '**Match Percentage:**' label first
    so an unrelated number/percent mentioned earlier in the text can't be picked up
    by mistake; fall back to the first bare percentage only if the label is missing."""
    text = text or ""
    m = re.search(r"\*\*Match Percentage:\*\*\s*(\d{1,3})\s*%", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,3})\s*%", text)
    return int(m.group(1)) if m else None

def decision_from_text(text: str) -> str:
    m = re.search(r"\*\*Decision:\*\*\s*(\w+)", text or "", re.IGNORECASE)
    return m.group(1).strip().title() if m else "Unknown"

# ─────────────────────────────────────────
# CACHED PIPELINE RUN (keyed on resume+JD content, so re-running the
# exact same pair doesn't burn tokens twice)
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def run_pipeline_cached(_graph_placeholder: str, candidate_name: str, resume_text: str, job_description: str) -> dict:
    graph = make_graph()
    return graph.invoke({
        "candidate_name": candidate_name,
        "resume_text": resume_text,
        "job_description": job_description,
    })

# ─────────────────────────────────────────
# PDF TEXT EXTRACTION (cached by file bytes so re-runs/reruns don't
# re-parse the same PDF over and over; also isolates failures per file)
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
        return "\n".join(p.page_content for p in pages)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

# ─────────────────────────────────────────
# LANGGRAPH NODES
# ─────────────────────────────────────────
def make_graph():

    def parse_resume(state):
        prompt = f"""Extract structured information from this resume.

Return ONLY a valid JSON object with these exact fields — no explanation, no markdown:
{{
  "name": "string",
  "email": "string",
  "phone": "string",
  "skills": ["list", "of", "skills"],
  "experience_years": 0,
  "education": "string",
  "certifications": ["list"],
  "projects": ["list of project titles"]
}}

Resume:
{trim(state["resume_text"], CONFIG["resume_char_limit"])}"""

        raw, model_used = safe_invoke(prompt, "Parse failed")
        parsed = extract_json(raw)
        if not parsed:
            parsed = empty_resume()
            if state.get("candidate_name"):
                parsed["name"] = state["candidate_name"]
        return {"parsed_resume": parsed, "model_used": model_used or state.get("model_used", "")}

    def analyze_jd(state):
        prompt = f"""Compare the resume and job description below. Be concise and structured.

Resume (summary):
{trim(state["resume_text"], 3000)}

Job Description:
{trim(state["job_description"], CONFIG["jd_char_limit"])}

Respond with exactly three sections:
**Matching Skills:** (bullet list)
**Missing Skills:** (bullet list)
**Fit Score:** X/100 — one sentence reason"""

        raw, model_used = safe_invoke(prompt, "JD analysis failed")
        return {"jd_analysis": raw, "model_used": model_used or state.get("model_used", "")}

    def calculate_match(state):
        prompt = f"""Based on the resume analysis below, produce a concise match report.

Parsed Resume:
{trim(str(state["parsed_resume"]), 1000)}

JD Analysis:
{trim(str(state["jd_analysis"]), 1500)}

Respond with exactly:
**Match Percentage:** X%
**Top 3 Strengths:**
1. ...
2. ...
3. ...
**Top 3 Gaps:**
1. ...
2. ...
3. ..."""

        raw, model_used = safe_invoke(prompt, "Match scoring failed")
        return {"match_score": raw, "model_used": model_used or state.get("model_used", "")}

    def generate_recommendation(state):
        prompt = f"""Based on this match report, give a clear hiring recommendation.

{trim(str(state["match_score"]), 1000)}

Respond with exactly:
**Decision:** Hire / Reject / Consider
**Confidence:** High / Medium / Low
**Reasoning:** (one paragraph)
**Suggested Next Step:** (one sentence)"""

        raw, model_used = safe_invoke(prompt, "Recommendation failed")
        return {"recommendation": raw, "model_used": model_used or state.get("model_used", "")}

    def generate_questions(state):
        skills = state["parsed_resume"].get("skills", [])
        skills_str = ", ".join(skills[:15]) if skills else "general software engineering"
        exp = state["parsed_resume"].get("experience_years", 0)

        prompt = f"""Generate 10 concise technical interview questions for a candidate with:
- Skills: {skills_str}
- Experience: ~{exp} years

Number each question (1–10). Mix difficulty levels. Be specific and practical. No preamble."""

        raw, model_used = safe_invoke(prompt, "Question generation failed")
        return {"interview_questions": raw, "model_used": model_used or state.get("model_used", "")}

    wf = StateGraph(ResumeState)
    for name, fn in [
        ("parse_resume", parse_resume),
        ("analyze_jd", analyze_jd),
        ("calculate_match", calculate_match),
        ("generate_recommendation", generate_recommendation),
        ("generate_questions", generate_questions),
    ]:
        wf.add_node(name, fn)

    wf.set_entry_point("parse_resume")
    wf.add_edge("parse_resume", "analyze_jd")
    wf.add_edge("analyze_jd", "calculate_match")
    wf.add_edge("calculate_match", "generate_recommendation")
    wf.add_edge("generate_recommendation", "generate_questions")
    wf.add_edge("generate_questions", END)

    return wf.compile()

# ─────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────
def match_gauge(pct: int, key: str):
    color = "#15803D" if pct >= 70 else ("#B45309" if pct >= 45 else "#B91C1C")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"family": "IBM Plex Mono, monospace", "color": "#0B1220"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#E4E7EC"},
            "bar": {"color": color},
            "bgcolor": "#FFFFFF",
            "bordercolor": "#E4E7EC",
            "steps": [
                {"range": [0, 45], "color": "#FEE2E2"},
                {"range": [45, 70], "color": "#FEF3C7"},
                {"range": [70, 100], "color": "#DCFCE7"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="#FFFFFF", font={"family": "Inter, sans-serif", "color": "#0B1220"},
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

def comparison_bar_chart(history: list):
    names = [h["candidate_name"] for h in history]
    pcts = [h["match_pct"] if h["match_pct"] is not None else 0 for h in history]
    colors = ["#15803D" if p >= 70 else ("#B45309" if p >= 45 else "#B91C1C") for p in pcts]
    fig = go.Figure(go.Bar(x=names, y=pcts, marker_color=colors, text=[f"{p}%" for p in pcts], textposition="auto"))
    fig.update_layout(
        height=340, margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Match %", range=[0, 100], gridcolor="#E4E7EC"),
        xaxis=dict(linecolor="#E4E7EC"),
        title=dict(text="Candidate comparison", font=dict(family="Sora, sans-serif", size=16, color="#0B1220")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font={"family": "Inter, sans-serif", "color": "#0B1220"},
    )
    st.plotly_chart(fig, use_container_width=True, key="comparison_chart")

# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink:        #0B1220;
    --ink-soft:   #475467;
    --bg:         #FFFFFF;
    --surface:    #F7F8FB;
    --border:     #E4E7EC;
    --indigo:     #4F46E5;
    --indigo-soft:#EEF2FF;
    --teal:       #0E7490;
    --teal-soft:  #ECFEFF;
    --green:      #15803D;
    --green-soft: #DCFCE7;
    --red:        #B91C1C;
    --red-soft:   #FEE2E2;
    --amber:      #B45309;
    --amber-soft: #FEF3C7;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }

h1, h2, h3, .stTitle, [data-testid="stMarkdownContainer"] h1 {
    font-family: 'Sora', sans-serif; font-weight: 700; color: var(--ink); letter-spacing: -0.01em;
}
[data-testid="stMarkdownContainer"] h3 { font-weight: 600; }

.app-subtitle {
    font-family: 'Inter', sans-serif; color: var(--ink-soft); font-size: 0.95rem;
    margin-top: -8px; margin-bottom: 1.6rem;
}

/* Section wayfinding: small colored eyebrow label above each panel heading */
.eyebrow {
    display: block; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 6px;
}
.eyebrow-indigo { color: var(--indigo); }
.eyebrow-teal   { color: var(--teal); }

/* White panel cards with a colored top rail — the input areas of the workflow */
.panel {
    background: var(--bg); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px 22px 6px 22px; margin-bottom: 18px;
}
.panel-indigo { border-top: 3px solid var(--indigo); }
.panel-teal   { border-top: 3px solid var(--teal); }

.metric-card {
    background: var(--surface); border-radius: 10px; padding: 16px 20px;
    border-left: 4px solid var(--indigo); margin-bottom: 12px;
}

/* Status badges: soft tint background, solid text — consistent across the app */
.badge-hire, .badge-reject, .badge-consider {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em;
    padding: 5px 14px; border-radius: 20px; text-transform: uppercase;
}
.badge-hire     { background: var(--green-soft); color: var(--green); }
.badge-reject   { background: var(--red-soft);   color: var(--red); }
.badge-consider { background: var(--amber-soft); color: var(--amber); }

.model-tag {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
    font-weight: 500; color: var(--teal); background: var(--teal-soft);
    padding: 5px 14px; border-radius: 20px; margin-bottom: 12px; border: 1px solid #CFFAFE;
}

.skill-chip {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: var(--indigo); background: var(--indigo-soft); border: 1px solid #E0E7FF;
    padding: 3px 10px; border-radius: 6px; margin: 0 6px 6px 0;
}

/* Tabs restyled as a segmented control */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: var(--surface); padding: 4px; border-radius: 10px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    height: 42px; padding: 0 18px; border-radius: 8px; font-weight: 600;
    color: var(--ink-soft);
}
.stTabs [aria-selected="true"] {
    background: var(--bg) !important; color: var(--indigo) !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}

.block-container { padding-top: 2rem; }
.stButton > button[kind="primary"] { background: var(--indigo); border-color: var(--indigo); }
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Resume Screening System")
st.markdown('<div class="app-subtitle">Enterprise resume intelligence — structured scoring, fit analysis, and interview prep in one pass.</div>', unsafe_allow_html=True)

if not GROQ_API_KEY:
    st.error("❌ `GROQ_API_KEY` not found. Add it to `.env` or Streamlit secrets.")
    st.stop()

# ── Session state ────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of {candidate_name, timestamp, match_pct, decision, result, model_used}
if "active_result_idx" not in st.session_state:
    st.session_state["active_result_idx"] = None

# ── Input ─────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

candidates = []  # list of (name, resume_text)

with col_left:
    st.markdown(
        '<div class="panel panel-indigo"><span class="eyebrow eyebrow-indigo">Input · Candidate</span>',
        unsafe_allow_html=True,
    )
    st.subheader("📋 Resume(s)")
    mode = st.radio("Input method", ["📄 Upload PDF(s)", "✏️ Paste Text"], horizontal=True)

    if mode == "📄 Upload PDF(s)":
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            load_errors = []
            for uf in uploaded_files:
                try:
                    text = extract_pdf_text(uf.getvalue())
                    if not text.strip():
                        load_errors.append(f"{uf.name} — no extractable text (likely a scanned image PDF)")
                        continue
                    candidates.append((uf.name.rsplit(".", 1)[0], text))
                except Exception as e:
                    load_errors.append(f"{uf.name} — failed to read ({e})")

            if candidates:
                st.success(f"✅ {len(candidates)} resume(s) loaded — {sum(len(t) for _, t in candidates):,} total chars")
                with st.expander(f"Preview ({len(candidates)} file(s))"):
                    for name, text in candidates:
                        st.caption(f"**{name}** — {len(text):,} chars")
            for err in load_errors:
                st.warning(f"⚠️ {err}")
    else:
        pasted = st.text_area(
            "Resume text", height=300,
            placeholder="Paste the full resume here…",
            label_visibility="collapsed",
        ).strip()
        candidate_name = st.text_input("Candidate name (optional)", placeholder="e.g. Jordan Lee")
        if pasted:
            candidates.append((candidate_name or "Candidate", pasted))
            st.caption(f"{len(pasted):,} characters")

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown(
        '<div class="panel panel-teal"><span class="eyebrow eyebrow-teal">Input · Role</span>',
        unsafe_allow_html=True,
    )
    st.subheader("💼 Job Description")
    job_description = st.text_area(
        "Job description", height=350,
        placeholder="Paste the job description here…",
        label_visibility="collapsed",
    ).strip()
    if job_description:
        st.caption(f"{len(job_description):,} characters")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
run_col, clear_col = st.columns([1, 1])
with run_col:
    analyze_clicked = st.button(
        f"🔍 Analyze {len(candidates)} Resume(s)" if len(candidates) > 1 else "🔍 Analyze Resume",
        type="primary", use_container_width=True,
    )
with clear_col:
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["active_result_idx"] = None
        st.rerun()

if analyze_clicked:
    if not candidates:
        st.warning("⚠️ Please provide at least one resume (PDF or text).")
    elif not job_description:
        st.warning("⚠️ Please paste a job description.")
    else:
        progress_bar = st.progress(0, text="Starting analysis…")
        for i, (name, resume_text) in enumerate(candidates):
            progress_bar.progress(
                i / len(candidates),
                text=f"Analyzing {name} ({i + 1}/{len(candidates)})…",
            )
            try:
                result = run_pipeline_cached("v1", name, resume_text, job_description)
            except Exception as e:
                st.error(f"❌ Failed to analyze {name}: {e}")
                continue

            entry = {
                "candidate_name": name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "match_pct": pct_from_text(result.get("match_score", "")),
                "decision": decision_from_text(result.get("recommendation", "")),
                "model_used": result.get("model_used", "unknown"),
                "result": result,
                "hash": content_hash(name, resume_text, job_description),
            }
            # avoid duplicate entries for the same candidate+resume+JD combo
            st.session_state["history"] = [
                h for h in st.session_state["history"] if h["hash"] != entry["hash"]
            ] + [entry]

        progress_bar.progress(1.0, text="Done")
        time.sleep(0.3)
        progress_bar.empty()
        st.session_state["active_result_idx"] = len(st.session_state["history"]) - 1
        st.rerun()

# ── Results ──────────────────────────────
history = st.session_state["history"]

if history:
    st.success(f"✅ {len(history)} analysis result(s) available", icon="🎉")

    if len(history) > 1:
        with st.expander("📊 Candidate comparison", expanded=True):
            comparison_bar_chart(history)
            st.dataframe(
                [
                    {
                        "Candidate": h["candidate_name"],
                        "Match %": h["match_pct"],
                        "Decision": h["decision"],
                        "Analyzed": h["timestamp"],
                        "Model": h["model_used"],
                    }
                    for h in history
                ],
                use_container_width=True,
                hide_index=True,
            )

    names = [h["candidate_name"] for h in history]
    default_idx = st.session_state["active_result_idx"] if st.session_state["active_result_idx"] is not None else len(history) - 1
    default_idx = max(0, min(default_idx, len(names) - 1))
    selected_name = st.selectbox("View candidate", names, index=default_idx)
    entry = next(h for h in history if h["candidate_name"] == selected_name)
    result = entry["result"]

    st.markdown(
        f"<span class='model-tag'>🧠 {entry['model_used']} · analyzed {entry['timestamp']}</span>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "👤 Parsed Resume", "📊 JD Analysis", "🎯 Match Score",
        "✅ Recommendation", "🎤 Interview Questions", "⬇️ Export",
    ])

    with tabs[0]:
        p = result["parsed_resume"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**👤 Name:** {p.get('name', '—')}")
            st.markdown(f"**📧 Email:** {p.get('email', '—')}")
            st.markdown(f"**📞 Phone:** {p.get('phone', '—')}")
            st.markdown(f"**🎓 Education:** {p.get('education', '—')}")
            st.markdown(f"**📅 Experience:** {p.get('experience_years', '—')} year(s)")
        with c2:
            skills = p.get("skills", [])
            st.markdown(f"**🛠️ Skills ({len(skills)}):**")
            if skills:
                chips = "".join(f'<span class="skill-chip">{s}</span>' for s in skills)
                st.markdown(chips, unsafe_allow_html=True)
            certs = p.get("certifications", [])
            if certs:
                st.markdown("**🏅 Certifications:**")
                for c in certs:
                    st.markdown(f"- {c}")
        projs = p.get("projects", [])
        if projs:
            st.markdown("**🚀 Projects:**")
            for pr in projs:
                st.markdown(f"- {pr}")
        with st.expander("Raw JSON"):
            st.json(p)

    with tabs[1]:
        st.markdown(result["jd_analysis"])

    with tabs[2]:
        pct = entry["match_pct"]
        if pct is not None:
            match_gauge(pct, key=f"gauge_{entry['hash']}")
        st.markdown(result["match_score"])

    with tabs[3]:
        d = entry["decision"].lower()
        badge_class = {"hire": "badge-hire", "reject": "badge-reject"}.get(d, "badge-consider")
        st.markdown(f"<span class='{badge_class}'>{entry['decision'].upper()}</span>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown(result["recommendation"])

    with tabs[4]:
        st.markdown(result["interview_questions"])

    with tabs[5]:
        export = {
            "candidate_name": entry["candidate_name"],
            "parsed_resume": result["parsed_resume"],
            "jd_analysis": result["jd_analysis"],
            "match_score": result["match_score"],
            "recommendation": result["recommendation"],
            "interview_questions": result["interview_questions"],
            "model_used": entry["model_used"],
        }
        st.download_button(
            "⬇️ Download this candidate's report (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"{entry['candidate_name']}_report.json",
            mime="application/json", use_container_width=True,
        )

        md_lines = [
            f"# AI Resume Screening Report — {entry['candidate_name']}\n",
            "## Parsed Resume\n",
            f"**Name:** {result['parsed_resume'].get('name', '—')}  ",
            f"**Email:** {result['parsed_resume'].get('email', '—')}  ",
            f"**Skills:** {', '.join(result['parsed_resume'].get('skills', []))}  \n",
            "## JD Analysis\n", result["jd_analysis"], "\n",
            "## Match Score\n", result["match_score"], "\n",
            "## Recommendation\n", result["recommendation"], "\n",
            "## Interview Questions\n", result["interview_questions"],
        ]
        st.download_button(
            "⬇️ Download this candidate's report (Markdown)",
            data="\n".join(md_lines),
            file_name=f"{entry['candidate_name']}_report.md",
            mime="text/markdown", use_container_width=True,
        )

        if len(history) > 1:
            st.divider()
            all_export = [
                {
                    "candidate_name": h["candidate_name"],
                    "match_pct": h["match_pct"],
                    "decision": h["decision"],
                    "timestamp": h["timestamp"],
                    "result": h["result"],
                }
                for h in history
            ]
            st.download_button(
                "⬇️ Download all candidates (JSON)",
                data=json.dumps(all_export, indent=2),
                file_name="all_candidates_report.json",
                mime="application/json", use_container_width=True,
            )
else:
    st.info("Upload or paste resume(s) and a job description, then click Analyze.")
