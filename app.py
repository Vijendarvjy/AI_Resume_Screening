import os
import re
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
from typing import TypedDict

# ─────────────────────────────────────────
# CONFIG  (edit here, not buried in code)
# ─────────────────────────────────────────
CONFIG = {
    "model":            "meta-llama/llama-4-scout-17b-16e-instruct",
    "max_tokens":       1024,
    "resume_char_limit": 6000,
    "jd_char_limit":    2000,
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
# LLM  (cached so it isn't rebuilt on every rerun)
# ─────────────────────────────────────────
@st.cache_resource
def get_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=CONFIG["model"],
        max_tokens=CONFIG["max_tokens"],
    )

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
class ResumeState(TypedDict):
    resume_text:        str
    job_description:    str
    parsed_resume:      dict
    jd_analysis:        str
    match_score:        str
    recommendation:     str
    interview_questions: str

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def trim(text: str, limit: int) -> str:
    """Trim to character limit; warns if truncation occurs."""
    s = str(text)
    if len(s) > limit:
        st.toast(f"⚠️ Input trimmed to {limit} chars for LLM context.", icon="✂️")
    return s[:limit]

def safe_invoke(llm, prompt: str, fallback: str = "Unavailable") -> str:
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"{fallback}: {e}"

def extract_json(raw: str) -> dict:
    """Robustly extract JSON even when the model wraps it in markdown fences."""
    # strip ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    # try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # find first {...} block
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

# ─────────────────────────────────────────
# LANGGRAPH NODES
# ─────────────────────────────────────────
def make_graph(llm):

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

        raw = safe_invoke(llm, prompt, "Parse failed")
        parsed = extract_json(raw)
        if not parsed:
            st.warning("⚠️ Could not parse resume JSON — using defaults.")
            parsed = EMPTY_RESUME.copy()
        return {"parsed_resume": parsed}

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

        return {"jd_analysis": safe_invoke(llm, prompt, "JD analysis failed")}

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

        return {"match_score": safe_invoke(llm, prompt, "Match scoring failed")}

    def generate_recommendation(state):
        prompt = f"""Based on this match report, give a clear hiring recommendation.

{trim(str(state["match_score"]), 1000)}

Respond with exactly:
**Decision:** Hire / Reject / Consider
**Confidence:** High / Medium / Low
**Reasoning:** (one paragraph)
**Suggested Next Step:** (one sentence)"""

        return {"recommendation": safe_invoke(llm, prompt, "Recommendation failed")}

    def generate_questions(state):
        skills = state["parsed_resume"].get("skills", [])
        skills_str = ", ".join(skills[:15]) if skills else "general software engineering"
        exp = state["parsed_resume"].get("experience_years", 0)

        prompt = f"""Generate 10 concise technical interview questions for a candidate with:
- Skills: {skills_str}
- Experience: ~{exp} years

Number each question (1–10). Mix difficulty levels. Be specific and practical. No preamble."""

        return {"interview_questions": safe_invoke(llm, prompt, "Question generation failed")}

    # Build graph
    wf = StateGraph(ResumeState)
    for name, fn in [
        ("parse_resume",          parse_resume),
        ("analyze_jd",            analyze_jd),
        ("calculate_match",       calculate_match),
        ("generate_recommendation", generate_recommendation),
        ("generate_questions",    generate_questions),
    ]:
        wf.add_node(name, fn)

    wf.set_entry_point("parse_resume")
    wf.add_edge("parse_resume",           "analyze_jd")
    wf.add_edge("analyze_jd",             "calculate_match")
    wf.add_edge("calculate_match",        "generate_recommendation")
    wf.add_edge("generate_recommendation","generate_questions")
    wf.add_edge("generate_questions",     END)

    return wf.compile()

# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #6366f1;
        margin-bottom: 12px;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700;
        color: #374151; margin-bottom: 4px;
    }
    .badge-hire    { background:#dcfce7; color:#166534; padding:4px 12px; border-radius:20px; font-weight:700; }
    .badge-reject  { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-weight:700; }
    .badge-consider{ background:#fef9c3; color:#854d0e; padding:4px 12px; border-radius:20px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────
st.title("🤖 AI Resume Screening System")
st.caption(f"Powered by `{CONFIG['model']}` via Groq")

if not GROQ_API_KEY:
    st.error("❌ `GROQ_API_KEY` not found. Add it to `.env` or Streamlit secrets.")
    st.stop()

llm = get_llm()

# ── Initialize session state ─────────────
for key in ["result", "resume_text", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_done" else False

# ── Input columns ────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("📋 Resume")
    mode = st.radio("Input method", ["📄 Upload PDF", "✏️ Paste Text"], horizontal=True)

    resume_text = ""
    if mode == "📄 Upload PDF":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            pages = PyPDFLoader(tmp_path).load()
            resume_text = "\n".join(p.page_content for p in pages)
            st.success(f"✅ {len(pages)} page(s) extracted — {len(resume_text):,} chars")
            with st.expander("Preview extracted text"):
                st.text(resume_text[:2000] + ("…" if len(resume_text) > 2000 else ""))
    else:
        resume_text = st.text_area(
            "Resume text",
            height=300,
            placeholder="Paste the full resume here…",
            label_visibility="collapsed",
        ).strip()
        if resume_text:
            st.caption(f"{len(resume_text):,} characters")

with col_right:
    st.subheader("💼 Job Description")
    job_description = st.text_area(
        "Job description",
        height=350,
        placeholder="Paste the job description here…",
        label_visibility="collapsed",
    ).strip()
    if job_description:
        st.caption(f"{len(job_description):,} characters")

# ── Analyze button ───────────────────────
st.divider()
run_col, _ = st.columns([1, 4])
with run_col:
    analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

if analyze_clicked:
    if not resume_text:
        st.warning("⚠️ Please provide a resume (PDF or text).")
    elif not job_description:
        st.warning("⚠️ Please paste a job description.")
    else:
        graph = make_graph(llm)
        steps = [
            ("🔎 Parsing resume…",           None),
            ("📊 Analyzing job description…", None),
            ("🎯 Calculating match score…",   None),
            ("✅ Generating recommendation…", None),
            ("🎤 Generating interview questions…", None),
        ]
        progress_bar  = st.progress(0, text="Starting analysis…")
        status_text   = st.empty()

        # Stream node completions via a simple callback approach
        completed = [0]
        total = len(steps)

        def on_step(step_name):
            completed[0] += 1
            label = {
                "parse_resume":           "✅ Resume parsed",
                "analyze_jd":             "✅ JD analyzed",
                "calculate_match":        "✅ Match scored",
                "generate_recommendation":"✅ Recommendation ready",
                "generate_questions":     "✅ Questions generated",
            }.get(step_name, step_name)
            progress_bar.progress(completed[0] / total, text=label)

        # Run graph — LangGraph doesn't expose per-node hooks in invoke(),
        # so we update progress after completion using stream()
        result_state = None
        for chunk in graph.stream({
            "resume_text":     resume_text,
            "job_description": job_description,
        }):
            for node_name in chunk:
                on_step(node_name)
            result_state = chunk

        # Merge full state
        full_result = graph.invoke({
            "resume_text":     resume_text,
            "job_description": job_description,
        })

        progress_bar.empty()
        status_text.empty()

        st.session_state["result"]        = full_result
        st.session_state["resume_text"]   = resume_text
        st.session_state["analysis_done"] = True
        st.rerun()

# ── Results ──────────────────────────────
if st.session_state["analysis_done"] and st.session_state["result"]:
    result = st.session_state["result"]
    st.success("✅ Analysis complete!", icon="🎉")

    tabs = st.tabs([
        "👤 Parsed Resume",
        "📊 JD Analysis",
        "🎯 Match Score",
        "✅ Recommendation",
        "🎤 Interview Questions",
        "⬇️ Export",
    ])

    # ── Tab 1: Parsed Resume ─────────────
    with tabs[0]:
        p = result["parsed_resume"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**👤 Name:** {p.get('name','—')}")
            st.markdown(f"**📧 Email:** {p.get('email','—')}")
            st.markdown(f"**📞 Phone:** {p.get('phone','—')}")
            st.markdown(f"**🎓 Education:** {p.get('education','—')}")
            st.markdown(f"**📅 Experience:** {p.get('experience_years', '—')} year(s)")
        with c2:
            skills = p.get("skills", [])
            st.markdown(f"**🛠️ Skills ({len(skills)}):**")
            if skills:
                st.markdown(" ".join(f"`{s}`" for s in skills))
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

    # ── Tab 2: JD Analysis ───────────────
    with tabs[1]:
        st.markdown(result["jd_analysis"])

    # ── Tab 3: Match Score ───────────────
    with tabs[2]:
        raw_match = result["match_score"]
        # Try to pull percentage for a big metric
        pct_match = re.search(r"(\d{1,3})\s*%", raw_match)
        if pct_match:
            pct = int(pct_match.group(1))
            color = "#22c55e" if pct >= 70 else ("#f59e0b" if pct >= 45 else "#ef4444")
            st.markdown(
                f"<div style='font-size:3rem;font-weight:800;color:{color};'>{pct}%</div>"
                f"<div style='color:#6b7280;margin-bottom:1rem;'>Match score</div>",
                unsafe_allow_html=True,
            )
        st.markdown(raw_match)

    # ── Tab 4: Recommendation ────────────
    with tabs[3]:
        rec = result["recommendation"]
        decision_match = re.search(r"\*\*Decision:\*\*\s*(\w+)", rec, re.IGNORECASE)
        if decision_match:
            d = decision_match.group(1).strip().lower()
            badge_class = {"hire": "badge-hire", "reject": "badge-reject"}.get(d, "badge-consider")
            st.markdown(
                f"<span class='{badge_class}'>{d.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")
        st.markdown(rec)

    # ── Tab 5: Interview Questions ────────
    with tabs[4]:
        qs = result["interview_questions"]
        st.markdown(qs)

    # ── Tab 6: Export ─────────────────────
    with tabs[5]:
        export = {
            "parsed_resume":     result["parsed_resume"],
            "jd_analysis":       result["jd_analysis"],
            "match_score":       result["match_score"],
            "recommendation":    result["recommendation"],
            "interview_questions": result["interview_questions"],
        }
        st.download_button(
            label="⬇️ Download full report (JSON)",
            data=json.dumps(export, indent=2),
            file_name="resume_screening_report.json",
            mime="application/json",
            use_container_width=True,
        )

        # Markdown report
        md_lines = [
            "# AI Resume Screening Report\n",
            "## Parsed Resume\n",
            f"**Name:** {result['parsed_resume'].get('name','—')}  ",
            f"**Email:** {result['parsed_resume'].get('email','—')}  ",
            f"**Skills:** {', '.join(result['parsed_resume'].get('skills',[]))}  \n",
            "## JD Analysis\n", result["jd_analysis"], "\n",
            "## Match Score\n",  result["match_score"],  "\n",
            "## Recommendation\n", result["recommendation"], "\n",
            "## Interview Questions\n", result["interview_questions"],
        ]
        st.download_button(
            label="⬇️ Download full report (Markdown)",
            data="\n".join(md_lines),
            file_name="resume_screening_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ── Clear results ────────────────────
    st.divider()
    if st.button("🔄 Clear & start over"):
        for key in ["result", "resume_text", "analysis_done"]:
            st.session_state[key] = None if key != "analysis_done" else False
        st.rerun()
