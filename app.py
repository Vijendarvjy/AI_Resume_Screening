import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# -------------------------------
# ENV SETUP
# -------------------------------
load_dotenv()

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment or secrets.")
    st.stop()

# -------------------------------
# LLM
# -------------------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    max_tokens=1024
)

# -------------------------------
# STATE
# -------------------------------
class ResumeState(TypedDict):
    resume_text: str
    job_description: str
    parsed_resume: dict
    jd_analysis: str
    match_score: str
    recommendation: str
    interview_questions: str

# -------------------------------
# HELPERS
# -------------------------------
def safe_trim(text: str, limit: int) -> str:
    return str(text)[:limit]

def safe_invoke(prompt: str, fallback: str = "Unavailable") -> str:
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"{fallback}: {str(e)}"

# -------------------------------
# NODE 1: PARSE RESUME
# -------------------------------
def parse_resume(state):
    resume_text = safe_trim(state["resume_text"], 6000)

    prompt = f"""Extract structured JSON from this resume.

Return ONLY a valid JSON object with these exact fields:
{{
  "name": "string",
  "email": "string",
  "phone": "string",
  "skills": ["list", "of", "skills"],
  "experience_years": 0,
  "education": "string",
  "certifications": ["list"],
  "projects": ["list"]
}}

Resume:
{resume_text}

Return ONLY the JSON. No explanation. No markdown."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return {"parsed_resume": json.loads(content.strip())}
    except Exception as e:
        st.warning(f"Resume parsing failed: {e}")
        return {"parsed_resume": {
            "name": "Unknown", "email": "", "phone": "",
            "skills": [], "experience_years": 0, "education": "",
            "certifications": [], "projects": []
        }}

# -------------------------------
# NODE 2: JD ANALYSIS
# -------------------------------
def analyze_jd(state):
    resume = safe_trim(state["resume_text"], 3000)
    jd = safe_trim(state["job_description"], 1500)

    prompt = f"""Compare this resume and job description. Be concise.

Resume:
{resume}

Job Description:
{jd}

List:
1. Matching skills
2. Missing skills
3. Overall fit score (0-100)"""

    return {"jd_analysis": safe_invoke(prompt, "JD Analysis failed")}

# -------------------------------
# NODE 3: MATCH SCORE
# -------------------------------
def calculate_match(state):
    jd_analysis = safe_trim(str(state["jd_analysis"]), 1500)
    parsed = safe_trim(str(state["parsed_resume"]), 1000)

    prompt = f"""Based on this analysis, provide a match report.

Parsed Resume Summary:
{parsed}

JD Analysis:
{jd_analysis}

Return:
- Match Percentage (e.g. 78%)
- Top 3 Strengths
- Top 3 Gaps"""

    return {"match_score": safe_invoke(prompt, "Match scoring failed")}

# -------------------------------
# NODE 4: RECOMMENDATION
# -------------------------------
def generate_recommendation(state):
    match = safe_trim(str(state["match_score"]), 1000)

    prompt = f"""Based on this match report, give a hiring recommendation.

{match}

Output:
- Decision: Hire / Reject / Consider
- One paragraph reason"""

    return {"recommendation": safe_invoke(prompt, "Recommendation failed")}

# -------------------------------
# NODE 5: INTERVIEW QUESTIONS
# -------------------------------
def generate_questions(state):
    skills = state["parsed_resume"].get("skills", [])
    skills_str = ", ".join(skills) if skills else "general software engineering"

    prompt = f"""Generate 10 concise technical interview questions for a candidate with these skills: {skills_str}.

Number each question. Be specific and practical."""

    return {"interview_questions": safe_invoke(prompt, "Question generation failed")}

# -------------------------------
# LANGGRAPH SETUP
# -------------------------------
workflow = StateGraph(ResumeState)

workflow.add_node("parse_resume", parse_resume)
workflow.add_node("analyze_jd", analyze_jd)
workflow.add_node("calculate_match", calculate_match)
workflow.add_node("generate_recommendation", generate_recommendation)
workflow.add_node("generate_questions", generate_questions)

workflow.set_entry_point("parse_resume")

workflow.add_edge("parse_resume", "analyze_jd")
workflow.add_edge("analyze_jd", "calculate_match")
workflow.add_edge("calculate_match", "generate_recommendation")
workflow.add_edge("generate_recommendation", "generate_questions")
workflow.add_edge("generate_questions", END)

graph = workflow.compile()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🤖 AI Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description", height=250)

if st.button("Analyze Resume"):
    if uploaded_file and job_description:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        resume_text = "\n".join([p.page_content for p in pages])

        with st.spinner("Analyzing resume..."):
            result = graph.invoke({
                "resume_text": resume_text,
                "job_description": job_description
            })

        st.subheader("📄 Parsed Resume")
        st.json(result["parsed_resume"])

        st.subheader("🎯 JD Analysis")
        st.write(result["jd_analysis"])

        st.subheader("📊 Match Score")
        st.write(result["match_score"])

        st.subheader("✅ Recommendation")
        st.write(result["recommendation"])

        st.subheader("🎤 Interview Questions")
        st.write(result["interview_questions"])
    else:
        st.warning("⚠️ Please upload a resume and enter a job description.")
