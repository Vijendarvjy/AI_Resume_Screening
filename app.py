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
    model="llama3-8b-8192",
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

# -------------------------------
# NODE 1: PARSE RESUME
# -------------------------------
def parse_resume(state):
    resume_text = safe_trim(state["resume_text"], 12000)

    prompt = f"""
Extract structured JSON from resume.

Fields:
- name
- email
- phone
- skills (list)
- experience_years
- education
- certifications (list)
- projects (list)

Resume:
{resume_text}

Return ONLY JSON.
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(content)

        return {"parsed_resume": parsed}

    except Exception as e:
        return {
            "parsed_resume": {
                "name": "Unknown",
                "email": "",
                "phone": "",
                "skills": [],
                "experience_years": 0,
                "education": "",
                "certifications": [],
                "projects": []
            }
        }

# -------------------------------
# NODE 2: JD ANALYSIS
# -------------------------------
def analyze_jd(state):
    resume = safe_trim(state["resume_text"], 4000)
    jd = safe_trim(state["job_description"], 2500)

    prompt = f"""
Compare Resume and Job Description.

Resume:
{resume}

Job Description:
{jd}

Return:
- Key Skills Match
- Missing Skills
- Score (0-100)
"""

    response = llm.invoke(prompt)

    return {"jd_analysis": response.content}

# -------------------------------
# NODE 3: MATCH SCORE
# -------------------------------
def calculate_match(state):
    prompt = f"""
Evaluate candidate fit.

Resume:
{state['parsed_resume']}

JD Analysis:
{state['jd_analysis']}

Return:
- Match Percentage
- Strengths
- Missing Skills
"""

    response = llm.invoke(prompt)

    return {"match_score": response.content}

# -------------------------------
# NODE 4: RECOMMENDATION
# -------------------------------
def generate_recommendation(state):
    prompt = f"""
Based on evaluation:

{state['match_score']}

Decide:
- Hire / Reject / Consider
- Reason
"""

    response = llm.invoke(prompt)

    return {"recommendation": response.content}

# -------------------------------
# NODE 5: INTERVIEW QUESTIONS
# -------------------------------
def generate_questions(state):
    prompt = f"""
Generate 10 technical interview questions
based on:

{state['parsed_resume']}
"""

    response = llm.invoke(prompt)

    return {"interview_questions": response.content}

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
