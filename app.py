import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# Load environment variables
load_dotenv()
# -------------------------------
# LOAD API KEY
# -------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()
# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# State Schema
class ResumeState(TypedDict):
    resume_text: str
    job_description: str
    parsed_resume: str
    jd_analysis: str
    match_score: str
    recommendation: str
    interview_questions: str

# Resume Parser Node
def parse_resume(state):
    prompt = f"""
    Extract the following from the resume:
    - Skills
    - Experience
    - Education
    - Projects

    Resume:
    {state['resume_text']}
    """
    response = llm.invoke(prompt)
    return {"parsed_resume": response.content}

# JD Analyzer Node
def analyze_jd(state):
    prompt = f"""
    Analyze this job description and extract:
    - Required Skills
    - Experience
    - Responsibilities

    Job Description:
    {state['job_description']}
    """
    response = llm.invoke(prompt)
    return {"jd_analysis": response.content}

# Match Score Node
def calculate_match(state):
    prompt = f"""
    Compare the resume and job description.

    Resume Analysis:
    {state['parsed_resume']}

    JD Analysis:
    {state['jd_analysis']}

    Provide:
    - Match Percentage
    - Missing Skills
    - Strengths
    """
    response = llm.invoke(prompt)
    return {"match_score": response.content}

# Recommendation Node
def generate_recommendation(state):
    prompt = f"""
    Based on the candidate evaluation, provide:
    - Hire / Reject / Consider
    - Reasoning
    """
    response = llm.invoke(prompt)
    return {"recommendation": response.content}

# Interview Questions Node
def generate_questions(state):
    prompt = f"""
    Generate 10 technical interview questions
    based on the candidate profile.
    """
    response = llm.invoke(prompt)
    return {"interview_questions": response.content}

# Build LangGraph
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

# Streamlit UI
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("🤖 AI Resume Screening System")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=300
)

if st.button("Analyze Resume"):
    if uploaded_file and job_description:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        resume_text = "\n".join([page.page_content for page in pages])

        result = graph.invoke({
            "resume_text": resume_text,
            "job_description": job_description
        })

        st.subheader("📄 Resume Analysis")
        st.write(result["parsed_resume"])

        st.subheader("🎯 Job Description Analysis")
        st.write(result["jd_analysis"])

        st.subheader("📊 Match Score")
        st.write(result["match_score"])

        st.subheader("✅ Recommendation")
        st.write(result["recommendation"])

        st.subheader("🎤 Interview Questions")
        st.write(result["interview_questions"])
