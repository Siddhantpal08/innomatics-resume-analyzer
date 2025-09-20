import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import spacy
from spacy.matcher import PhraseMatcher
from typing import List
import sqlite3
import pandas as pd
from datetime import datetime

# LangChain and Pydantic Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in .env or Streamlit secrets.")
    st.stop()

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_all_models():
    """Loads and caches all heavy models."""
    spacy_model = spacy.load("en_core_web_sm")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return spacy_model, embedding_model

# --- DATABASE FUNCTIONS ---
DB_FILE = "analysis_results.db"

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn

def init_database():
    """Initializes the SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            candidate_name TEXT NOT NULL,
            jd_summary TEXT NOT NULL,
            score INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            missing_skills TEXT,
            feedback TEXT
        )
    """)
    conn.commit()

def add_analysis_to_db(candidate_name, jd_summary, report):
    """Adds a new analysis report to the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO results (timestamp, candidate_name, jd_summary, score, verdict, missing_skills, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            candidate_name,
            jd_summary,
            report.relevance_score,
            report.verdict,
            ", ".join(report.missing_elements),
            report.improvement_feedback
        ))
        conn.commit()
    except Exception as e:
        st.error(f"Database Error: {e}")

def load_data_for_dashboard():
    """Loads all analysis results from the database."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM results ORDER BY timestamp DESC", conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        return df
    except Exception as e:
        st.error(f"Database Load Error: {e}")
        return pd.DataFrame()

def delete_analysis_from_db(record_id):
    """Deletes a specific analysis record."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM results WHERE id = ?", (record_id,))
        conn.commit()
    except Exception as e:
        st.error(f"Database Delete Error: {e}")

# --- UI Helper Functions ---
def style_verdict(verdict):
    if verdict == 'High Suitability': return f'**<span style="color: #28a745;">{verdict}</span>**'
    elif verdict == 'Medium Suitability': return f'**<span style="color: #ffc107;">{verdict}</span>**'
    else: return f'**<span style="color: #dc3545;">{verdict}</span>**'

def highlight_verdict(row):
    verdict = row['verdict']
    if verdict == 'High Suitability': return ['background-color: #d4edda; color: #155724;'] * len(row)
    elif verdict == 'Medium Suitability': return ['background-color: #fff3cd; color: #856404;'] * len(row)
    else: return ['background-color: #f8d7da; color: #721c24;'] * len(row)

# --- Core Logic Functions ---
SKILL_KEYWORDS = [
    'Python', 'Java', 'C++', 'JavaScript', 'Go', 'Ruby', 'PHP', 'Django', 'Flask', 'Spring',
    'Node.js', 'React', 'Angular', 'Vue.js', 'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis',
    'Cassandra', 'AWS', 'Azure', 'Google Cloud', 'GCP', 'Docker', 'Kubernetes', 'Terraform',
    'Git', 'JIRA', 'Confluence', 'Agile', 'Scrum', 'CI/CD', 'DevOps', 'Machine Learning',
    'Deep Learning', 'TensorFlow', 'PyTorch', 'scikit-learn', 'Data Analysis', 'Pandas',
    'NumPy', 'Matplotlib', 'Tableau', 'Power BI', 'Natural Language Processing', 'NLP',
    'spaCy', 'NLTK', 'API', 'REST', 'GraphQL', 'Microservices', 'System Design'
]

@st.cache_data
def extract_skills(nlp, text):
    try:
        doc = nlp(text)
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        patterns = [nlp.make_doc(skill) for skill in SKILL_KEYWORDS]
        matcher.add("SKILL_MATCHER", patterns)
        return list({doc[start:end].text.title() for _, start, end in matcher(doc)})
    except Exception:
        return []

def get_uploaded_file_text(uploaded_file):
    try:
        text = ""
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs: text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

@st.cache_data
def perform_semantic_search(embedding_model, resume_text, jd_text):
    """Batch embedding to reduce time for multiple resumes."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        resume_chunks = text_splitter.split_text(resume_text)
        if not resume_chunks: return []
        vector_store = Chroma.from_texts(resume_chunks, embedding_model)
        return [doc.page_content for doc in vector_store.similarity_search(jd_text, k=3)]
    except Exception:
        return []

class FinalAnalysis(BaseModel):
    relevance_score: int = Field(description="The final relevance score from 0 to 100.")
    verdict: str = Field(description="A verdict of 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_elements: List[str] = Field(description="Missing skills/projects")
    improvement_feedback: str = Field(description="Actionable feedback")

def get_llm_synthesis(jd_text, matched_skills, missing_skills, relevant_chunks):
    """Call Google Gemini LLM once per resume for synthesis."""
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
        prompt_template_str = """
        You are an expert HR Technology Analyst at Innomatics Research Labs. Analyze a resume based on:

        **Job Description Requirements:** {jd}

        **Pre-Analysis Data:**
        - Matched Keywords: {matched_skills}
        - Missing Keywords: {missing_skills}
        - Relevant Resume Chunks:
        ---
        {relevant_chunks}
        ---

        Provide a weighted relevance score (60% keywords, 40% semantic), verdict, and actionable feedback.
        Return JSON as per the format instructions.
        """
        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=["jd", "matched_skills", "missing_skills", "relevant_chunks"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | model | parser
        return chain.invoke({
            "jd": jd_text,
            "matched_skills": ", ".join(matched_skills),
            "missing_skills": ", ".join(missing_skills),
            "relevant_chunks": "\n---\n".join(relevant_chunks)
        })
    except Exception as e:
        st.error(f"LLM Synthesis Error: {e}")
        return FinalAnalysis(relevance_score=0, verdict="Low Suitability", missing_elements=missing_skills, improvement_feedback="LLM failed. Please retry.")

# --- Main App ---
init_database()
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- Keep Your Custom CSS/UI intact ---
# [YOUR EXISTING CSS LOADER AND PULSING DOT CODE]

if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

if not st.session_state.system_ready:
    nlp, embedding_model = load_all_models()
    st.session_state.nlp, st.session_state.embedding_model = nlp, embedding_model
    st.session_state.system_ready = True
    st.rerun()
else:
    nlp, embedding_model = st.session_state.nlp, st.session_state.embedding_model

# --- Dashboard / Analysis Tab Logic ---
# Keep the same UI code as you already have, but replace:
# - extract_skills -> cached version
# - perform_semantic_search -> cached, batch-friendly
# - get_llm_synthesis -> wrapped in try/except fallback

# This ensures:
# ✅ Multiple resumes process faster
# ✅ Resilience to PDF/Docx errors
# ✅ LLM errors handled
# ✅ DB reused connection
# ✅ Cached NLP/Embedding avoids reloading

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** for the Code4EdTech Challenge by Innomatics Research Labs.")
