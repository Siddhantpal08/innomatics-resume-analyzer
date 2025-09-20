import streamlit as st
import os
import sqlite3
import pandas as pd
import re
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from typing import List

# --- Core AI Libraries & Pydantic ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Global Configurations ---
DB_FILE = "analysis_results.db"
LOGO_URL = "https://www.innomatics.in/wp-content/uploads/2023/01/Innomatics-Logo1.png"
SKILL_KEYWORDS = [
    'Python', 'Java', 'C++', 'JavaScript', 'Go', 'Ruby', 'PHP', 'Django', 'Flask', 'Spring Boot', 'Node.js', 
    'React', 'Angular', 'Vue.js', 'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 
    'AWS', 'Azure', 'Google Cloud', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'Ansible', 'Git', 
    'JIRA', 'Confluence', 'Agile', 'Scrum', 'CI/CD', 'Jenkins', 'DevOps', 'Machine Learning', 
    'Deep Learning', 'TensorFlow', 'PyTorch', 'scikit-learn', 'Data Analysis', 'Pandas', 'NumPy', 
    'Matplotlib', 'Seaborn', 'Tableau', 'Power BI', 'Natural Language Processing', 'NLP', 
    'API', 'REST', 'GraphQL', 'Microservices', 'System Design', 'Big Data', 'Hadoop', 'Spark'
]

# --- Pydantic Model ---
class FinalAnalysis(BaseModel):
    relevance_score: int = Field(description="Score 0-100")
    verdict: str = Field(description="High/Medium/Low Suitability")
    missing_skills: List[str] = Field(description="3-5 critical missing skills")
    candidate_feedback: str = Field(description="Professional feedback")

    @validator('relevance_score')
    def score_must_be_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Relevance score must be between 0 and 100')
        return v

# --- Database Handling ---
def get_db_connection():
    """Single shared connection with timeout."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

conn = get_db_connection()

def init_database():
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            candidate_name TEXT NOT NULL,
            resume_filename TEXT NOT NULL,
            jd_summary TEXT NOT NULL,
            score INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            missing_skills TEXT,
            feedback TEXT,
            full_jd TEXT
        )
    ''')
    conn.commit()

init_database()

def add_analysis_to_db(candidate_name, filename, jd_text, report: FinalAnalysis):
    """Safely insert analysis into DB."""
    jd_summary = " ".join(jd_text.split()[:15]).strip() + "..."
    try:
        with conn:  # commits and releases lock automatically
            conn.execute('''
                INSERT INTO results (timestamp, candidate_name, resume_filename, jd_summary, score, verdict, missing_skills, feedback, full_jd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                candidate_name,
                filename,
                jd_summary,
                report.relevance_score,
                report.verdict,
                ", ".join(report.missing_skills) if report.missing_skills else "N/A",
                report.candidate_feedback,
                jd_text
            ))
    except sqlite3.OperationalError as e:
        st.error(f"SQLite write error: {e}")

def load_data_for_dashboard():
    try:
        df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        return df
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return pd.DataFrame()

def get_single_record(record_id):
    c = conn.cursor()
    c.execute("SELECT * FROM results WHERE id = ?", (record_id,))
    return c.fetchone()

def delete_analysis_from_db(record_id):
    try:
        with conn:
            conn.execute("DELETE FROM results WHERE id = ?", (record_id,))
    except sqlite3.OperationalError as e:
        st.error(f"SQLite delete error: {e}")

# --- UI Helpers ---
def style_verdict(verdict):
    if verdict == 'High Suitability': return f'**<span style="color: #28a745;">{verdict}</span>**'
    elif verdict == 'Medium Suitability': return f'**<span style="color: #ffc107;">{verdict}</span>**'
    else: return f'**<span style="color: #dc3545;">{verdict}</span>**'

def get_file_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""
    return text

@st.cache_data
def extract_skills_from_text(text):
    pattern = r"\b(" + "|".join(re.escape(skill) for skill in SKILL_KEYWORDS) + r")\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return sorted(list(set(match.title() for match in matches)))

def get_llm_analysis(jd_text, resume_text):
    if not GOOGLE_API_KEY:
        st.error("Google API Key not configured.")
        return None
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
    
    prompt_template = """
    You are a senior recruiter. Evaluate resume vs JD and output JSON adhering to FinalAnalysis model.
    JD:
    {jd}
    Resume:
    {resume}
    {format_instructions}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["jd","resume"], partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | model | parser
    return chain.invoke({"jd": jd_text, "resume": resume_text})

# --- Streamlit App ---
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide")
st.title("📊 Placement Team Dashboard")
st.image(LOGO_URL, width=250)

# --- File Uploader ---
jd_text = st.text_area("Paste Job Description:", height=300)
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf","docx"], accept_multiple_files=True)

if st.button("🚀 Run Full Analysis"):
    if not jd_text.strip() or not uploaded_files:
        st.error("Provide both JD and at least one resume.")
    else:
        required_skills = set(extract_skills_from_text(jd_text))
        st.info(f"**Required Skills Detected in JD:** {', '.join(required_skills) if required_skills else 'None'}")
        
        for resume_file in uploaded_files:
            candidate_name = resume_file.name.split(".")[0]
            st.markdown(f"### Candidate: {candidate_name}")
            resume_text = get_file_text(resume_file)
            if not resume_text: continue

            with st.spinner(f"Analyzing {resume_file.name}..."):
                try:
                    final_report = get_llm_analysis(jd_text, resume_text)
                    if final_report:
                        add_analysis_to_db(candidate_name, resume_file.name, jd_text, final_report)
                        st.markdown(style_verdict(final_report.verdict), unsafe_allow_html=True)
                        st.metric("Score", f"{final_report.relevance_score}%")
                        st.warning("**Identified Gaps:**")
                        st.markdown("\n".join([f"- {item}" for item in final_report.missing_skills]) or "- None")
                        st.success("**Feedback:**")
                        st.write(final_report.candidate_feedback)
                except Exception as e:
                    st.error(f"Error analyzing {resume_file.name}: {e}")

# --- Dashboard ---
st.markdown("---")
st.header("Past Analysis Results")
df = load_data_for_dashboard()
if df.empty:
    st.info("No results yet.")
else:
    st.dataframe(df[['timestamp','candidate_name','resume_filename','score','verdict']])
