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
    relevance_score: int = Field(description="The final relevance score from 0 to 100.")
    verdict: str = Field(description="A verdict of 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_skills: List[str] = Field(description="List of 3-5 critical skills missing from the resume.")
    candidate_feedback: str = Field(description="Concise, actionable feedback for the candidate.")

    @validator('relevance_score')
    def score_must_be_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Relevance score must be between 0 and 100')
        return v

# --- Database Management ---
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row 
    return conn

def init_database():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
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

def migrate_database():
    """Ensures existing database has all required columns."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("PRAGMA table_info(results)")
    columns = [col[1] for col in c.fetchall()]
    required_columns = ["resume_filename", "jd_summary", "score", "verdict", "missing_skills", "feedback", "full_jd"]
    for col in required_columns:
        if col not in columns:
            c.execute(f"ALTER TABLE results ADD COLUMN {col} TEXT")
    conn.commit()

init_database()
migrate_database()

def add_analysis_to_db(filename, jd_text, report: FinalAnalysis):
    conn = get_db_connection()
    c = conn.cursor()
    jd_summary = " ".join(jd_text.split()[:15]).strip() + "..."
    c.execute('''
        INSERT INTO results (timestamp, resume_filename, jd_summary, score, verdict, missing_skills, feedback, full_jd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now(),
        filename,
        jd_summary,
        report.relevance_score,
        report.verdict,
        ", ".join(report.missing_skills) if report.missing_skills else "N/A",
        report.candidate_feedback,
        jd_text
    ))
    conn.commit()

def load_data_for_dashboard():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT id, timestamp, resume_filename, jd_summary, score, verdict, missing_skills FROM results ORDER BY id DESC", conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        return df
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return pd.DataFrame()

def get_single_record(record_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM results WHERE id = ?", (record_id,))
    return c.fetchone()

def delete_analysis_from_db(record_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM results WHERE id = ?", (record_id,))
    conn.commit()

# --- UI Helper Functions ---
def style_verdict(verdict):
    if verdict == 'High Suitability': return f'**<span style="color: #28a745;">{verdict}</span>**'
    elif verdict == 'Medium Suitability': return f'**<span style="color: #ffc107;">{verdict}</span>**'
    else: return f'**<span style="color: #dc3545;">{verdict}</span>**'

# --- Core Logic Functions ---
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
    skill_pattern = r"\b(" + "|".join(re.escape(skill) for skill in SKILL_KEYWORDS) + r")\b"
    matches = re.findall(skill_pattern, text, re.IGNORECASE)
    return sorted(list(set(match.title() for match in matches)))

def get_llm_analysis(jd_text, resume_text):
    if not GOOGLE_API_KEY:
        st.error("Google API Key is not configured.")
        return None
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
    
    prompt_template = """
    You are a world-class Senior Technical Recruiter...
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["jd", "resume"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | model | parser
    return chain.invoke({"jd": jd_text, "resume": resume_text})

# --- Streamlit UI ---
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = str(datetime.now().timestamp())
if 'jd_text_key' not in st.session_state:
    st.session_state.jd_text_key = ''

st.markdown("""
<style>
body[data-theme="dark"] .stImage > img { filter: invert(1); }
</style>
""", unsafe_allow_html=True)

title_col, button_col = st.columns([4, 1])
with title_col:
    st.image(LOGO_URL, width=250)
    st.title("Placement Team Dashboard")
with button_col:
    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
    if st.button("🧹 Clear Session", key="clear_button", use_container_width=True):
        st.session_state.jd_text_key = ""
        st.session_state.file_uploader_key = str(datetime.now().timestamp())
        st.rerun()

analysis_tab, dashboard_tab = st.tabs(["📊 Analysis", "🗂️ Dashboard"])

with analysis_tab:
    st.header("Run a New Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📋 Job Description")
        jd_text = st.text_area("Paste the Job Description here:", height=300, key="jd_text_key")
    with col2:
        st.subheader("📄 Candidate Resumes")
        uploaded_files = st.file_uploader("Upload resumes:", type=["pdf", "docx"], accept_multiple_files=True, key=st.session_state.file_uploader_key)

    if st.button("🚀 Run Full Analysis", key="analysis_button"):
        if not jd_text.strip() or not uploaded_files:
            st.error("Provide both JD and at least one resume.")
        else:
            required_skills = set(extract_skills_from_text(jd_text))
            st.info(f"Required Skills Detected: {', '.join(required_skills) if required_skills else 'None'}")
            progress_bar = st.progress(0)
            for i, resume_file in enumerate(uploaded_files):
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Analyzing {resume_file.name}...")
                resume_text = get_file_text(resume_file)
                if not resume_text: continue
                with st.spinner("Analyzing..."):
                    try:
                        final_report = get_llm_analysis(jd_text, resume_text)
                        if final_report:
                            add_analysis_to_db(resume_file.name, jd_text, final_report)
                            st.markdown(f"### {resume_file.name}")
                            st.markdown(style_verdict(final_report.verdict), unsafe_allow_html=True)
                            st.metric("Score", f"{final_report.relevance_score}%")
                            st.warning("**Identified Gaps:**")
                            st.markdown("\n".join([f"- {item}" for item in final_report.missing_skills]) or "- None")
                            st.success("**Personalized Feedback:**")
                            st.write(final_report.candidate_feedback)
                    except Exception as e:
                        st.error(f"Error analyzing {resume_file.name}: {e}")
                        st.exception(e)
            progress_bar.progress(1.0)
            st.success("All analyses complete!")
            st.balloons()

with dashboard_tab:
    st.header("Past Analysis Results")
    df = load_data_for_dashboard()
    if df.empty:
        st.info("No results found.")
    else:
        st.subheader("Filter & Manage Results")
        filt_col1, filt_col2, filt_col3 = st.columns(3)
        with filt_col1:
            jd_options = ["All JDs"] + list(df['jd_summary'].unique())
            selected_jd = st.selectbox("Filter by JD:", options=jd_options)
        df_filtered = df[df['jd_summary'] == selected_jd] if selected_jd != "All JDs" else df
        with filt_col2:
            verdict_options = ["All Verdicts"] + list(df_filtered['verdict'].unique())
            selected_verdict = st.selectbox("Filter by Verdict:", options=verdict_options)
        df_filtered = df_filtered[df_filtered['verdict'] == selected_verdict] if selected_verdict != "All Verdicts" else df_filtered
        with filt_col3:
            min_score = st.slider("Minimum Score:", 0, 100, 0)
        final_df = df_filtered[df_filtered['score'] >= min_score].reset_index(drop=True)
        st.subheader(f"Displaying {len(final_df)} Results")
        if not final_df.empty:
            for index, row in final_df.iterrows():
                st.markdown(f"**{row['resume_filename']}** | Score: `{row['score']}%` | Verdict: **{row['verdict']}**")
                if st.button("Delete", key=f"delete_{row['id']}"):
                    delete_analysis_from_db(row['id'])
                    st.success(f"Deleted record for {row['resume_filename']}.")
                    st.rerun()

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** for the Code4EdTech Challenge by Innomatics Research Labs.")
