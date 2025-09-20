import streamlit as st
import os
import sqlite3
import pandas as pd
import re
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from typing import List, Optional

# --- Core AI Libraries & Pydantic ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Global Configurations ---
DB_FILE = "analysis_results.db"
LOGO_URL = "https://www.innomatics.in/wp-content/uploads/2023/01/Innomatics-Logo1.png"

# --- Pydantic Models for a more robust AI chain ---
class JDSkills(BaseModel):
    job_title: str = Field(description="The specific job title from the description (e.g., 'Senior Python Developer').")
    hard_skills: List[str] = Field(description="A list of 5-7 critical technical skills, frameworks, or technologies.")
    experience_years: Optional[str] = Field(description="The required years of experience (e.g., '3+ years', '5 years'). If not mentioned, this should be null.")

class ResumeSkills(BaseModel):
    demonstrated_skills: List[str] = Field(description="A list of skills the candidate has explicitly used in a project or work experience.")
    listed_skills: List[str] = Field(description="A list of skills mentioned in a general skills section but not tied to a specific experience.")

class FinalAnalysis(BaseModel):
    relevance_score: int = Field(description="The final relevance score from 0 to 100.")
    verdict: str = Field(description="A verdict of 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_skills: List[str] = Field(description="A list of the 3-5 most critical skills from the JD that are completely missing from the resume.")
    candidate_feedback: str = Field(description="A concise, professional, and actionable feedback paragraph for the candidate.")

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
            jd_title TEXT NOT NULL,
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

def add_analysis_to_db(filename, jd_title, jd_text, report: FinalAnalysis):
    conn = get_db_connection()
    c = conn.cursor()
    jd_summary = " ".join(jd_text.split()[:15]).strip() + "..."
    c.execute('''
        INSERT INTO results (timestamp, resume_filename, jd_title, jd_summary, score, verdict, missing_skills, feedback, full_jd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now(), filename, jd_title, jd_summary, report.relevance_score,
        report.verdict, ", ".join(report.missing_skills) if report.missing_skills else "N/A",
        report.candidate_feedback, jd_text
    ))
    conn.commit()

def load_data_for_dashboard():
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT id, timestamp, resume_filename, jd_title, jd_summary, score, verdict, missing_skills FROM results ORDER BY id DESC", conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        return df
    except Exception:
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

def check_if_exists(filename, jd_summary):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM results WHERE resume_filename = ? AND jd_summary = ?", (filename, jd_summary))
    return c.fetchone() is not None

# --- UI Helper Functions ---
def get_verdict_color(verdict):
    if verdict == 'High Suitability': return '#28a745'
    elif verdict == 'Medium Suitability': return '#ffc107'
    else: return '#dc3545'

# --- Core Logic Functions ---
def get_file_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages: text += page.extract_text() or ""
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            for para in doc.paragraphs: text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""
    return text

def get_llm_analysis(jd_text, resume_text):
    if not GOOGLE_API_KEY:
        st.error("Google API Key is not configured. Please set it in your secrets.")
        return None, None
        
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

    # Step 1: Extract structured data from JD and Resume
    jd_parser = PydanticOutputParser(pydantic_object=JDSkills)
    resume_parser = PydanticOutputParser(pydantic_object=ResumeSkills)

    jd_prompt = PromptTemplate(template="Extract the key skills and job title from this job description.\n{format_instructions}\nJD:\n{jd}", input_variables=["jd"], partial_variables={"format_instructions": jd_parser.get_format_instructions()})
    resume_prompt = PromptTemplate(template="Extract the key skills from this resume, separating skills listed in a skills section from those demonstrated in work experience.\n{format_instructions}\nResume:\n{resume}", input_variables=["resume"], partial_variables={"format_instructions": resume_parser.get_format_instructions()})

    jd_chain = jd_prompt | model | jd_parser
    resume_chain = resume_prompt | model | resume_parser

    jd_skills = jd_chain.invoke({"jd": jd_text})
    resume_skills = resume_chain.invoke({"resume": resume_text})

    # Step 2: Analyze the structured data
    analysis_parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
    analysis_prompt_template = """
    You are a fair and highly experienced Senior Technical Recruiter. Your task is to provide an accurate relevance score by comparing the structured data from a Job Description and a Resume.

    **JOB REQUIREMENTS:**
    - Job Title: {job_title}
    - Critical Skills: {jd_hard_skills}
    - Required Experience: {jd_experience}

    **CANDIDATE'S SKILLS:**
    - Skills Demonstrated in Work/Projects: {resume_demonstrated}
    - Skills Simply Listed in a List: {resume_listed}

    **EVALUATION TASKS:**

    1.  **SCORE (0-100):** Calculate a score based on how well the candidate's skills match the critical skills.
        - Start with a base score of 0.
        - For each skill in 'Critical Skills' that is also in 'Demonstrated Skills', add 15 points. This is a strong match.
        - For each skill in 'Critical Skills' that is only in 'Listed Skills', add 5 points. This is a weak match.
        - If 'Required Experience' is mentioned and the resume seems to align, add 10 bonus points.
        - Cap the total score at 100.

    2.  **VERDICT:** Based on the score, provide a verdict:
        - **High Suitability (70-100)**
        - **Medium Suitability (40-69)**
        - **Low Suitability (<40)**
        
    3.  **MISSING SKILLS:** List the top 3-5 skills from 'Critical Skills' that are absent from BOTH of the candidate's skill lists.

    4.  **FEEDBACK:** Write a brief, constructive paragraph. Start with a positive point and then mention the key missing skills.

    Provide your final analysis in the required JSON format.
    {format_instructions}
    """
    analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["job_title", "jd_hard_skills", "jd_experience", "resume_demonstrated", "resume_listed"], partial_variables={"format_instructions": analysis_parser.get_format_instructions()})
    analysis_chain = analysis_prompt | model | analysis_parser
    
    final_report = analysis_chain.invoke({
        "job_title": jd_skills.job_title,
        "jd_hard_skills": jd_skills.hard_skills,
        "jd_experience": jd_skills.experience_years or "Not Specified",
        "resume_demonstrated": resume_skills.demonstrated_skills,
        "resume_listed": resume_skills.listed_skills
    })
    
    return final_report, jd_skills.job_title

# --- Main App UI & Logic ---

# --- CSS Styling ---
st.markdown("""
<style>
    /* Invert logo in dark mode */
    body[data-theme="dark"] [data-testid="stImage"] > img {
        filter: invert(1);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.image(LOGO_URL, width=250)
st.title("Automated Resume Relevance Checker")
st.markdown("---")

# --- Main App Body ---
analysis_tab, dashboard_tab = st.tabs(["📊 Analysis", "🗂️ Dashboard"])

with analysis_tab:
    st.header("1. Input Job and Resume Data")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            jd_text = st.text_area("Paste the full Job Description here:", height=300, key="jd_text_key", placeholder="e.g., 'Seeking a Python developer with 3+ years of experience in Django, REST APIs, and PostgreSQL...'")
        with col2:
            uploaded_files = st.file_uploader("Upload one or more resumes (PDF, DOCX):", type=["pdf", "docx"], accept_multiple_files=True, key="file_uploader_key")

    st.write("") 

    if st.button("🚀 Analyze Resumes", type="primary", key="analysis_button", use_container_width=True):
        if not jd_text.strip() or not uploaded_files:
            st.error("Please provide both a Job Description and at least one resume.")
        else:
            files_to_process = []
            jd_summary = " ".join(jd_text.split()[:15]).strip() + "..."
            for file in uploaded_files:
                if check_if_exists(file.name, jd_summary):
                    st.warning(f"Skipping '{file.name}': This resume has already been analyzed for this job description.")
                else:
                    files_to_process.append(file)
            
            if files_to_process:
                progress_bar = st.progress(0, text="Initializing...")
                for i, resume_file in enumerate(files_to_process):
                    progress_bar.progress((i + 1) / len(files_to_process), text=f"Analyzing {resume_file.name}...")
                    
                    with st.container(border=True):
                        st.markdown(f"### Candidate: {resume_file.name}")
                        resume_text = get_file_text(resume_file)
                        if not resume_text: continue

                        with st.spinner("AI is performing a deep analysis..."):
                            try:
                                final_report, job_title = get_llm_analysis(jd_text, resume_text)
                                if final_report:
                                    add_analysis_to_db(resume_file.name, job_title, jd_text, final_report)
                                    
                                    res_col1, res_col2 = st.columns([1, 3])
                                    with res_col1:
                                        st.markdown(f"<p style='color:{get_verdict_color(final_report.verdict)};'><strong>{final_report.verdict}</strong></p>", unsafe_allow_html=True)
                                        st.metric("Score", f"{final_report.relevance_score}%")
                                    with res_col2:
                                        st.warning("**Identified Gaps:**")
                                        st.markdown("\n".join([f"- {item}" for item in final_report.missing_skills]) or "- None")
                                        st.success("**Personalized Feedback:**")
                                        st.write(final_report.candidate_feedback)
                            except Exception as e:
                                st.error(f"An error occurred while analyzing {resume_file.name}: {e}")
                                st.exception(e)
                st.success("All new resumes have been analyzed!")
                st.balloons()

@st.dialog("Full Report")
def show_report_modal(record_id):
    record = get_single_record(record_id)
    if record:
        st.subheader(f"Analysis for: {record['resume_filename']}")
        st.metric("Relevance Score", f"{record['score']}%")
        st.markdown(f"<p style='color:{get_verdict_color(record['verdict'])};'><strong>Verdict: {record['verdict']}</strong></p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Identified Gaps")
        st.markdown(record['missing_skills'])

        st.markdown("---")
        st.subheader("Candidate Feedback")
        st.info(record['feedback'])
        
        st.markdown("---")
        with st.expander("Show Original Job Description"):
            st.text_area("JD", value=record['full_jd'], height=200, disabled=True, label_visibility="collapsed")
    else:
        st.error("Could not retrieve report.")

with dashboard_tab:
    st.header("Past Analysis Results")
    df = load_data_for_dashboard()
    
    if df.empty:
        st.info("No results found. Run a new analysis in the 'Analysis' tab.")
    else:
        st.subheader("Filter & Manage Results")
        filt_col1, filt_col2, filt_col3 = st.columns(3)
        
        with filt_col1:
            jd_options = ["All JDs"] + list(df['jd_title'].unique())
            selected_jd = st.selectbox("Filter by Job Title:", options=jd_options, key="jd_filter")
        
        df_filtered = df[df['jd_title'] == selected_jd] if selected_jd != "All JDs" else df
        
        with filt_col2:
            verdict_options = ["All Verdicts"] + list(df_filtered['verdict'].unique())
            selected_verdict = st.selectbox("Filter by Verdict:", options=verdict_options, key="verdict_filter")

        df_filtered = df_filtered[df_filtered['verdict'] == selected_verdict] if selected_verdict != "All Verdicts" else df_filtered

        with filt_col3:
            min_score = st.slider("Minimum Score:", 0, 100, 0, key="score_slider")

        final_df = df_filtered[df_filtered['score'] >= min_score].reset_index(drop=True)
        
        st.subheader(f"Displaying {len(final_df)} Results")
        st.write("")

        if not final_df.empty:
            for index, row in final_df.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([5, 1, 1])
                    with col1:
                        st.markdown(f"**{row['resume_filename']}** for **{row['jd_title']}**")
                        
                        verdict_color = get_verdict_color(row['verdict'])
                        missing_skills_summary = row['missing_skills']
                        if len(missing_skills_summary) > 45:
                            missing_skills_summary = missing_skills_summary[:45] + "..."
                        
                        st.markdown(f"Score: `{row['score']}%` | Verdict: <span style='color:{verdict_color};'>**{row['verdict']}**</span> | Gaps: *{missing_skills_summary}*", unsafe_allow_html=True)
                    
                    with col2:
                        st.write("") 
                        if st.button("Details", key=f"view_{row['id']}", use_container_width=True):
                            show_report_modal(row['id'])
                    with col3:
                        st.write("")
                        if st.button("Delete", key=f"delete_{row['id']}", type="secondary", use_container_width=True):
                            delete_analysis_from_db(row['id'])
                            st.success(f"Deleted record for {row['resume_filename']}.")
                            st.rerun()
        else:
            st.info("No records match the current filter criteria.")

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** for the Code4EdTech Challenge by Innomatics Research Labs.")
