import streamlit as st
import sqlite3
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import re
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Skill Database ---
SKILL_DB = [
    'python', 'java', 'c++', 'javascript', 'html', 'css', 'sql', 'mysql', 'postgresql', 'mongodb', 'react',
    'angular', 'vue', 'django', 'flask', 'spring', 'nodejs', 'expressjs', 'rubyonrails', 'php', 'laravel',
    'dotnet', 'asp.net', 'git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform', 'ansible',
    'jenkins', 'ci/cd', 'agile', 'scrum', 'jira', 'confluence', 'linux', 'unix', 'shellscripting',
    'machinelearning', 'deeplearning', 'tensorflow', 'pytorch', 'keras', 'scikitlearn', 'pandas',
    'numpy', 'datascience', 'dataanalysis', 'datavisualization', 'bigdata', 'hadoop', 'spark'
]

# --- Database Setup and Connection ---
DB_FILE = "resume_analysis.db"

@st.cache_resource
def get_db_connection():
    """Creates a cached, singleton connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn

def init_db():
    """Initializes the database table if it doesn't exist."""
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

# Initialize the database on first run
init_db()


# --- Helper Functions ---
def get_resume_text(uploaded_file):
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
def get_hard_skills_matches(resume_text):
    matched_skills = []
    resume_text_lower = resume_text.lower()
    for skill in SKILL_DB:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, resume_text_lower):
            matched_skills.append(skill)
    return list(set(matched_skills))

# --- Pydantic Models for Structured Output ---
class AnalysisResult(BaseModel):
    relevance_score: int = Field(description="A score from 0 to 100 representing how relevant the resume is to the job description.")
    verdict: str = Field(description="A final verdict: 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_skills: list[str] = Field(description="A list of key skills from the job description that are missing in the resume.")
    candidate_feedback: str = Field(description="Constructive, personalized feedback for the candidate on how to improve their resume for this specific role.")

# --- UI Helper Functions ---
def highlight_verdict(row):
    verdict = row['verdict']
    if verdict == 'High Suitability':
        return ['background-color: #d4edda; color: #155724;'] * len(row)
    elif verdict == 'Medium Suitability':
        return ['background-color: #fff3cd; color: #856404;'] * len(row)
    else:
        return ['background-color: #f8d7da; color: #721c24;'] * len(row)

# --- Streamlit UI ---
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide")

# --- Page Selection ---
page = st.sidebar.radio("Navigation", ["Analyze New Resumes", "Analysis Dashboard"])

# --- Analysis Page ---
def analysis_page():
    st.title("🤖 Resume Relevance Analyzer")
    st.markdown("##### Provide a Job Description and upload resumes to begin.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            jd_text = st.text_area("Paste the Job Description here:", height=300)
        with col2:
            uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("🚀 Run Analysis", type="primary"):
        if not jd_text or not uploaded_files:
            st.warning("Please provide both a Job Description and at least one resume.")
            return
        if not GOOGLE_API_KEY:
            st.error("Google API Key not found. Please ensure it's set in your environment variables.")
            return

        progress_bar = st.progress(0, text="Starting Analysis...")
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i) / len(uploaded_files), text=f"Analyzing {uploaded_file.name}...")
            
            with st.container(border=True):
                resume_text = get_resume_text(uploaded_file)
                if not resume_text:
                    st.error(f"Could not extract text from {uploaded_file.name}. Skipping.")
                    continue

                matched_skills = get_hard_skills_matches(resume_text)
                
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
                parser = JsonOutputParser(pydantic_object=AnalysisResult)

                prompt_template = """
                You are an expert HR Technology Analyst. Your task is to provide a precise, data-driven analysis of a resume against a job description.

                CONTEXT:
                - Job Description: {jd}
                - Resume Text: {resume}
                - Pre-matched Hard Skills from Resume: {skills}

                INSTRUCTIONS: Follow these steps precisely.
                1.  **Analyze JD Requirements:** First, identify the top 5-7 most important technical skills and qualifications explicitly required by the job description.
                2.  **Compare and Identify Missing Skills:** Compare the list of required skills from the JD against the ENTIRE resume content and the provided "Pre-matched Hard Skills". List only the critical skills that are genuinely missing from the resume. Do not infer skills that aren't present.
                3.  **Calculate Relevance Score:** Based on the comparison, determine a score from 0-100. The score should reflect the alignment of skills, experience, and qualifications. A high score requires strong evidence of multiple key skills from the JD.
                4.  **Give a Verdict:** Provide a verdict based on the score: 'High Suitability' (score > 75), 'Medium Suitability' (score 45-75), or 'Low Suitability' (score < 45).
                5.  **Generate Candidate Feedback:** Write a brief, professional paragraph for the candidate. Acknowledge their strengths and then suggest specific ways to improve their resume for this role, mentioning 2-3 key missing skills they should focus on acquiring or highlighting.

                Format your entire output as a single JSON object with these exact keys: "relevance_score", "verdict", "missing_skills", "candidate_feedback".
                {format_instructions}
                """
                
                prompt = PromptTemplate(template=prompt_template, input_variables=["jd", "resume", "skills"], partial_variables={"format_instructions": parser.get_format_instructions()})
                chain = prompt | llm | parser
                
                try:
                    analysis = chain.invoke({"jd": jd_text, "resume": resume_text, "skills": ", ".join(matched_skills)})
                    st.subheader(f"Analysis for: {uploaded_file.name}")
                    col1, col2 = st.columns(2)
                    col1.metric("Relevance Score", f"{analysis['relevance_score']}%")
                    col2.metric("Verdict", analysis['verdict'])
                    
                    with st.expander("View Details"):
                        st.warning(f"**Missing Skills:** {', '.join(analysis['missing_skills']) if analysis['missing_skills'] else 'None'}")
                        st.info(f"**Feedback for Candidate:** {analysis['candidate_feedback']}")
                    
                    # Save to DB
                    conn = get_db_connection()
                    c = conn.cursor()
                    jd_summary = " ".join(jd_text.split()[:10]).strip() + "..."
                    c.execute('''
                        INSERT INTO results (timestamp, resume_filename, jd_summary, score, verdict, missing_skills, feedback, full_jd)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (datetime.now(), uploaded_file.name, jd_summary, analysis['relevance_score'], analysis['verdict'], ", ".join(analysis['missing_skills']), analysis['candidate_feedback'], jd_text))
                    conn.commit()

                except Exception as e:
                    st.error(f"An error occurred with the AI model for {uploaded_file.name}: {e}")

        progress_bar.progress(1.0, text="Analysis Complete!")
        st.success("Analysis complete for all resumes!")
        st.balloons()


# --- Dashboard Page ---
def dashboard_page():
    st.title("🗂️ Analysis Dashboard")
    conn = get_db_connection()
    
    try:
        df = pd.read_sql_query("SELECT id, timestamp, resume_filename, jd_summary, score, verdict, missing_skills, feedback, full_jd FROM results ORDER BY id DESC", conn)
        if not df.empty:
             df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        st.error(f"Could not load data from database: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No results found. Run a new analysis on the 'Analyze New Resumes' page.")
        return

    st.subheader("Filter & Manage Results")
    filt_col1, filt_col2, filt_col3 = st.columns(3)
    
    with filt_col1:
        jd_options = ["All JDs"] + list(df['jd_summary'].unique())
        selected_jd = st.selectbox("Filter by Job Description:", options=jd_options)
    
    df_filtered = df[df['jd_summary'] == selected_jd] if selected_jd != "All JDs" else df
    
    with filt_col2:
        verdict_options = ["All Verdicts"] + list(df_filtered['verdict'].unique())
        selected_verdict = st.selectbox("Filter by Verdict:", options=verdict_options)

    df_filtered = df_filtered[df_filtered['verdict'] == selected_verdict] if selected_verdict != "All Verdicts" else df_filtered

    with filt_col3:
        min_score = st.slider("Minimum Score:", 0, 100, 0)
    
    final_df = df_filtered[df_filtered['score'] >= min_score]
    
    st.subheader(f"Displaying {len(final_df)} Results")
    st.dataframe(
        final_df[['id', 'timestamp', 'resume_filename', 'jd_summary', 'score', 'verdict']].style.apply(highlight_verdict, axis=1),
        hide_index=True,
        use_container_width=True
    )
    
    if not final_df.empty:
        st.markdown("---")
        col_del, col_view = st.columns(2)
        with col_del:
            st.subheader("Delete a Record")
            delete_options = [f"{row['id']} - {row['resume_filename']}" for _, row in final_df.iterrows()]
            record_to_delete_display = st.selectbox("Select record:", options=[""] + delete_options, label_visibility="collapsed")
            
            if st.button("❌ Delete Selected Record") and record_to_delete_display:
                record_id_to_delete = int(record_to_delete_display.split(" - ")[0])
                c = conn.cursor()
                c.execute("DELETE FROM results WHERE id = ?", (record_id_to_delete,))
                conn.commit()
                st.success(f"Record ID {record_id_to_delete} has been deleted.")
                st.rerun()

        with col_view:
            st.subheader("View Full Details")
            view_options = [f"{row['id']} - {row['resume_filename']}" for _, row in final_df.iterrows()]
            record_to_view_display = st.selectbox("Select record:", options=[""] + view_options, label_visibility="collapsed")

            if record_to_view_display:
                record_id_to_view = int(record_to_view_display.split(" - ")[0])
                record = df[df['id'] == record_id_to_view].iloc[0]
                with st.container(border=True):
                    st.write(f"**Candidate:** {record['resume_filename']}")
                    st.text_area("Full Job Description", record['full_jd'], height=150, disabled=True)
                    st.warning(f"**Missing Skills:** {record['missing_skills']}")
                    st.info(f"**Feedback:** {record['feedback']}")

# --- Page Routing ---
if page == "Analyze New Resumes":
    analysis_page()
else:
    dashboard_page()

