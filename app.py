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

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
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

# --- Pydantic Model for Structured LLM Output ---
class FinalAnalysis(BaseModel):
    relevance_score: int = Field(description="The final relevance score from 0 to 100.")
    verdict: str = Field(description="A verdict of 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_skills: List[str] = Field(description="A list of the 3-5 most critical skills or qualifications explicitly mentioned in the job description but completely missing from the resume.")
    candidate_feedback: str = Field(description="A concise, professional, and actionable feedback paragraph for the candidate.")

    @validator('relevance_score')
    def score_must_be_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Relevance score must be between 0 and 100')
        return v

# --- Database Management ---
@st.cache_resource
def get_db_connection():
    """Creates a cached, singleton connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row 
    return conn

def init_database():
    """Initializes the database table."""
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

init_database()

def add_analysis_to_db(filename, jd_text, report: FinalAnalysis):
    """Adds a new analysis report to the database."""
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
    """Loads all analysis results into a Pandas DataFrame."""
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
    """Retrieves a single, complete record from the database by its ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM results WHERE id = ?", (record_id,))
    return c.fetchone()

def delete_analysis_from_db(record_id):
    """Deletes a specific analysis record by its ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM results WHERE id = ?", (record_id,))
    conn.commit()
    
# --- UI Helper Functions ---
def get_verdict_color(verdict):
    if verdict == 'High Suitability': return '#28a745'
    elif verdict == 'Medium Suitability': return '#ffc107'
    else: return '#dc3545'
    
# --- Core Logic Functions ---
def get_file_text(uploaded_file):
    """Extracts text from uploaded PDF or DOCX file."""
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
    """Extracts skills using a robust regular expression."""
    skill_pattern = r"\b(" + "|".join(re.escape(skill) for skill in SKILL_KEYWORDS) + r")\b"
    matches = re.findall(skill_pattern, text, re.IGNORECASE)
    return sorted(list(set(match.title() for match in matches)))

def get_llm_analysis(jd_text, resume_text):
    """Orchestrates the LLM call for a comprehensive and fair analysis."""
    if not GOOGLE_API_KEY:
        st.error("Google API Key is not configured. Please set it in your secrets.")
        return None
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
    
    prompt_template = """
    You are a highly experienced Senior Technical Recruiter. Your primary goal is to be **fair and accurate**. You must evaluate a candidate's resume against a job description by focusing on **demonstrated experience**, not just keyword matching.

    **CONTEXT:**
    - **Job Description (JD):**
      ```
      {jd}
      ```
    - **Candidate's Resume Content:**
      ```
      {resume}
      ```

    **EVALUATION CRITERIA (Follow these steps precisely):**

    1.  **Identify Core Requirements:** Analyze the JD to identify the 5-7 most critical skills and qualifications.
    
    2.  **Evidence-Based Analysis:** For each core requirement, find **direct evidence** in the resume's "Work Experience" or "Projects" sections. A skill simply listed in a "Skills" section is a weak match. A skill demonstrated in a project or professional role is a **strong match**.
    
    3.  **FAIR SCORING (0-100):**
        - **High Suitability (70-100):** The candidate provides strong, demonstrated evidence for nearly all core requirements. Their experience is directly and obviously relevant.
        - **Medium Suitability (40-69):** The candidate demonstrates some core skills but is missing others, or their experience is related but not a direct match. The candidate is plausible.
        - **Low Suitability (<40):** The resume is missing the majority of core requirements or lacks any demonstrated experience for the listed skills.

    4.  **Actionable Feedback:**
        - Identify the 3-5 most critical missing skills or qualifications.
        - Write a professional, constructive feedback paragraph for the candidate, highlighting a strength first, then clearly stating the key areas for improvement for this specific type of role.

    **OUTPUT FORMAT:**
    You MUST format your entire response as a single, valid JSON object. Do not add any text or markdown before or after the JSON object.
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["jd", "resume"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | model | parser
    return chain.invoke({
        "jd": jd_text,
        "resume": resume_text,
    })

# --- Main App UI & Logic ---

# Session state initialization
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = str(datetime.now().timestamp())
if 'jd_text_key' not in st.session_state:
    st.session_state.jd_text_key = ''

# CSS to make the logo compatible with Streamlit's native dark mode
st.markdown("""
<style>
    /* Invert logo in dark mode */
    [data-testid="stAppViewContainer"] [data-testid="stImage"] > img {
        filter: invert(1);
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
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

# --- Main App Body ---
analysis_tab, dashboard_tab = st.tabs(["📊 Analysis", "🗂️ Dashboard"])

with analysis_tab:
    st.header("Run a New Analysis")
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📋 Job Description")
            jd_text = st.text_area(
                "Paste the Job Description text here:", 
                height=300, 
                key="jd_text_key", 
                label_visibility="collapsed",
                placeholder="e.g., 'Seeking a Python developer with 3+ years of experience in Django, REST APIs, and PostgreSQL. Experience with AWS is a plus...'"
            )
        with col2:
            st.subheader("📄 Candidate Resumes")
            uploaded_files = st.file_uploader("Upload resumes:", type=["pdf", "docx"], accept_multiple_files=True, key=st.session_state.file_uploader_key, label_visibility="collapsed")
    st.write("") 

    if st.button("🚀 Run Full Analysis", type="primary", key="analysis_button", use_container_width=True):
        if not jd_text.strip() or not uploaded_files:
            st.error("Please provide both a Job Description and at least one resume.")
        else:
            required_skills = set(extract_skills_from_text(jd_text))
            st.info(f"**Required Skills Detected in JD:** {', '.join(required_skills) if required_skills else 'None'}")
            
            progress_bar = st.progress(0, text="Initializing...")
            
            for i, resume_file in enumerate(uploaded_files):
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Analyzing {resume_file.name}...")
                
                with st.container(border=True):
                    st.markdown(f"### Candidate: {resume_file.name}")
                    resume_text = get_file_text(resume_file)
                    if not resume_text: continue

                    with st.spinner("AI is analyzing and generating the report..."):
                        try:
                            final_report = get_llm_analysis(jd_text, resume_text)
                            if final_report:
                                add_analysis_to_db(resume_file.name, jd_text, final_report)
                                
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

            progress_bar.progress(1.0, text="All analyses complete!")
            st.success("All analyses complete!")
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
            jd_options = ["All JDs"] + list(df['jd_summary'].unique())
            selected_jd = st.selectbox("Filter by Job Description:", options=jd_options, key="jd_filter")
        
        df_filtered = df[df['jd_summary'] == selected_jd] if selected_jd != "All JDs" else df
        
        with filt_col2:
            verdict_options = ["All Verdicts"] + list(df_filtered['verdict'].unique())
            selected_verdict = st.selectbox("Filter by Verdict:", options=verdict_options, key="verdict_filter")

        df_filtered = df_filtered[df_filtered['verdict'] == selected_verdict] if selected_verdict != "All Verdicts" else df_filtered

        with filt_col3:
            min_score = st.slider("Minimum Score:", 0, 100, 0, key="score_slider")

        final_df = df_filtered[df_filtered['score'] >= min_score].reset_index(drop=True)
        
        st.subheader(f"Displaying {len(final_df)} Results")
        
        if not final_df.empty:
            for index, row in final_df.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([5, 1, 1])
                    with col1:
                        st.markdown(f"**{row['resume_filename']}**")
                        
                        verdict_color = get_verdict_color(row['verdict'])
                        missing_skills_summary = row['missing_skills']
                        if len(missing_skills_summary) > 45:
                            missing_skills_summary = missing_skills_summary[:45] + "..."
                        
                        st.markdown(f"Score: `{row['score']}%` | Verdict: <span style='color:{verdict_color};'>**{row['verdict']}**</span> | Gaps: *{missing_skills_summary}*", unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("Details", key=f"view_{row['id']}", use_container_width=True):
                            show_report_modal(row['id'])
                    with col3:
                        if st.button("Delete", key=f"delete_{row['id']}", type="secondary", use_container_width=True):
                            delete_analysis_from_db(row['id'])
                            st.success(f"Deleted record for {row['resume_filename']}.")
                            st.rerun()
        else:
            st.info("No records match the current filter criteria.")

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** for the Code4EdTech Challenge by Innomatics Research Labs.")
