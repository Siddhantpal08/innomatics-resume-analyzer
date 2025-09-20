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
def style_verdict(verdict):
    if verdict == 'High Suitability': return f'**<span style="color: #28a745;">{verdict}</span>**'
    elif verdict == 'Medium Suitability': return f'**<span style="color: #ffc107;">{verdict}</span>**'
    else: return f'**<span style="color: #dc3545;">{verdict}</span>**'
    
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
    """Orchestrates the LLM call for a comprehensive analysis."""
    if not GOOGLE_API_KEY:
        st.error("Google API Key is not configured. Please set it in your secrets.")
        return None
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
    
    prompt_template = """
    You are a world-class Senior Technical Recruiter with 20 years of experience, known for your meticulous, evidence-based analysis. Your task is to provide a rigorous evaluation of a resume against a job description. Your reputation depends on your precision and honesty.

    **CONTEXT:**
    1.  **Job Description (JD):**
        ```
        {jd}
        ```
    2.  **Candidate's Resume Content:**
        ```
        {resume}
        ```

    **YOUR TASK (Follow these steps precisely):**

    1.  **Identify Core Requirements:** Scrutinize the JD and list the 5-7 most critical, non-negotiable hard skills, technologies, and experience qualifications (e.g., "5+ years of experience in Python", "experience with AWS S3 and EC2", "CI/CD pipeline management"). These are your primary evaluation criteria.
    
    2.  **Evidence-Based Skill Gap Analysis:** For each core requirement identified in Step 1, you must perform a forensic scan of the **entire Resume Content**. Find direct evidence. If a skill is mentioned, note it. If it is described with project experience, note that as a stronger match. If it is completely absent, you MUST list it as a missing skill. Do not make assumptions or infer skills that aren't explicitly stated.
    
    3.  **Calculate Relevance Score (0-100):** Based *only* on your evidence-based analysis, provide a score.
        - **High Suitability (75-100):** The candidate's resume provides strong, explicit evidence for almost all ( > 80%) of the core requirements. The experience described is directly relevant.
        - **Medium Suitability (45-74):** The resume shows evidence for some core requirements (40-70%), but is missing others, or the experience lacks depth and specific examples. The candidate is plausible but not a perfect match.
        - **Low Suitability (<45):** The resume is missing a majority of the core requirements. The candidate is a clear mismatch for this specific role.

    4.  **Write Professional Feedback:** Create a professional, constructive feedback paragraph for the candidate. Begin by acknowledging a specific, tangible strength from their resume. Then, clearly and directly state the 2-3 most critical missing skills you identified, explaining why they are important for this type of role. This feedback should be actionable and helpful.

    **OUTPUT FORMAT:**
    You MUST format your entire response as a single, valid JSON object that adheres to the following structure. Do not add any text, explanations, or markdown before or after the JSON object.
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
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 'initial'
if 'jd_text_key' not in st.session_state:
    st.session_state.jd_text_key = ''

# CSS for theming and logo inversion
st.markdown(f"""
<style>
    :root {{
        --bg-color: #FFFFFF;
        --secondary-bg-color: #F0F2F6;
        --text-color: #31333F;
        --secondary-text-color: #5A5A64;
        --border-color: #E6E6E6;
    }}
    html[data-theme="dark"] {{
        --bg-color: #0E1117;
        --secondary-bg-color: #262730;
        --text-color: #FAFAFA;
        --secondary-text-color: #B9B9C3;
        --border-color: #31333F;
    }}
    .stApp {{
        background-color: var(--bg-color);
    }}
    /* Invert logo in dark mode */
    html[data-theme="dark"] .innomatics-logo img {{
        filter: invert(1) hue-rotate(180deg);
    }}
</style>
""", unsafe_allow_html=True)

# JavaScript to apply the theme
st.components.v1.html(f"""
<script>
    const streamlitDoc = parent.document;
    const theme = {'true' if st.session_state.dark_mode else 'false'} ? 'dark' : 'light';
    streamlitDoc.documentElement.setAttribute('data-theme', theme);
</script>
""", height=0)


# --- Header ---
title_col, button_col = st.columns([3, 1])
with title_col:
    st.markdown('<div class="innomatics-logo">', unsafe_allow_html=True)
    st.image(LOGO_URL, width=250)
    st.markdown('</div>', unsafe_allow_html=True)
    st.title("Placement Team Dashboard")

with button_col:
    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        if st.button("🧹 Clear", key="clear_button"):
            st.session_state.jd_text_key = ""
            st.session_state.file_uploader_key = str(datetime.now().timestamp())
            st.rerun()
    with sub_col2:
        st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")


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
                                    st.markdown(style_verdict(final_report.verdict), unsafe_allow_html=True)
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
        st.markdown(f"**Verdict:** {record['verdict']}")
        
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
                    row_col1, row_col2, row_col3 = st.columns([4, 1, 1])
                    with row_col1:
                        st.markdown(f"**{row['resume_filename']}**")
                        st.markdown(f"Score: **{row['score']}%** | Verdict: **{row['verdict']}**")
                    with row_col2:
                        if st.button("View Details", key=f"view_{row['id']}", use_container_width=True):
                            show_report_modal(row['id'])
                    with row_col3:
                        if st.button("Delete", key=f"delete_{row['id']}", type="secondary", use_container_width=True):
                            delete_analysis_from_db(row['id'])
                            st.success(f"Deleted record for {row['resume_filename']}.")
                            st.rerun()
        else:
            st.info("No records match the current filter criteria.")

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** for the Code4EdTech Challenge by Innomatics Research Labs.")
