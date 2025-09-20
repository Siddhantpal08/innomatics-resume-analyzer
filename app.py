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

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Global Configurations ---
DB_FILE = "analysis_results.db"
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
    return conn

def init_database():
    """Initializes the database table, adding the missing_skills column if it doesn't exist."""
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
            feedback TEXT,
            full_jd TEXT
        )
    ''')
    # Check if 'missing_skills' column exists and add it if it doesn't
    c.execute("PRAGMA table_info(results)")
    columns = [info[1] for info in c.fetchall()]
    if 'missing_skills' not in columns:
        c.execute("ALTER TABLE results ADD COLUMN missing_skills TEXT")

    conn.commit()

init_database()

def add_analysis_to_db(filename, jd_text, report: FinalAnalysis):
    """Adds a new analysis report to the database."""
    conn = get_db_connection()
    c = conn.cursor()
    jd_summary = " ".join(jd_text.split()[:10]).strip() + "..."
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

def highlight_verdict(row):
    verdict = row['verdict']
    color_map = {
        'High Suitability': 'background-color: #d4edda; color: #155724;',
        'Medium Suitability': 'background-color: #fff3cd; color: #856404;',
        'Low Suitability': 'background-color: #f8d7da; color: #721c24;'
    }
    style = color_map.get(verdict, '')
    return [style] * len(row)
    
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
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide")

# --- Dark Mode CSS & JS ---
# Session state initialization must happen before widgets are created
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 'initial'
if 'jd_text_key' not in st.session_state:
    st.session_state.jd_text_key = ''

# JavaScript to toggle the theme attribute on the parent document's body
theme_js = f"""
<script>
    function applyTheme(isDarkMode) {{
        const parentBody = parent.document.body;
        if (parentBody) {{
            parentBody.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
        }}
    }}
    applyTheme({str(st.session_state.dark_mode).lower()});
</script>
"""
st.components.v1.html(theme_js, height=0)

# CSS that uses the data-theme attribute and inverts the logo
st.markdown("""
<style>
    body[data-theme="dark"] .innomatics-logo img {
        filter: invert(1) hue-rotate(180deg);
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
title_col, button_col = st.columns([3, 1])
with title_col:
    st.markdown('<div class="innomatics-logo">', unsafe_allow_html=True) # Class for CSS targeting
    st.image("https://www.innomatics.in/wp-content/uploads/2023/01/Innomatics-Logo1.png", width=250)
    st.markdown('</div>', unsafe_allow_html=True)
    st.title("Placement Team Dashboard")

with button_col:
    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True) # Vertical Spacer
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        if st.button("🧹 Clear", key="clear_button"):
            st.session_state.jd_text_key = ""
            st.session_state.file_uploader_key = str(datetime.now().timestamp())
            st.rerun()
    with sub_col2:
        # The on_change callback ensures the state is set before the script reruns
        st.toggle("🌙 Dark", value=st.session_state.dark_mode, key="dark_mode_toggle")


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
    st.write("") # Spacer

    if st.button("🚀 Run Full Analysis", type="primary", key="analysis_button", use_container_width=True):
        if not jd_text.strip() or not uploaded_files:
            st.error("Please provide both a Job Description and at least one resume.")
        elif not GOOGLE_API_KEY:
            st.error("Google API Key not found. Please set it in your environment variables/secrets.")
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

            progress_bar.progress(1.0, text="All analyses complete!")
            st.success("All analyses complete!")
            st.balloons()

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
            st.dataframe(
                final_df[['id', 'timestamp', 'resume_filename', 'jd_summary', 'score', 'verdict', 'missing_skills']].style.apply(highlight_verdict, axis=1),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "timestamp": st.column_config.TextColumn("Time"),
                    "resume_filename": st.column_config.TextColumn("Resume"),
                    "jd_summary": st.column_config.TextColumn("JD"),
                    "score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                    "verdict": st.column_config.TextColumn("Verdict"),
                    "missing_skills": st.column_config.TextColumn("Missing Skills", width="large"),
                }
            )
            
            st.markdown("---")
            st.subheader("Delete a Record")
            delete_options = [f"ID {row['id']} - {row['resume_filename']} ({row['jd_summary']})" for _, row in final_df.iterrows()]
            record_to_delete_display = st.selectbox("Select a record to delete:", options=[""] + delete_options)
            
            if st.button("❌ Delete Selected Record", type="secondary") and record_to_delete_display:
                record_id_to_delete = int(record_to_delete_display.split(" - ")[0].replace("ID ", ""))
                delete_analysis_from_db(record_id_to_delete)
                st.success(f"Record ID {record_id_to_delete} has been deleted.")
                st.rerun()
        else:
            st.info("No records match the current filter criteria.")

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** for the Code4EdTech Challenge by Innomatics Research Labs.")
