import streamlit as st
import os
import sqlite3
import pandas as pd
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from typing import List

# --- Core AI/ML Libraries ---
import spacy
from spacy.matcher import PhraseMatcher
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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
    'spaCy', 'NLTK', 'API', 'REST', 'GraphQL', 'Microservices', 'System Design', 'Big Data', 'Hadoop', 'Spark'
]

# --- Pydantic Models for Structured LLM Output ---
class FinalAnalysis(BaseModel):
    relevance_score: int = Field(description="The final relevance score from 0 to 100.")
    verdict: str = Field(description="A verdict of 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_skills: List[str] = Field(description="A list of the most critical skills or qualifications explicitly mentioned in the job description but completely missing from the resume.")
    candidate_feedback: str = Field(description="A concise, professional, and actionable feedback paragraph for the candidate.")

    @validator('relevance_score')
    def score_must_be_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Relevance score must be between 0 and 100')
        return v

# --- Cached Model Loading ---
@st.cache_resource
def load_models():
    """Loads and caches all necessary AI/ML models."""
    with st.spinner("Warming up AI models... This may take a moment."):
        nlp = spacy.load("en_core_web_sm")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return nlp, embedding_model

# --- Database Management ---
@st.cache_resource
def get_db_connection():
    """Creates a cached, singleton connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
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
    df = pd.read_sql_query("SELECT id, timestamp, resume_filename, jd_summary, score, verdict FROM results ORDER BY id DESC", conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    return df

def delete_analysis_from_db(record_id):
    """Deletes a specific analysis record by its ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM results WHERE id = ?", (record_id,))
    conn.commit()

# --- Core Analysis Functions ---
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

def extract_skills_with_spacy(nlp, text):
    """Uses spaCy's PhraseMatcher for efficient keyword extraction."""
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill) for skill in SKILL_KEYWORDS]
    matcher.add("SKILL_MATCHER", patterns)
    doc = nlp(text)
    return list(set(doc[start:end].text.title() for _, start, end in matcher(doc)))

def get_llm_analysis(jd_text, resume_text, matched_skills):
    """Orchestrates the LLM call for a comprehensive analysis."""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
    
    prompt_template = """
    As an expert HR Technology Analyst, your task is to provide a rigorous, data-driven analysis of a resume against a job description.

    **CONTEXT:**
    1.  **Job Description (JD):**
        ```
        {jd}
        ```
    2.  **Resume Content:**
        ```
        {resume}
        ```
    3.  **Keywords Found via Hard Match:** {skills}

    **YOUR TASK (Follow these steps precisely):**

    1.  **Identify Core Requirements:** Systematically analyze the JD to identify the 5-7 most critical skills, technologies, and experience qualifications. These are the primary evaluation criteria.
    
    2.  **Evidence-Based Skill Gap Analysis:** Compare the core requirements from the JD against the **entire Resume Content**. The "Keywords Found" list is a guide, but your primary analysis must be on the full resume text. Identify which core requirements are genuinely **missing** or only weakly implied in the resume.
    
    3.  **Calculate Relevance Score (0-100):**
        - **75+ (High Suitability):** The resume strongly demonstrates a majority of the core requirements with explicit project experience or detailed descriptions.
        - **45-75 (Medium Suitability):** The resume mentions several core requirements but may lack depth, project examples, or is missing a few key skills.
        - **<45 (Low Suitability):** The resume is missing most of the core requirements and shows a significant mismatch for the role.
        Your final score must be logically justified by your skill gap analysis.

    4.  **Write Personalized Feedback:** Create a professional, encouraging feedback paragraph for the candidate. Start by acknowledging a strength, then clearly state 2-3 of the most critical missing skills they should focus on highlighting or acquiring to be a stronger candidate for this *specific type of role*.

    **OUTPUT FORMAT:**
    You MUST format your entire response as a single, valid JSON object that adheres to the following structure. Do not add any text before or after the JSON object.
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["jd", "resume", "skills"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | model | parser
    return chain.invoke({
        "jd": jd_text,
        "resume": resume_text,
        "skills": ", ".join(matched_skills) if matched_skills else "None"
    })

# --- UI Layout and Pages ---

def main_page():
    """The main application page for running analysis."""
    # --- HEADER ---
    st.image("https://www.innomatics.in/wp-content/uploads/2023/01/Innomatics-Logo1.png", width=250)
    st.title("Automated Resume Relevance Checker")
    st.markdown("---")

    # --- ANALYSIS INPUTS ---
    st.header("1. Input Job and Resume Data")
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            jd_text = st.text_area("Paste the full Job Description here:", height=300, key="jd_input")
        with col2:
            uploaded_files = st.file_uploader("Upload one or more resumes (PDF, DOCX):", type=["pdf", "docx"], accept_multiple_files=True)

    # --- RUN ANALYSIS ---
    if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
        if not jd_text or not uploaded_files:
            st.error("Please provide both a Job Description and at least one resume.")
            return
        
        nlp, _ = load_models()
        required_skills = set(extract_skills_with_spacy(nlp, jd_text))
        
        st.info(f"**Identified Key Skills in JD:** {', '.join(required_skills) if required_skills else 'None detected'}")
        
        progress_bar = st.progress(0, text="Starting...")
        
        for i, resume_file in enumerate(uploaded_files):
            progress_bar.progress((i) / len(uploaded_files), text=f"Analyzing {resume_file.name}...")
            
            resume_text = get_file_text(resume_file)
            if not resume_text:
                continue

            with st.spinner(f"AI is synthesizing the report for {resume_file.name}..."):
                matched_skills = extract_skills_with_spacy(nlp, resume_text)
                
                try:
                    final_report = get_llm_analysis(jd_text, resume_text, matched_skills)
                    add_analysis_to_db(resume_file.name, jd_text, final_report)

                    # --- DISPLAY RESULTS for each resume ---
                    with st.container(border=True):
                        st.subheader(f"Analysis Result: {resume_file.name}")
                        res_col1, res_col2 = st.columns([1, 2])
                        with res_col1:
                            st.metric("Relevance Score", f"{final_report.relevance_score}%")
                            st.markdown(f"**Verdict:** {final_report.verdict}")
                        with res_col2:
                            st.warning(f"**Missing Skills:** {', '.join(final_report.missing_skills) if final_report.missing_skills else 'No critical skills appear to be missing.'}")
                        st.info(f"**Candidate Feedback:** {final_report.candidate_feedback}")

                except Exception as e:
                    st.error(f"Failed to analyze {resume_file.name}. Error: {e}")
        
        progress_bar.progress(1.0, text="Analysis complete!")
        st.success("All resumes have been analyzed successfully!")
        st.balloons()


def dashboard_page():
    """The dashboard page to view and manage past results."""
    st.title("🗂️ Analysis History Dashboard")
    
    df = load_data_for_dashboard()
    if df.empty:
        st.info("No results found. Run a new analysis to see the dashboard.")
        return

    st.subheader("Filter & Manage Results")
    filt_col1, filt_col2, filt_col3 = st.columns(3)
    
    with filt_col1:
        jd_options = ["All JDs"] + list(df['jd_summary'].unique())
        selected_jd = st.selectbox("Filter by Job:", options=jd_options)
    
    df_filtered = df[df['jd_summary'] == selected_jd] if selected_jd != "All JDs" else df
    
    with filt_col2:
        verdict_options = ["All Verdicts"] + list(df_filtered['verdict'].unique())
        selected_verdict = st.selectbox("Filter by Verdict:", options=verdict_options)

    df_filtered = df_filtered[df_filtered['verdict'] == selected_verdict] if selected_verdict != "All Verdicts" else df_filtered

    with filt_col3:
        min_score = st.slider("Minimum Score:", 0, 100, 0)
    
    final_df = df_filtered[df_filtered['score'] >= min_score].reset_index(drop=True)
    
    st.subheader(f"Displaying {len(final_df)} Results")
    st.dataframe(final_df.style.apply(highlight_verdict, axis=1), hide_index=True, use_container_width=True)
    
    if not final_df.empty:
        st.markdown("---")
        st.subheader("Delete a Record")
        delete_options = [f"{row['id']} - {row['resume_filename']} ({row['jd_summary']})" for _, row in final_df.iterrows()]
        record_to_delete_display = st.selectbox("Select a record to delete:", options=[""] + delete_options)
        
        if st.button("❌ Delete Selected Record", type="secondary") and record_to_delete_display:
            record_id_to_delete = int(record_to_delete_display.split(" - ")[0])
            delete_analysis_from_db(record_id_to_delete)
            st.success(f"Record ID {record_id_to_delete} has been deleted.")
            st.rerun()

# --- Main App Execution ---
if 'page' not in st.session_state:
    st.session_state.page = "Analyze"

st.sidebar.title("Navigation")
if st.sidebar.button("📊 Analyze New Resumes", use_container_width=True):
    st.session_state.page = "Analyze"
if st.sidebar.button("🗂️ View Dashboard", use_container_width=True):
    st.session_state.page = "Dashboard"

# Initializing models on first run
load_models()

if st.session_state.page == "Analyze":
    analysis_page()
else:
    dashboard_page()
