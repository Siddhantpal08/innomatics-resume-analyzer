import streamlit as st
import sqlite3
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import re
import os
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

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('resume_analysis.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            jd_text TEXT,
            resume_filename TEXT,
            relevance_score INTEGER,
            verdict TEXT,
            missing_skills TEXT,
            candidate_feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Helper Functions ---
def get_resume_text(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
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

# --- Streamlit UI ---
st.set_page_config(page_title="🤖 Automated Resume Relevance Checker", layout="wide")

# --- Initialize Session State ---
if 'selected_record_id' not in st.session_state:
    st.session_state.selected_record_id = None

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analyze Resumes", "View Analysis History"])
st.sidebar.markdown("---")
st.sidebar.write("Developed for the Innomatics Code4EdTech Hackathon.")

# --- Main App Page ---
def main_app():
    st.title("🤖 Automated Resume Relevance Checker")
    st.markdown("##### Upload a Job Description and one or more resumes to get an AI-powered relevance analysis.")
    st.markdown("---")
    
    jd_text = st.text_area("Paste the Job Description here:", height=200, key="jd_text_main")
    uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Analyze Resumes", type="primary"):
        if not jd_text or not uploaded_files:
            st.warning("Please provide both a Job Description and at least one resume.")
            return
        if not GOOGLE_API_KEY:
            st.error("Google API Key not found. Please set it in your secrets.")
            return

        progress_bar = st.progress(0, text="Starting Analysis...")
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress((i) / total_files, text=f"Analyzing {uploaded_file.name}...")
            with st.container():
                try:
                    resume_text = get_resume_text(uploaded_file)
                    if not resume_text:
                        st.error(f"Could not extract text from {uploaded_file.name}. Skipping.")
                        continue
                    
                    # 1. Hard Skill Match
                    matched_skills = get_hard_skills_matches(resume_text)

                    # 2. Semantic Match & LLM Analysis
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
                    parser = JsonOutputParser(pydantic_object=AnalysisResult)

                    # --- IMPROVED PROMPT ---
                    prompt_template = """
                    You are an expert HR Technology Analyst. Your task is to provide a precise, data-driven analysis of a resume against a job description.

                    CONTEXT:
                    - Job Description: {jd}
                    - Resume Text: {resume}
                    - Pre-matched Hard Skills: {skills}

                    INSTRUCTIONS: Follow these steps precisely.
                    1.  **Analyze JD Requirements:** First, identify the top 5-7 most important technical skills and qualifications explicitly required by the job description.
                    2.  **Compare and Identify Missing Skills:** Compare the list of required skills from the JD against the ENTIRE resume content and the provided "Pre-matched Hard Skills". List only the critical skills that are genuinely missing from the resume. Do not infer skills that aren't present.
                    3.  **Calculate Relevance Score:** Based on the comparison, determine a score from 0-100. The score should reflect the alignment of skills, experience, and qualifications. A high score requires strong evidence of multiple key skills from the JD.
                    4.  **Give a Verdict:** Provide a verdict based on the score: 'High Suitability' (score > 75), 'Medium Suitability' (score 45-75), or 'Low Suitability' (score < 45).
                    5.  **Generate Candidate Feedback:** Write a brief, professional paragraph for the candidate. Acknowledge their strengths and then suggest specific ways to improve their resume for this role, mentioning 2-3 key missing skills they should focus on acquiring or highlighting.

                    Format your entire output as a single JSON object with these exact keys: "relevance_score", "verdict", "missing_skills", "candidate_feedback".
                    {format_instructions}
                    """
                    
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["jd", "resume", "skills"],
                        partial_variables={"format_instructions": parser.get_format_instructions()}
                    )

                    chain = prompt | llm | parser
                    
                    analysis = chain.invoke({
                        "jd": jd_text,
                        "resume": resume_text,
                        "skills": ", ".join(matched_skills)
                    })

                    # Display results
                    st.subheader(f"Analysis for: {uploaded_file.name}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Relevance Score", f"{analysis['relevance_score']}%")
                    with col2:
                        st.metric("Verdict", analysis['verdict'])
                    
                    with st.expander("View Details"):
                        st.write("**Missing Skills Identified:**")
                        st.warning(", ".join(analysis['missing_skills']) if analysis['missing_skills'] else "No critical skills appear to be missing.")
                        st.write("**Personalized Feedback for Candidate:**")
                        st.info(analysis['candidate_feedback'])
                    
                    # Save to DB
                    conn = sqlite3.connect('resume_analysis.db')
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO analysis_results (jd_text, resume_filename, relevance_score, verdict, missing_skills, candidate_feedback)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (jd_text, uploaded_file.name, analysis['relevance_score'], analysis['verdict'], ", ".join(analysis['missing_skills']), analysis['candidate_feedback']))
                    conn.commit()
                    conn.close()

                except Exception as e:
                    st.error(f"An error occurred while analyzing {uploaded_file.name}: {str(e)}")
                
                finally:
                    st.markdown("---")
        
        progress_bar.progress(1.0, text="Analysis Complete!")
        st.success("Analysis complete for all resumes!")

# --- History Page ---
def history_page():
    st.title("📄 Analysis History")
    
    conn = sqlite3.connect('resume_analysis.db')
    try:
        df = pd.read_sql_query("SELECT id, resume_filename, relevance_score, verdict FROM analysis_results ORDER BY id DESC", conn)
    except pd.io.sql.DatabaseError:
        st.warning("No analysis history found.")
        df = pd.DataFrame()
    conn.close()

    if not df.empty:
        # --- Filters ---
        st.sidebar.subheader("Filter History")
        # To filter by JD, we need to load the full data first.
        full_df = pd.read_sql_query("SELECT * FROM analysis_results", conn)
        jd_options = ["All"] + list(full_df['jd_text'].unique())
        selected_jd = st.sidebar.selectbox("Filter by Job Description", options=jd_options)
        
        # Apply JD filter if selected
        if selected_jd != "All":
            df = pd.read_sql_query(f"SELECT id, resume_filename, relevance_score, verdict FROM analysis_results WHERE jd_text = ? ORDER BY id DESC", conn, params=(selected_jd,))
        
        verdict_options = ["All"] + list(df['verdict'].unique())
        selected_verdict = st.sidebar.selectbox("Filter by Verdict", options=verdict_options)

        score_range = st.sidebar.slider("Filter by Score", 0, 100, (0, 100))
        
        # Apply other filters
        filtered_df = df[
            (df['verdict'] == selected_verdict if selected_verdict != "All" else True) &
            (df['relevance_score'] >= score_range[0]) &
            (df['relevance_score'] <= score_range[1])
        ]

        st.write(f"Displaying {len(filtered_df)} of {len(df)} records.")
        
        # --- Display Table Header ---
        header_cols = st.columns([1, 4, 2, 2, 2])
        header_cols[0].write("**ID**")
        header_cols[1].write("**Filename**")
        header_cols[2].write("**Score**")
        header_cols[3].write("**Verdict**")
        header_cols[4].write("**Actions**")
        st.markdown("---")

        # --- Display Table Rows ---
        for index, row in filtered_df.iterrows():
            row_cols = st.columns([1, 4, 2, 2, 2])
            row_cols[0].write(str(row['id']))
            row_cols[1].write(row['resume_filename'])
            row_cols[2].write(f"{row['relevance_score']}%")
            row_cols[3].write(row['verdict'])
            
            # Action buttons
            if row_cols[4].button("View Details", key=f"details_{row['id']}"):
                st.session_state.selected_record_id = row['id']
            if row_cols[4].button("Delete", key=f"delete_{row['id']}"):
                conn = sqlite3.connect('resume_analysis.db')
                c = conn.cursor()
                c.execute("DELETE FROM analysis_results WHERE id = ?", (row['id'],))
                conn.commit()
                conn.close()
                st.success(f"Deleted record {row['id']}.")
                st.rerun()
            
            st.markdown("---")

    else:
        st.info("No analysis has been performed yet. Go to 'Analyze Resumes' to get started.")

    # --- "Modal" for Viewing Details ---
    if st.session_state.selected_record_id is not None:
        conn = sqlite3.connect('resume_analysis.db')
        record = pd.read_sql_query("SELECT * FROM analysis_results WHERE id = ?", conn, params=(st.session_state.selected_record_id,)).iloc[0]
        conn.close()

        with st.container(border=True):
            st.subheader(f"Details for {record['resume_filename']} (ID: {record['id']})")
            st.text_area("Job Description", value=record['jd_text'], height=150, disabled=True, key=f"jd_modal_{record['id']}")
            st.write(f"**Missing Skills:** {record['missing_skills']}")
            st.info(f"**Candidate Feedback:** {record['candidate_feedback']}")
            if st.button("Close Details"):
                st.session_state.selected_record_id = None
                st.rerun()

# --- Page Routing ---
if page == "Analyze Resumes":
    main_app()
else:
    history_page()
