import streamlit as st
import os
import sqlite3
import pandas as pd
import re
import hashlib
import uuid
from datetime import datetime
from PyPDF2 import PdfReader
from docx import Document
from typing import List, Optional

# --- Optional / fallback libraries ---
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# --- LLM & helpers ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="Innomatics Resume Analyzer — Improved", layout="wide", initial_sidebar_state="expanded")

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- DB & App Config ---
DB_FILE = "analysis_results.db"
LOGO_URL = "https://www.innomatics.in/wp-content/uploads/2023/01/Innomatics-Logo1.png"

# --- Pydantic Models ---
class JDSkills(BaseModel):
    job_title: str
    hard_skills: List[str]
    experience_years: Optional[str] = None

class ResumeSkills(BaseModel):
    demonstrated_skills: List[str]
    listed_skills: List[str]

class FinalAnalysis(BaseModel):
    relevance_score: int = Field(..., description="0-100")
    verdict: str
    missing_skills: List[str]
    candidate_feedback: str

    @validator('relevance_score')
    def score_must_be_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Relevance score must be between 0 and 100')
        return v

# --- Database management with migration support ---
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_db_connection()
    c = conn.cursor()
    # Create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            resume_filename TEXT NOT NULL,
            jd_title TEXT NOT NULL,
            jd_summary TEXT NOT NULL,
            jd_hash TEXT NOT NULL,
            score INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            missing_skills TEXT,
            feedback TEXT,
            full_jd TEXT
        )
    ''')
    conn.commit()

init_database()

# --- Utility helpers ---
def compute_jd_hash(jd_text: str) -> str:
    return hashlib.sha256(jd_text.encode('utf-8')).hexdigest()

def check_if_exists(filename, jd_hash):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM results WHERE resume_filename = ? AND jd_hash = ?", (filename, jd_hash))
    return c.fetchone() is not None

def add_analysis_to_db(filename, jd_title, jd_text, report: FinalAnalysis):
    conn = get_db_connection()
    c = conn.cursor()
    jd_summary = " ".join(jd_text.split()[:15]).strip() + "..."
    jd_hash = compute_jd_hash(jd_text)
    c.execute('''
        INSERT INTO results (timestamp, resume_filename, jd_title, jd_summary, jd_hash, score, verdict, missing_skills, feedback, full_jd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now(), filename, jd_title, jd_summary, jd_hash, report.relevance_score,
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

# --- Text extraction & parsing ---
@st.cache_data
def get_file_text(uploaded_file) -> str:
    text = ""
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith('.pdf'):
            # Prefer pdfplumber when available
            if HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(uploaded_file) as pdf:
                        pages = [p.extract_text() or "" for p in pdf.pages]
                        text = "\n".join(pages)
                except Exception:
                    # Fallback to PyPDF2
                    pdf_reader = PdfReader(uploaded_file)
                    pages = [p.extract_text() or "" for p in pdf_reader.pages]
                    text = "\n".join(pages)
            else:
                pdf_reader = PdfReader(uploaded_file)
                pages = [p.extract_text() or "" for p in pdf_reader.pages]
                text = "\n".join(pages)
        elif fname.endswith('.docx'):
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""
    return text

def extract_contact_info(text: str):
    emails = re.findall(r"[\w\.-]+@[\w\.-]+", text)
    phones = re.findall(r"\+?\d[\d\-\s]{7,}\d", text)
    return (list(dict.fromkeys(emails)), list(dict.fromkeys(phones)))

# heuristic skill section extractor
def heuristic_extract_skills(text: str, max_items=30):
    text_lower = text.lower()
    # Look for common headings
    headings = ['skills', 'technical skills', 'skill set', 'technical summary', 'skills & technologies']
    for h in headings:
        idx = text_lower.find(h)
        if idx != -1:
            snippet = text[idx: idx + 600]
            # split by line or bullets
            items = re.split(r"[\n\r•\-\*]+", snippet)
            # the first element is header; drop it
            items = items[1:]
            skills = []
            for it in items:
                # split by commas
                for part in re.split(r"[,;]\s*", it):
                    candidate = re.sub(r"[^A-Za-z0-9\+\#\s\.\-]", '', part).strip()
                    if 2 <= len(candidate) <= 60:
                        skills.append(candidate)
                    if len(skills) >= max_items:
                        break
                if len(skills) >= max_items:
                    break
            if skills:
                return [s for s in skills if s]
    # fallback: collect capitalized tokens (very rough)
    tokens = re.findall(r"\b[A-Za-z0-9\+\#]{2,30}\b", text)
    return list(dict.fromkeys(tokens))[:max_items]

# --- LLM structured extraction with fallbacks ---
@st.cache_data
def llm_extract_structured(jd_text: str, resume_text: str):
    # Try to use the langchain + Google model to get structured outputs; if that fails, fallback to heuristics
    jd_skills = None
    resume_skills = None

    # Build parsers
    jd_parser = PydanticOutputParser(pydantic_object=JDSkills)
    resume_parser = PydanticOutputParser(pydantic_object=ResumeSkills)

    # Build prompts
    jd_prompt = PromptTemplate(template="Extract the key skills and job title from this job description.\n{format_instructions}\nJD:\n{jd}", input_variables=["jd"], partial_variables={"format_instructions": jd_parser.get_format_instructions()})
    resume_prompt = PromptTemplate(template="Extract the key skills from this resume, separating skills listed in a skills section from those demonstrated in work experience.\n{format_instructions}\nResume:\n{resume}", input_variables=["resume"], partial_variables={"format_instructions": resume_parser.get_format_instructions()})

    if GOOGLE_API_KEY:
        try:
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)
            jd_chain = jd_prompt | model | jd_parser
            resume_chain = resume_prompt | model | resume_parser
            jd_skills = jd_chain.invoke({"jd": jd_text})
            resume_skills = resume_chain.invoke({"resume": resume_text})
        except Exception as e:
            # We'll fallback
            jd_skills = None
            resume_skills = None

    # Fallback heuristics when LLM not available or failed
    if not jd_skills:
        # Attempt to parse title as first line + skills heuristics
        lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
        title = lines[0] if lines else "Job"
        # Extract potential skill tokens using comma heuristics from first 2 paragraphs
        paragraphs = jd_text.split('\n\n')
        top_paras = ' '.join(paragraphs[:2])
        # split by commas and common separators
        tokens = re.split(r"[,;\n\r\t\|\/\\]+", top_paras)
        candidate_skills = []
        for t in tokens:
            t_clean = re.sub(r"[^A-Za-z0-9\+\#\s\.\-]", '', t).strip()
            if len(t_clean) > 2 and len(t_clean.split()) <= 4:
                candidate_skills.append(t_clean)
        jd_skills = JDSkills(job_title=title, hard_skills=candidate_skills[:8], experience_years=None)

    if not resume_skills:
        # Heuristic skills extraction
        demonstrated = heuristic_extract_skills(resume_text, max_items=20)
        listed = demonstrated[:20]
        resume_skills = ResumeSkills(demonstrated_skills=demonstrated, listed_skills=listed)

    return jd_skills, resume_skills

# --- Matching & scoring ---

def normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9\+\#\.\-]", ' ', s.lower()).strip()

def fuzzy_match(a: str, b: str) -> int:
    a_n, b_n = a.lower(), b.lower()
    if HAS_RAPIDFUZZ:
        return fuzz.token_sort_ratio(a_n, b_n)
    else:
        # simple fallback: partial substring match scaled to 0/100
        return 100 if (a_n in b_n or b_n in a_n) else 0

def compute_scores(jd_skills: JDSkills, resume_skills: ResumeSkills, jd_text: str, resume_text: str):
    jd_list = [normalize_skill(s) for s in jd_skills.hard_skills]
    demo = [normalize_skill(s) for s in resume_skills.demonstrated_skills]
    listed = [normalize_skill(s) for s in resume_skills.listed_skills]

    demonstrated_count = 0
    listed_count = 0
    missing = []

    for req in jd_list:
        matched = False
        # exact or fuzzy in demonstrated
        for s in demo:
            score = fuzzy_match(req, s)
            if score >= 85:
                demonstrated_count += 1
                matched = True
                break
        if matched:
            continue
        # check in listed
        for s in listed:
            score = fuzzy_match(req, s)
            if score >= 80:
                listed_count += 1
                matched = True
                break
        if not matched:
            missing.append(req)

    # Hard score: demonstrated stronger than listed
    hard_score = min(60, demonstrated_count * 12 + listed_count * 5)

    # Experience bonus: try to extract numbers
    exp_bonus = 0
    if jd_skills.experience_years:
        nums = re.findall(r"(\d+)", jd_skills.experience_years)
        if nums:
            required = int(nums[0])
            # look for similar numbers in resume text
            rnums = re.findall(r"(\d+)\+?\s*(?:years|yrs|y)", resume_text.lower())
            candidate_years = int(rnums[0]) if rnums else 0
            if candidate_years >= required:
                exp_bonus = 8

    # Semantic similarity (TF-IDF)
    sem_score = 0
    if HAS_SKLEARN and jd_text and resume_text:
        try:
            vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=2000)
            X = vec.fit_transform([jd_text, resume_text])
            sim = cosine_similarity(X[0:1], X[1:2])[0][0]
            sem_score = int(sim * 30)  # scale to 0..30
        except Exception:
            sem_score = 0

    total = hard_score + sem_score + exp_bonus
    total = min(100, int(round(total)))

    # Verdict
    if total >= 70:
        verdict = 'High Suitability'
    elif total >= 40:
        verdict = 'Medium Suitability'
    else:
        verdict = 'Low Suitability'

    # Top missing (presentable form)
    missing_presentable = [m for m in missing][:5]

    return total, verdict, missing_presentable, demonstrated_count, listed_count, hard_score, sem_score, exp_bonus

# --- Feedback generation via LLM (fallback to template) ---
def generate_feedback_via_llm(job_title, matched_count, missing_skills, strengths_summary):
    if GOOGLE_API_KEY:
        try:
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
            prompt = PromptTemplate(template=(
                "You are a helpful career coach. Given a job title, a short strengths summary, and a list of top missing skills,"
                " write a concise, professional, actionable feedback paragraph (2-3 sentences) that starts with something positive, then mentions the main gaps and suggests 1-2 concrete steps the candidate can take (projects, keywords to add, certifications or learning resources)."
                "\n\nJob Title: {job_title}\nMatched Skills Count: {matched_count}\nStrengths: {strengths}\nMissing Skills: {missing}\n\nKeep the feedback under 80 words."
            ), input_variables=["job_title","matched_count","strengths","missing"])
            chain = prompt | model
            out = chain.invoke({"job_title": job_title, "matched_count": matched_count, "strengths": strengths_summary, "missing": ", ".join(missing_skills)})
            return out.strip()
        except Exception:
            pass
    # fallback template
    positive = "Good experience in the areas you have demonstrated."
    if missing_skills:
        suggestions = f"To improve your fit, consider working on: {', '.join(missing_skills[:3])}. Build small projects showcasing these skills and mention them explicitly in your experience section."
    else:
        suggestions = "Your resume already matches the critical skills — highlight measurable outcomes and projects to further strengthen it."
    return positive + " " + suggestions

# --- UI helpers ---
def get_verdict_color(verdict):
    if verdict == 'High Suitability': return '#28a745'
    elif verdict == 'Medium Suitability': return '#ffc107'
    else: return '#dc3545'

# --- Session state keys for uploader reset ---
if 'uploader_version' not in st.session_state:
    st.session_state['uploader_version'] = 0

uploader_key = f"file_uploader_key_{st.session_state['uploader_version']}"

# --- App Layout ---
st.image(LOGO_URL, width=200)
st.title("Automated Resume Relevance Checker — Improved")
st.markdown("---")

analysis_tab, dashboard_tab = st.tabs(["📊 Analysis", "🗂️ Dashboard"])

with analysis_tab:
    st.header("1. Input Job and Resume Data")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            jd_text = st.text_area("Paste the full Job Description here:", height=300, key="jd_text_key", placeholder="e.g., 'Seeking a Python developer with 3+ years of experience in Django, REST APIs, and PostgreSQL...'", value=(st.session_state.get('jd_text_key', '') or ''))
            st.write("")
            # Mode & options
            opts_col1, opts_col2 = st.columns(2)
            with opts_col1:
                mode = st.selectbox("Scoring Mode:", ["Deterministic (recommended)", "LLM-only"], index=0, help="Deterministic uses local rules + optional TF-IDF semantic similarity for consistent results.")
                use_semantic = st.checkbox("Use TF-IDF semantic similarity (if available)", value=HAS_SKLEARN)
            with opts_col2:
                st.write("")
                if not HAS_SKLEARN:
                    st.caption("Tip: install scikit-learn to enable TF-IDF semantic similarity for improved accuracy.")
                if not HAS_RAPIDFUZZ:
                    st.caption("Tip: install rapidfuzz to enable fuzzy matching for better skill matching.")

        with col2:
            uploaded_files = st.file_uploader("Upload one or more resumes (PDF, DOCX):", type=["pdf", "docx"], accept_multiple_files=True, key=uploader_key)

    # Clear inputs button
    clear_col1, clear_col2 = st.columns([1,4])
    with clear_col1:
        if st.button("Clear Inputs"):
            # Reset JD text and bump uploader key to clear files
            st.session_state['jd_text_key'] = ""
            st.session_state['uploader_version'] += 1
            st.experimental_rerun()
    with clear_col2:
        st.info("Use 'Clear Inputs' to quickly reset the JD and uploaded files before re-running analysis.")

    st.write("")

    if st.button("🚀 Analyze Resumes", type="primary"):
        if not jd_text.strip() or not uploaded_files:
            st.error("Please provide both a Job Description and at least one resume.")
        else:
            jd_hash = compute_jd_hash(jd_text)
            files_to_process = []
            for file in uploaded_files:
                if check_if_exists(file.name, jd_hash):
                    st.warning(f"Skipping '{file.name}': This resume has already been analyzed for this job description.")
                else:
                    files_to_process.append(file)

            if not files_to_process:
                st.info("No new resumes to analyze.")
            else:
                progress = st.progress(0, text="Initializing...")
                results_for_export = []
                for i, resume_file in enumerate(files_to_process):
                    progress.progress(int((i / len(files_to_process)) * 100), text=f"Reading {resume_file.name}...")
                    resume_text = get_file_text(resume_file)
                    if not resume_text:
                        st.error(f"Could not extract text from {resume_file.name}. Skipping.")
                        continue

                    progress.progress(int(((i + 0.2) / len(files_to_process)) * 100), text="Extracting structured info...")
                    jd_struct, resume_struct = llm_extract_structured(jd_text, resume_text)

                    # If mode is Deterministic, compute local score + use LLM only for feedback (if available)
                    if mode.startswith('Deterministic'):
                        score, verdict, missing, demo_cnt, list_cnt, hard_score, sem_score, exp_bonus = compute_scores(jd_struct, resume_struct, jd_text if use_semantic else "", resume_text)
                        strengths = f"Demonstrated: {demo_cnt}, Listed: {list_cnt}"
                        feedback = generate_feedback_via_llm(jd_struct.job_title, demo_cnt + list_cnt, missing, strengths)
                        report = FinalAnalysis(relevance_score=int(score), verdict=verdict, missing_skills=[m for m in missing], candidate_feedback=feedback)
                    else:
                        # LLM-only: ask LLM to craft full analysis (fallback to previous approach) — keep deterministic as safe backup
                        try:
                            # Reuse earlier Pydantic-based chain to ask for final JSON
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
                            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)
                            analysis_chain = analysis_prompt | model | analysis_parser
                            final_report = analysis_chain.invoke({
                                "job_title": jd_struct.job_title,
                                "jd_hard_skills": jd_struct.hard_skills,
                                "jd_experience": jd_struct.experience_years or "Not Specified",
                                "resume_demonstrated": resume_struct.demonstrated_skills,
                                "resume_listed": resume_struct.listed_skills
                            })
                            report = final_report
                        except Exception as e:
                            # fallback to deterministic
                            score, verdict, missing, demo_cnt, list_cnt, hard_score, sem_score, exp_bonus = compute_scores(jd_struct, resume_struct, jd_text if use_semantic else "", resume_text)
                            strengths = f"Demonstrated: {demo_cnt}, Listed: {list_cnt}"
                            feedback = generate_feedback_via_llm(jd_struct.job_title, demo_cnt + list_cnt, missing, strengths)
                            report = FinalAnalysis(relevance_score=int(score), verdict=verdict, missing_skills=[m for m in missing], candidate_feedback=feedback)

                    # Save to DB & show
                    add_analysis_to_db(resume_file.name, jd_struct.job_title, jd_text, report)

                    progress.progress(int(((i + 0.8) / len(files_to_process)) * 100), text=f"Saving results for {resume_file.name}...")

                    # UI output for this candidate
                    with st.container():
                        st.markdown(f"### Candidate: {resume_file.name}")
                        col1, col2 = st.columns([1,3])
                        with col1:
                            st.markdown(f"<p style='color:{get_verdict_color(report.verdict)};'><strong>{report.verdict}</strong></p>", unsafe_allow_html=True)
                            st.metric("Score", f"{report.relevance_score}%")
                        with col2:
                            st.warning("**Identified Gaps:**")
                            if report.missing_skills:
                                st.markdown('\n'.join([f"- {item}" for item in report.missing_skills]))
                            else:
                                st.markdown("- None")
                            st.success("**Personalized Feedback:**")
                            st.write(report.candidate_feedback)

                    results_for_export.append({
                        'filename': resume_file.name,
                        'jd_title': jd_struct.job_title,
                        'score': report.relevance_score,
                        'verdict': report.verdict,
                        'missing_skills': ", ".join(report.missing_skills),
                        'feedback': report.candidate_feedback
                    })

                progress.progress(100, text="Done")
                st.success("All new resumes have been analyzed!")
                st.balloons()

                # Offer CSV export for this batch
                if results_for_export:
                    df_export = pd.DataFrame(results_for_export)
                    csv_bytes = df_export.to_csv(index=False).encode('utf-8')
                    st.download_button("Download batch results as CSV", data=csv_bytes, file_name=f"analysis_batch_{uuid.uuid4().hex[:8]}.csv", mime='text/csv')

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

        if not final_df.empty:
            for index, row in final_df.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([5,1,1])
                    with c1:
                        st.markdown(f"**{row['resume_filename']}** for **{row['jd_title']}**")
                        verdict_color = get_verdict_color(row['verdict'])
                        missing_skills_summary = row['missing_skills']
                        if len(missing_skills_summary) > 60:
                            missing_skills_summary = missing_skills_summary[:57] + "..."
                        st.markdown(f"Score: `{row['score']}%` | Verdict: <span style='color:{verdict_color};'>**{row['verdict']}**</span> | Gaps: *{missing_skills_summary}*", unsafe_allow_html=True)
                    with c2:
                        if st.button("Details", key=f"view_{row['id']}", use_container_width=True):
                            rec = get_single_record(row['id'])
                            st.subheader(f"Analysis for: {rec['resume_filename']}")
                            st.metric("Relevance Score", f"{rec['score']}%")
                            st.markdown(f"<p style='color:{get_verdict_color(rec['verdict'])};'><strong>Verdict: {rec['verdict']}</strong></p>", unsafe_allow_html=True)
                            st.markdown("---")
                            st.subheader("Identified Gaps")
                            st.markdown(rec['missing_skills'])
                            st.markdown("---")
                            st.subheader("Candidate Feedback")
                            st.info(rec['feedback'])
                            st.markdown("---")
                            with st.expander("Show Original Job Description"):
                                st.text_area("JD", value=rec['full_jd'], height=200, disabled=True, label_visibility="collapsed")
                    with c3:
                        if st.button("Delete", key=f"delete_{row['id']}", type="secondary", use_container_width=True):
                            delete_analysis_from_db(row['id'])
                            st.success(f"Deleted record for {row['resume_filename']}.")
                            st.experimental_rerun()

        # Dashboard actions
        st.markdown("---")
        db_col1, db_col2 = st.columns([1,1])
        with db_col1:
            if st.button("Export full DB as CSV"):
                conn = get_db_connection()
                df_all = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
                csv_bytes = df_all.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv_bytes, file_name=f"analysis_full_{uuid.uuid4().hex[:8]}.csv", mime='text/csv')
        with db_col2:
            if st.button("Clear entire DB (use with caution)"):
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("DELETE FROM results")
                conn.commit()
                st.success("All records deleted.")
                st.experimental_rerun()

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** — Improved version. Good luck with Code4EdTech!")
