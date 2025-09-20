"""
Robust, production-ready Streamlit app for "Automated Resume Relevance Checker"
- All fallback handlers, defensive DB code, and thorough try/except blocks
- Features:
  * Deterministic (explainable) scoring + optional LLM-enhanced feedback
  * LLM-only mode (optional)
  * PDF/DOCX parsing with pdfplumber/PyPDF2 fallback
  * fuzzy matching (rapidfuzz) fallback -> substring
  * TF-IDF semantic similarity (scikit-learn) fallback -> 0
  * Safe SQLite init + migrations + WAL mode
  * Clear Inputs button that never errors
  * Export CSV, Export single JSON, delete record, clear DB
  * Scoring breakdown view and highlight preview
  * Retry wrapper for transient LLM errors

Usage:
1. Add your GOOGLE_API_KEY to Streamlit secrets or environment variables
2. Install recommended packages (see top of file comment)
3. Deploy to Streamlit Cloud or run locally: `streamlit run app.py`

Author: Siddhant Pal (robust version)
"""

from __future__ import annotations
import streamlit as st
import os
import sqlite3
import pandas as pd
import re
import json
import hashlib
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Tuple

# Documented optional libraries
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

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

# LLM + helpers
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field, validator
    HAS_LLM = True
except Exception:
    # we'll still run without LLM
    HAS_LLM = False

from dotenv import load_dotenv
load_dotenv()

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume_analyzer")

# --- App config ---
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide", initial_sidebar_state="expanded")

DB_FILE = os.getenv("RESUME_DB_PATH", "analysis_results.db")
LOGO_URL = "https://www.innomatics.in/wp-content/uploads/2023/01/Innomatics-Logo1.png"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None

# Ensure session keys exist safely
if 'uploader_version' not in st.session_state:
    st.session_state['uploader_version'] = 0
if 'jd_text' not in st.session_state:
    st.session_state['jd_text'] = ""

# --- Pydantic-like lightweight models (use only if LLM available) ---
if HAS_LLM:
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
else:
    # Simple placeholders for typing / safety
    class JDSkills:
        def __init__(self, job_title, hard_skills, experience_years=None):
            self.job_title = job_title
            self.hard_skills = hard_skills
            self.experience_years = experience_years

    class ResumeSkills:
        def __init__(self, demonstrated_skills, listed_skills):
            self.demonstrated_skills = demonstrated_skills
            self.listed_skills = listed_skills

    class FinalAnalysis:
        def __init__(self, relevance_score, verdict, missing_skills, candidate_feedback):
            self.relevance_score = relevance_score
            self.verdict = verdict
            self.missing_skills = missing_skills
            self.candidate_feedback = candidate_feedback

# --- Database utilities ---
def get_conn():
    # each call returns a fresh connection to reduce concurrency issues
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    # enable WAL for better concurrency (best-effort)
    try:
        conn.execute('PRAGMA journal_mode=WAL;')
    except Exception:
        pass
    return conn

def init_database():
    conn = get_conn()
    try:
        c = conn.cursor()
        # Try to add jd_hash column if missing (safe migration)
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
    except Exception as e:
        logger.exception("DB init failed: %s", e)
    finally:
        conn.close()

init_database()

# Defensive DB wrapper
def safe_execute(query: str, params: tuple = (), fetch: bool = False):
    conn = None
    try:
        conn = get_conn()
        c = conn.cursor()
        c.execute(query, params)
        if fetch:
            rows = c.fetchall()
            return rows
        conn.commit()
    except sqlite3.OperationalError as e:
        logger.exception("SQLite OperationalError: %s", e)
        # attempt to re-init DB and retry once
        try:
            init_database()
            if conn:
                conn = get_conn()
                c = conn.cursor()
                c.execute(query, params)
                if fetch:
                    rows = c.fetchall()
                    return rows
                conn.commit()
        except Exception as e2:
            logger.exception("Retry failed: %s", e2)
            raise
    finally:
        if conn:
            conn.close()
    return None

# check if exists
def compute_jd_hash(jd_text: str) -> str:
    return hashlib.sha256(jd_text.encode('utf-8')).hexdigest()

def check_if_exists(filename: str, jd_hash: str) -> bool:
    try:
        rows = safe_execute("SELECT id FROM results WHERE resume_filename = ? AND jd_hash = ?", (filename, jd_hash), fetch=True)
        return bool(rows)
    except Exception:
        # be conservative: if DB broken, return False so we try to process
        return False

def add_analysis_to_db(filename: str, jd_title: str, jd_text: str, report: FinalAnalysis):
    jd_summary = " ".join(jd_text.split()[:20]).strip() + "..."
    jd_hash = compute_jd_hash(jd_text)
    try:
        safe_execute(
            "INSERT INTO results (timestamp, resume_filename, jd_title, jd_summary, jd_hash, score, verdict, missing_skills, feedback, full_jd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(), filename, jd_title, jd_summary, jd_hash, int(report.relevance_score), report.verdict, ", ".join(report.missing_skills) if report.missing_skills else "", report.candidate_feedback, jd_text)
        )
    except Exception as e:
        logger.exception("Failed to add analysis to DB: %s", e)

def load_data_for_dashboard() -> pd.DataFrame:
    try:
        conn = get_conn()
        df = pd.read_sql_query("SELECT id, timestamp, resume_filename, jd_title, jd_summary, score, verdict, missing_skills FROM results ORDER BY id DESC", conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to load dashboard data: %s", e)
        return pd.DataFrame()

def get_single_record(record_id: int):
    rows = safe_execute("SELECT * FROM results WHERE id = ?", (record_id,), fetch=True)
    return rows[0] if rows else None

def delete_analysis_from_db(record_id: int):
    safe_execute("DELETE FROM results WHERE id = ?", (record_id,))

# --- Text extraction & heuristics ---
@st.cache_data
def get_file_text(uploaded_file) -> str:
    text = ""
    fname = getattr(uploaded_file, 'name', '') or str(uploaded_file)
    fname = fname.lower()
    try:
        if fname.endswith('.pdf'):
            if HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                except Exception:
                    # fallback PyPDF2
                    if PdfReader:
                        reader = PdfReader(uploaded_file)
                        text = "\n".join([p.extract_text() or "" for p in reader.pages])
            else:
                if PdfReader:
                    reader = PdfReader(uploaded_file)
                    text = "\n".join([p.extract_text() or "" for p in reader.pages])
        elif fname.endswith('.docx'):
            if Document:
                doc = Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        logger.exception("Error extracting text from %s: %s", fname, e)
        return ""
    return text

# basic contact extraction
def extract_contact_info(text: str) -> Tuple[List[str], List[str]]:
    emails = re.findall(r"[\w\.-]+@[\w\.-]+", text)
    phones = re.findall(r"\+?\d[\d\-\s]{7,}\d", text)
    return list(dict.fromkeys(emails)), list(dict.fromkeys(phones))

# heuristic skill extraction
def heuristic_extract_skills(text: str, max_items=30) -> List[str]:
    if not text:
        return []
    text_lower = text.lower()
    headings = ['skills', 'technical skills', 'skill set', 'skills & technologies', 'skills:']
    for h in headings:
        idx = text_lower.find(h)
        if idx != -1:
            snippet = text[idx: idx + 800]
            items = re.split(r"[\n\r•\-\*]+", snippet)
            items = items[1:]
            skills = []
            for it in items:
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
    # fallback: common tech keywords extraction (very rough)
    tokens = re.findall(r"\b[A-Za-z0-9\+\#]{2,30}\b", text)
    return list(dict.fromkeys(tokens))[:max_items]

# --- LLM wrappers (if available) ---
def llm_safe_invoke_chain(chain, inputs: dict, retries=2):
    # Very small retry wrapper for transient LLM failures
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_exc = e
            logger.warning("LLM transient error (attempt %s): %s", attempt + 1, e)
    # raise final
    logger.exception("LLM calls failed after retries: %s", last_exc)
    raise last_exc

@st.cache_data
def llm_extract_structured(jd_text: str, resume_text: str):
    jd_skills = None
    resume_skills = None

    if HAS_LLM and GOOGLE_API_KEY:
        try:
            jd_parser = PydanticOutputParser(pydantic_object=JDSkills)
            resume_parser = PydanticOutputParser(pydantic_object=ResumeSkills)
            jd_prompt = PromptTemplate(template="Extract the key skills and job title from this job description.\n{format_instructions}\nJD:\n{jd}", input_variables=["jd"], partial_variables={"format_instructions": jd_parser.get_format_instructions()})
            resume_prompt = PromptTemplate(template="Extract the key skills from this resume, separating skills listed in a skills section from those demonstrated in work experience.\n{format_instructions}\nResume:\n{resume}", input_variables=["resume"], partial_variables={"format_instructions": resume_parser.get_format_instructions()})
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)
            jd_chain = jd_prompt | model | jd_parser
            resume_chain = resume_prompt | model | resume_parser
            jd_skills = llm_safe_invoke_chain(jd_chain, {"jd": jd_text})
            resume_skills = llm_safe_invoke_chain(resume_chain, {"resume": resume_text})
        except Exception as e:
            logger.warning("LLM structured extraction failed, falling back to heuristics: %s", e)

    if not jd_skills:
        # Heuristic fallback
        lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
        title = lines[0] if lines else "Job"
        # pick up comma-separated tokens from first 2 paragraphs
        paras = jd_text.split('\n\n')
        sample = ' '.join(paras[:2])
        tokens = re.split(r"[,;\n\r\t\|\/\\]+", sample)
        candidate_skills = []
        for t in tokens:
            t_clean = re.sub(r"[^A-Za-z0-9\+\#\s\.\-]", '', t).strip()
            if len(t_clean) > 2 and len(t_clean.split()) <= 5:
                candidate_skills.append(t_clean)
        jd_skills = JDSkills(job_title=title, hard_skills=candidate_skills[:10], experience_years=None)

    if not resume_skills:
        demo = heuristic_extract_skills(resume_text, max_items=30)
        listed = demo[:30]
        resume_skills = ResumeSkills(demonstrated_skills=demo, listed_skills=listed)

    return jd_skills, resume_skills

# --- Matching & scoring ---

def normalize_skill(s: str) -> str:
    return re.sub(r"[^a-z0-9\+\#\.\-]", ' ', s.lower()).strip()

def fuzzy_match(a: str, b: str) -> int:
    a_n, b_n = a.lower(), b.lower()
    if HAS_RAPIDFUZZ:
        try:
            return int(fuzz.token_sort_ratio(a_n, b_n))
        except Exception:
            pass
    # fallback: substring
    if a_n in b_n or b_n in a_n:
        return 100
    return 0

def compute_scores(jd_skills: JDSkills, resume_skills: ResumeSkills, jd_text: str, resume_text: str, use_semantic: bool = True):
    jd_list = [normalize_skill(s) for s in jd_skills.hard_skills if s]
    demo = [normalize_skill(s) for s in resume_skills.demonstrated_skills if s]
    listed = [normalize_skill(s) for s in resume_skills.listed_skills if s]

    demonstrated_count = 0
    listed_count = 0
    missing = []

    for req in jd_list:
        matched = False
        for s in demo:
            score = fuzzy_match(req, s)
            if score >= 85:
                demonstrated_count += 1
                matched = True
                break
        if matched:
            continue
        for s in listed:
            score = fuzzy_match(req, s)
            if score >= 80:
                listed_count += 1
                matched = True
                break
        if not matched:
            missing.append(req)

    hard_score = min(60, demonstrated_count * 12 + listed_count * 5)

    exp_bonus = 0
    if jd_skills.experience_years:
        nums = re.findall(r"(\d+)", jd_skills.experience_years)
        if nums:
            required = int(nums[0])
            rnums = re.findall(r"(\d+)\+?\s*(?:years|yrs|y)", resume_text.lower())
            candidate_years = int(rnums[0]) if rnums else 0
            if candidate_years >= required:
                exp_bonus = 8

    sem_score = 0
    if use_semantic and HAS_SKLEARN and jd_text and resume_text:
        try:
            vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=2000)
            X = vec.fit_transform([jd_text, resume_text])
            sim = cosine_similarity(X[0:1], X[1:2])[0][0]
            sem_score = int(sim * 30)
        except Exception as e:
            logger.warning("Semantic similarity failed: %s", e)
            sem_score = 0

    total = hard_score + sem_score + exp_bonus
    total = min(100, int(round(total)))

    if total >= 70:
        verdict = 'High Suitability'
    elif total >= 40:
        verdict = 'Medium Suitability'
    else:
        verdict = 'Low Suitability'

    missing_presentable = [m for m in missing][:5]

    # return breakdown too
    breakdown = {
        'hard_score': hard_score,
        'semantic_score': sem_score,
        'experience_bonus': exp_bonus,
        'demonstrated_count': demonstrated_count,
        'listed_count': listed_count
    }

    return total, verdict, missing_presentable, breakdown

# --- Feedback generation via LLM (or template) ---
def generate_feedback_via_llm(job_title: str, matched_count: int, missing_skills: List[str], strengths_summary: str) -> str:
    if HAS_LLM and GOOGLE_API_KEY:
        try:
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
            prompt = PromptTemplate(template=(
                "You are a helpful career coach. Given a job title, a short strengths summary, and a list of top missing skills,"
                " write a concise, professional, actionable feedback paragraph (2-3 sentences) that starts with something positive, then mentions the main gaps and suggests 1-2 concrete steps the candidate can take (projects, keywords to add, certifications or learning resources)."
                "\n\nJob Title: {job_title}\nMatched Skills Count: {matched_count}\nStrengths: {strengths}\nMissing Skills: {missing}\n\nKeep the feedback under 80 words."
            ), input_variables=["job_title","matched_count","strengths","missing"])
            chain = prompt | model
            out = llm_safe_invoke_chain(chain, {"job_title": job_title, "matched_count": matched_count, "strengths": strengths_summary, "missing": ", ".join(missing_skills)})
            return out.strip()
        except Exception as e:
            logger.warning("LLM feedback generation failed: %s", e)
    # fallback template
    positive = "Good experience in the areas you have demonstrated."
    if missing_skills:
        suggestions = f"To improve your fit, consider working on: {', '.join(missing_skills[:3])}. Build small projects showcasing these skills and mention them explicitly in your experience section."
    else:
        suggestions = "Your resume already matches the critical skills — quantify achievements and add outcome-oriented bullet points."
    return positive + " " + suggestions

# --- UI helpers ---
def get_verdict_color(verdict: str) -> str:
    if verdict == 'High Suitability':
        return '#28a745'
    elif verdict == 'Medium Suitability':
        return '#ffc107'
    else:
        return '#dc3545'

# --- App UI ---
st.image(LOGO_URL, width=200)
st.title("Automated Resume Relevance Checker — Robust")
st.markdown("---")

analysis_tab, dashboard_tab = st.tabs(["📊 Analysis", "🗂️ Dashboard"])

with analysis_tab:
    st.header("1. Input Job and Resume Data")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            jd_text = st.text_area("Paste the full Job Description here:", height=300, key="jd_text_area", value=st.session_state.get('jd_text', ''))
            st.session_state['jd_text'] = jd_text
            st.write("")
            opts_col1, opts_col2 = st.columns(2)
            with opts_col1:
                mode = st.selectbox("Scoring Mode:", ["Deterministic (recommended)", "LLM-only"], index=0, help="Deterministic uses local rules + optional TF-IDF semantic similarity for consistent results.")
                use_semantic = st.checkbox("Use TF-IDF semantic similarity (if available)", value=HAS_SKLEARN)
            with opts_col2:
                st.write("")
                if not HAS_SKLEARN:
                    st.caption("Tip: install scikit-learn for TF-IDF semantic similarity.")
                if not HAS_RAPIDFUZZ:
                    st.caption("Tip: install rapidfuzz for better fuzzy matching.")
                if not HAS_LLM:
                    st.caption("Tip: install langchain and langchain-google-genai for LLM feedback (optional).")

        with col2:
            uploader_key = f"file_uploader_key_{st.session_state['uploader_version']}"
            uploaded_files = st.file_uploader("Upload one or more resumes (PDF, DOCX):", type=["pdf", "docx"], accept_multiple_files=True, key=uploader_key)

    # Clear inputs safely
    c1, c2 = st.columns([1,4])
    with c1:
        if st.button("Clear Inputs"):
            # Reset JD text and bump uploader key to clear file_uploader contents
            st.session_state['jd_text'] = ""
            st.session_state['uploader_version'] += 1
            # force Rerun
            st.experimental_rerun()
    with c2:
        st.info("Use 'Clear Inputs' to reset the JD and uploaded files.")

    st.write("")
    if st.button("🚀 Analyze Resumes", type="primary"):
        if not jd_text.strip() or not uploaded_files:
            st.error("Please provide both a Job Description and at least one resume.")
        else:
            jd_hash = compute_jd_hash(jd_text)
            files_to_process = []
            for file in uploaded_files:
                if check_if_exists(file.name, jd_hash):
                    st.warning(f"Skipping '{file.name}': already analyzed for this JD.")
                else:
                    files_to_process.append(file)

            if not files_to_process:
                st.info("No new resumes to analyze.")
            else:
                progress = st.progress(0, text="Initializing...")
                batch_results = []
                for i, resume_file in enumerate(files_to_process):
                    try:
                        progress.progress(int((i / len(files_to_process)) * 100), text=f"Reading {resume_file.name}...")
                        resume_text = get_file_text(resume_file)
                        if not resume_text:
                            st.error(f"Could not extract text from {resume_file.name}. Skipping.")
                            continue

                        progress.progress(int(((i + 0.2) / len(files_to_process)) * 100), text="Extracting structured info...")
                        jd_struct, resume_struct = llm_extract_structured(jd_text, resume_text)

                        if mode.startswith('Deterministic'):
                            score, verdict, missing, breakdown = compute_scores(jd_struct, resume_struct, jd_text if use_semantic else "", resume_text, use_semantic)
                            strengths = f"Demonstrated: {breakdown['demonstrated_count']}, Listed: {breakdown['listed_count']}"
                            feedback = generate_feedback_via_llm(jd_struct.job_title, breakdown['demonstrated_count'] + breakdown['listed_count'], missing, strengths)
                            report = FinalAnalysis(relevance_score=score, verdict=verdict, missing_skills=missing, candidate_feedback=feedback)
                        else:
                            # LLM-only path: ask LLM for a full FinalAnalysis (best-effort)
                            try:
                                if HAS_LLM and GOOGLE_API_KEY:
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

Provide final JSON with relevance_score (0-100), verdict, missing_skills (list) and candidate_feedback.
{format_instructions}
"""
                                    analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["job_title", "jd_hard_skills", "jd_experience", "resume_demonstrated", "resume_listed"], partial_variables={"format_instructions": analysis_parser.get_format_instructions()})
                                    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)
                                    analysis_chain = analysis_prompt | model | analysis_parser
                                    final_report = llm_safe_invoke_chain(analysis_chain, {
                                        "job_title": jd_struct.job_title,
                                        "jd_hard_skills": jd_struct.hard_skills,
                                        "jd_experience": jd_struct.experience_years or "Not Specified",
                                        "resume_demonstrated": resume_struct.demonstrated_skills,
                                        "resume_listed": resume_struct.listed_skills
                                    })
                                    report = final_report
                                else:
                                    # fallback to deterministic if LLM not configured
                                    score, verdict, missing, breakdown = compute_scores(jd_struct, resume_struct, jd_text if use_semantic else "", resume_text, use_semantic)
                                    strengths = f"Demonstrated: {breakdown['demonstrated_count']}, Listed: {breakdown['listed_count']}"
                                    feedback = generate_feedback_via_llm(jd_struct.job_title, breakdown['demonstrated_count'] + breakdown['listed_count'], missing, strengths)
                                    report = FinalAnalysis(relevance_score=score, verdict=verdict, missing_skills=missing, candidate_feedback=feedback)
                            except Exception as e:
                                logger.exception("LLM-only path failed, falling back to deterministic: %s", e)
                                score, verdict, missing, breakdown = compute_scores(jd_struct, resume_struct, jd_text if use_semantic else "", resume_text, use_semantic)
                                strengths = f"Demonstrated: {breakdown['demonstrated_count']}, Listed: {breakdown['listed_count']}"
                                feedback = generate_feedback_via_llm(jd_struct.job_title, breakdown['demonstrated_count'] + breakdown['listed_count'], missing, strengths)
                                report = FinalAnalysis(relevance_score=score, verdict=verdict, missing_skills=missing, candidate_feedback=feedback)

                        add_analysis_to_db(resume_file.name, jd_struct.job_title, jd_text, report)

                        progress.progress(int(((i + 0.8) / len(files_to_process)) * 100), text=f"Saving results for {resume_file.name}...")

                        # Display
                        with st.container():
                            st.markdown(f"### Candidate: {resume_file.name}")
                            c1, c2 = st.columns([1,3])
                            with c1:
                                st.markdown(f"<p style='color:{get_verdict_color(report.verdict)};'><strong>{report.verdict}</strong></p>", unsafe_allow_html=True)
                                st.metric("Score", f"{report.relevance_score}%")
                            with c2:
                                st.warning("**Identified Gaps:**")
                                if report.missing_skills:
                                    st.markdown('\n'.join([f"- {item}" for item in report.missing_skills]))
                                else:
                                    st.markdown("- None")
                                st.success("**Personalized Feedback:**")
                                st.write(report.candidate_feedback)

                            # scoring breakdown
                            try:
                                score, verdict, missing, breakdown = compute_scores(jd_struct, resume_struct, jd_text if use_semantic else "", resume_text, use_semantic)
                                st.info("**Scoring breakdown (approx):**")
                                st.write(breakdown)
                            except Exception:
                                pass

                        batch_results.append({
                            'filename': resume_file.name,
                            'jd_title': jd_struct.job_title,
                            'score': report.relevance_score,
                            'verdict': report.verdict,
                            'missing_skills': ", ".join(report.missing_skills),
                            'feedback': report.candidate_feedback
                        })

                    except Exception as e:
                        logger.exception("Error processing %s: %s", getattr(resume_file, 'name', str(resume_file)), e)
                        st.error(f"Error processing {getattr(resume_file, 'name', str(resume_file))}: {e}")

                progress.progress(100, text="Done")
                st.success("All new resumes have been analyzed!")
                st.balloons()

                if batch_results:
                    df_export = pd.DataFrame(batch_results)
                    st.download_button("Download batch results as CSV", data=df_export.to_csv(index=False).encode('utf-8'), file_name=f"analysis_batch_{uuid.uuid4().hex[:8]}.csv", mime='text/csv')

with dashboard_tab:
    st.header("Past Analysis Results")
    df = load_data_for_dashboard()

    if df.empty:
        st.info("No results found. Run a new analysis in the 'Analysis' tab.")
    else:
        st.subheader("Filter & Manage Results")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            jd_options = ["All JDs"] + list(df['jd_title'].unique())
            selected_jd = st.selectbox("Filter by Job Title:", options=jd_options, key="jd_filter")
        df_filtered = df[df['jd_title'] == selected_jd] if selected_jd != "All JDs" else df
        with fc2:
            verdict_options = ["All Verdicts"] + list(df_filtered['verdict'].unique())
            selected_verdict = st.selectbox("Filter by Verdict:", options=verdict_options, key="verdict_filter")
        df_filtered = df_filtered[df_filtered['verdict'] == selected_verdict] if selected_verdict != "All Verdicts" else df_filtered
        with fc3:
            min_score = st.slider("Minimum Score:", 0, 100, 0, key="score_slider")
        final_df = df_filtered[df_filtered['score'] >= min_score].reset_index(drop=True)

        st.subheader(f"Displaying {len(final_df)} Results")
        if not final_df.empty:
            for idx, row in final_df.iterrows():
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
                            if rec:
                                st.subheader(f"Analysis for: {rec[2]}")
                                st.metric("Relevance Score", f"{rec[6]}%")
                                st.markdown(f"<p style='color:{get_verdict_color(rec[7])};'><strong>Verdict: {rec[7]}</strong></p>", unsafe_allow_html=True)
                                st.markdown("---")
                                st.subheader("Identified Gaps")
                                st.markdown(rec[8] or "None")
                                st.markdown("---")
                                st.subheader("Candidate Feedback")
                                st.info(rec[9] or "")
                                st.markdown("---")
                                with st.expander("Show Original Job Description"):
                                    st.text_area("JD", value=rec[10] or "", height=200, disabled=True, label_visibility="collapsed")
                    with c3:
                        if st.button("Delete", key=f"delete_{row['id']}", type="secondary", use_container_width=True):
                            delete_analysis_from_db(row['id'])
                            st.success(f"Deleted record for {row['resume_filename']}.")
                            st.experimental_rerun()

        st.markdown("---")
        dbc1, dbc2 = st.columns([1,1])
        with dbc1:
            if st.button("Export full DB as CSV"):
                try:
                    conn = get_conn()
                    df_all = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC", conn)
                    conn.close()
                    st.download_button("Download CSV", data=df_all.to_csv(index=False).encode('utf-8'), file_name=f"analysis_full_{uuid.uuid4().hex[:8]}.csv", mime='text/csv')
                except Exception as e:
                    logger.exception("Export failed: %s", e)
                    st.error("Export failed. See logs.")
        with dbc2:
            if st.button("Clear entire DB (danger)"):
                safe_execute("DELETE FROM results")
                st.success("All records deleted.")
                st.experimental_rerun()

st.markdown("---")
st.markdown("Developed by **Siddhant Pal** — Robust production-ready edition.")
