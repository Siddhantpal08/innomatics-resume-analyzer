import streamlit as st
import sqlite3
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
import docx
import os
from dotenv import load_dotenv
import google.generativeai as genai

# ============ CONFIG ============
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

DB_FILE = "results.db"

# ============ DB INIT ============
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_filename TEXT,
            jd_hash TEXT,
            score REAL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn, c

# ============ HELPERS ============
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def extract_text_from_resume(file) -> str:
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = ""
            for page in reader.pages[:10]:  # limit to 10 pages
                text += page.extract_text() or ""
            return text.strip()
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return file.read().decode(errors="ignore")
    except Exception as e:
        return f"⚠️ Failed to parse: {e}"

async def score_resume_with_gemini(resume_text: str, jd_text: str, timeout: int = 25):
    """Use Google Gemini API for scoring, fallback if it fails."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Compare the following resume with the job description and return only a numeric relevance score (0-100).
    Then give a brief explanation.

    Job Description:
    {jd_text}

    Resume:
    {resume_text[:3000]}  # truncated for safety
    """

    try:
        # run Gemini call in thread (to avoid blocking asyncio)
        loop = asyncio.get_event_loop()
        resp = await asyncio.wait_for(
            loop.run_in_executor(None, model.generate_content, prompt),
            timeout=timeout
        )

        text_out = resp.text.strip()
        # crude numeric parse
        digits = "".join([c for c in text_out if c.isdigit()])
        score = float(digits) if digits else 0
        return min(max(score, 0), 100), text_out
    except Exception as e:
        return None, f"⚠️ Gemini failed: {e}"

def deterministic_score(resume_text: str, jd_text: str):
    """Keyword overlap fallback scoring"""
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    if not jd_words:
        return 0, "No JD words"
    overlap = resume_words.intersection(jd_words)
    score = round(len(overlap) / len(jd_words) * 100, 2)
    return score, f"Deterministic keyword match: {len(overlap)} common words"

def save_to_db(filename, jd_hash, score, details):
    conn, c = init_db()
    try:
        c.execute("INSERT INTO results (resume_filename, jd_hash, score, details) VALUES (?, ?, ?, ?)",
                  (filename, jd_hash, score, details))
        conn.commit()
    except Exception as e:
        st.error(f"DB Error: {e}")
    finally:
        conn.close()

def check_if_exists(filename, jd_hash):
    conn, c = init_db()
    c.execute("SELECT id FROM results WHERE resume_filename = ? AND jd_hash = ?", (filename, jd_hash))
    res = c.fetchone()
    conn.close()
    return res is not None

# ============ MAIN PROCESS ============
def process_resume(file, jd_text, jd_hash):
    """Thread-safe sync wrapper for each resume"""
    try:
        resume_text = extract_text_from_resume(file)

        # run Gemini scoring inside thread
        score, details = asyncio.run(score_resume_with_gemini(resume_text, jd_text))
        if score is None:  # fallback if Gemini fails
            score, details = deterministic_score(resume_text, jd_text)

        save_to_db(file.name, jd_hash, score, details)
        return file.name, score, details
    except Exception as e:
        return file.name, 0, f"⚠️ Error: {e}"

# ============ STREAMLIT UI ============
st.set_page_config(page_title="Resume Analyzer", layout="wide")

if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""

st.title("📊 AI-Powered Resume Analyzer")

st.subheader("1️⃣ Paste Job Description")
jd_text = st.text_area("Paste the JD here:", value=st.session_state["jd_text"], height=200)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Clear Input"):
        st.session_state["jd_text"] = ""
        jd_text = ""
with col2:
    uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if jd_text and uploaded_files:
    jd_hash = hash_text(jd_text)

    if st.button("🚀 Analyze Resumes"):
        start_time = time.time()
        st.info("Processing resumes in parallel... Please wait.")

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_resume, f, jd_text, jd_hash) for f in uploaded_files]
            for future in as_completed(futures):
                results.append(future.result())

        elapsed = round(time.time() - start_time, 2)

        st.success(f"✅ Completed in {elapsed} seconds")

        for fname, score, details in results:
            st.write(f"**{fname}** → Score: {score}")
            with st.expander("Details"):
                st.write(details)
