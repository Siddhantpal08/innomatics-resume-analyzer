import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import spacy
from spacy.matcher import PhraseMatcher
from typing import List
import sqlite3
import pandas as pd
from datetime import datetime

# LangChain and Pydantic Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def load_all_models():
    """Loads and caches all heavy models."""
    spacy_model = spacy.load("en_core_web_sm")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return spacy_model, embedding_model

# --- DATABASE FUNCTIONS ---
DB_FILE = "analysis_results.db"

def init_database():
    """Initializes the SQLite database and creates the results table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            candidate_name TEXT NOT NULL,
            jd_summary TEXT NOT NULL,
            score INTEGER NOT NULL,
            verdict TEXT NOT NULL,
            missing_skills TEXT,
            feedback TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_analysis_to_db(candidate_name, jd_summary, report):
    """Adds a new analysis report to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO results (timestamp, candidate_name, jd_summary, score, verdict, missing_skills, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(),
        candidate_name,
        jd_summary,
        report.relevance_score,
        report.verdict,
        ", ".join(report.missing_elements),
        report.improvement_feedback
    ))
    conn.commit()
    conn.close()

def load_data_for_dashboard():
    """Loads all analysis results from the database into a Pandas DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM results ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    return df

def delete_analysis_from_db(record_id):
    """Deletes a specific analysis record from the database by its ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM results WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

# --- UI Helper Functions ---
def style_verdict(verdict):
    if verdict == 'High Suitability': return f'**<span style="color: #28a745;">{verdict}</span>**'
    elif verdict == 'Medium Suitability': return f'**<span style="color: #ffc107;">{verdict}</span>**'
    else: return f'**<span style="color: #dc3545;">{verdict}</span>**'

def highlight_verdict(row):
    verdict = row['verdict']
    if verdict == 'High Suitability': return ['background-color: #d4edda; color: #155724;'] * len(row)
    elif verdict == 'Medium Suitability': return ['background-color: #fff3cd; color: #856404;'] * len(row)
    else: return ['background-color: #f8d7da; color: #721c24;'] * len(row)

# --- Core Logic Functions ---
SKILL_KEYWORDS = [
    'Python', 'Java', 'C++', 'JavaScript', 'Go', 'Ruby', 'PHP', 'Django', 'Flask', 'Spring', 'Node.js', 'React', 'Angular', 'Vue.js', 'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'AWS', 'Azure', 'Google Cloud', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'Git', 'JIRA', 'Confluence', 'Agile', 'Scrum', 'CI/CD', 'DevOps', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'scikit-learn', 'Data Analysis', 'Pandas', 'NumPy', 'Matplotlib', 'Tableau', 'Power BI', 'Natural Language Processing', 'NLP', 'spaCy', 'NLTK', 'API', 'REST', 'GraphQL', 'Microservices', 'System Design'
]

def extract_skills(nlp, text):
    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill) for skill in SKILL_KEYWORDS]
    matcher.add("SKILL_MATCHER", patterns)
    return list({doc[start:end].text.title() for _, start, end in matcher(doc)})

def get_uploaded_file_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages: text += page.extract_text() or ""
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs: text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None
    return text

def perform_semantic_search(embedding_model, resume_text, jd_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    resume_chunks = text_splitter.split_text(resume_text)
    if not resume_chunks: return []
    vector_store = Chroma.from_texts(resume_chunks, embedding_model)
    return [doc.page_content for doc in vector_store.similarity_search(jd_text, k=3)]

class FinalAnalysis(BaseModel):
    relevance_score: int = Field(description="The final relevance score from 0 to 100.")
    verdict: str = Field(description="A verdict of 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_elements: List[str] = Field(description="A list of key skills, certifications, or projects missing from the resume.")
    improvement_feedback: str = Field(description="Personalized, actionable feedback for the student on how to improve their resume for this specific job.")

def get_llm_synthesis(jd_text, matched_skills, missing_skills, relevant_chunks):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=FinalAnalysis)
    prompt_template_str = """
    You are an expert HR Technology Analyst at Innomatics Research Labs. Your task is to provide a final, synthesized analysis of a resume based on pre-processed data. Base your entire analysis on the data provided below.

    **Job Description Requirements:** {jd}
    
    **Pre-Analysis Data:**
    - Matched Keywords: {matched_skills}
    - Missing Keywords: {missing_skills}
    - Semantically Relevant Experience Chunks from Resume: 
    --- 
    {relevant_chunks} 
    ---

    Based **only** on the information above, provide a final analysis. Generate a weighted relevance score where keyword matches account for 60% and semantic relevance accounts for 40% of the final score. The verdict must be based on this score. The feedback must be specific and actionable for the student.

    **Provide your output in the following JSON format:**
    {format_instructions}
    """
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["jd", "matched_skills", "missing_skills", "relevant_chunks"], partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = prompt | model | parser
    return chain.invoke({"jd": jd_text, "matched_skills": ", ".join(matched_skills), "missing_skills": ", ".join(missing_skills), "relevant_chunks": "\n---\n".join(relevant_chunks)})

# --- Main App ---
init_database()
st.set_page_config(page_title="Innomatics Resume Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        /* More robust selector for the "Running..." indicator */
        [data-testid="stStatusWidget"] > div {
            opacity: 0; /* Hide the default Streamlit spinner */
        }
        /* Create our own custom pulsing dot */
        [data-testid="stStatusWidget"]::after {
            content: '';
            display: block;
            width: 10px;
            height: 10px;
            background-color: #4A90E2;
            animation: pulse 2s infinite;
            border-radius: 50%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(74, 144, 226, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(74, 144, 226, 0); }
            100% { box-shadow: 0 0 0 0 rgba(74, 144, 226, 0); }
        }
    </style>
""", unsafe_allow_html=True)


if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

if not st.session_state.system_ready:
    st.markdown("""
        <style>
            .loader-container { display: flex; justify-content: center; align-items: center; height: 80vh; flex-direction: column; }
            .dots-container { display: flex; gap: 10px; }
            .dot { width: 20px; height: 20px; border-radius: 50%; background-color: #4A90E2; animation: pulse-loader 1.5s infinite ease-in-out; }
            .dot:nth-child(2) { animation-delay: 0.25s; }
            .dot:nth-child(3) { animation-delay: 0.5s; }
            @keyframes pulse-loader { 0%, 100% { transform: scale(0.5); opacity: 0.5; } 50% { transform: scale(1); opacity: 1; } }
        </style>
        <div class="loader-container">
            <div class="dots-container"> <div class="dot"></div> <div class="dot"></div> <div class="dot"></div> </div>
            <h2 style="margin-top: 20px;">Initializing System...</h2> <p>AI models are loading, please wait.</p>
        </div>
    """, unsafe_allow_html=True)
    
    nlp, embedding_model = load_all_models()
    st.session_state.nlp, st.session_state.embedding_model = nlp, embedding_model
    st.session_state.system_ready = True
    st.rerun()

else:
    nlp, embedding_model = st.session_state.nlp, st.session_state.embedding_model

    title_col, button_col = st.columns([4, 1])
    with title_col:
        st.image("https://placehold.co/200x70/ffffff/000000?text=Innomatics&font=lato", width=200)
        st.title("Placement Team Dashboard")
    with button_col:
        st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True) # Vertical Spacer
        if st.button("🧹 Clear & Reset Session", key="clear_button"):
            st.session_state.jd_text = ""
            st.session_state.file_uploader_key = str(datetime.now().timestamp())
            st.rerun()

    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 'initial'

    analysis_tab, dashboard_tab = st.tabs(["📊 Analysis", "🗂️ Dashboard"])

    with analysis_tab:
        st.header("Run a New Analysis")
        with st.container(border=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📋 Job Description")
                jd_text = st.text_area("Paste the Job Description text here:", height=300, key="jd_text", label_visibility="collapsed")
            with col2:
                st.subheader("📄 Candidate Resumes")
                uploaded_files = st.file_uploader("Upload resumes:", type=["pdf", "docx"], accept_multiple_files=True, key=st.session_state.file_uploader_key, label_visibility="collapsed")
        st.write("")
        if st.button("🚀 Run Full Analysis", type="primary", key="analysis_button"):
            if not jd_text or not uploaded_files:
                st.error("Please provide both a JD and at least one resume.")
            else:
                with st.spinner("Processing resumes..."):
                    required_skills = set(extract_skills(nlp, jd_text))
                    st.info(f"**Required Skills in JD:** {', '.join(required_skills) if required_skills else 'None'}")
                    progress_bar = st.progress(0, text="Analyzing resumes...")
                    for i, resume_file in enumerate(uploaded_files):
                        with st.container(border=True):
                            st.markdown(f"### Candidate: {resume_file.name}")
                            resume_text = get_uploaded_file_text(resume_file)
                            if not resume_text: continue

                            resume_skills = set(extract_skills(nlp, resume_text))
                            matched, missing = required_skills.intersection(resume_skills), required_skills.difference(resume_skills)
                            chunks = perform_semantic_search(embedding_model, resume_text, jd_text)
                            final_report = get_llm_synthesis(jd_text, matched, missing, chunks)

                            if final_report:
                                jd_summary = " ".join(jd_text.split()[:10]).strip() + "..."
                                add_analysis_to_db(resume_file.name, jd_summary, final_report)
                                res_col1, res_col2 = st.columns([1, 3])
                                with res_col1:
                                    st.markdown(style_verdict(final_report.verdict), unsafe_allow_html=True)
                                    st.metric("Score", f"{final_report.relevance_score}%")
                                with res_col2:
                                    st.warning("Identified Gaps:")
                                    st.markdown("\n".join([f"- {item}" for item in final_report.missing_elements]) or "- None")
                                    st.success("Personalized Feedback:")
                                    st.write(final_report.improvement_feedback)
                        progress_bar.progress((i + 1) / len(uploaded_files), text=f"Analyzed {i+1} of {len(uploaded_files)}")
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

            final_df = df_filtered[df_filtered['score'] >= min_score]
            st.subheader(f"Displaying {len(final_df)} Results")
            st.dataframe(final_df.style.apply(highlight_verdict, axis=1), hide_index=True)
            
            if not final_df.empty:
                st.markdown("---")
                st.subheader("Delete a Record")
                delete_options = [f"{row['id']} - {row['candidate_name']} ({row['jd_summary']})" for index, row in final_df.iterrows()]
                record_to_delete_display = st.selectbox("Select a record to delete:", options=delete_options, key="delete_select")
                
                if st.button("❌ Delete Selected Record", type="secondary", key="delete_button"):
                    if record_to_delete_display:
                        record_id_to_delete = int(record_to_delete_display.split(" - ")[0])
                        delete_analysis_from_db(record_id_to_delete)
                        st.success(f"Record ID {record_id_to_delete} has been deleted.")
                        st.rerun()
                    else:
                        st.warning("No record selected to delete.")

    st.markdown("---")
    st.markdown("Developed by **Siddhant Pal** for the Code4EdTech Challenge by Innomatics Research Labs.")

