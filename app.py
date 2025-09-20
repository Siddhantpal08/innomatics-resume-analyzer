import streamlit as st
import sqlite3
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
import re
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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
    """
    Performs a simple, case-insensitive search for skills in the resume text.
    This replaces the spaCy PhraseMatcher with pure Python for better compatibility.
    """
    matched_skills = []
    # Prepare the resume text for searching
    resume_text_lower = resume_text.lower()
    
    for skill in SKILL_DB:
        # Use regex with word boundaries to find whole words only
        # This prevents matching "java" in "javascript", for example.
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, resume_text_lower):
            matched_skills.append(skill)
            
    return list(set(matched_skills))

@st.cache_resource
def get_conversational_chain(jd_text):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        
        # Split JD text into chunks for vector store
        text_chunks = [jd_text] # Simple split, can be improved
        
        vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, collection_name="jd_collection")
        
        memory = ConversationBufferMemory(name="chat_history", memory_key="chat_history", return_messages=True)
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2),
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True
        )
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

# --- Pydantic Models for Structured Output ---
class AnalysisResult(BaseModel):
    relevance_score: int = Field(description="A score from 0 to 100 representing how relevant the resume is to the job description.")
    verdict: str = Field(description="A final verdict: 'High Suitability', 'Medium Suitability', or 'Low Suitability'.")
    missing_skills: list[str] = Field(description="A list of key skills from the job description that are missing in the resume.")
    candidate_feedback: str = Field(description="Constructive, personalized feedback for the candidate on how to improve their resume for this specific role.")

# --- Streamlit UI ---

st.set_page_config(page_title="🤖 Automated Resume Relevance Checker", layout="wide")

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analyze Resumes", "View Analysis History"])
st.sidebar.markdown("---")
st.sidebar.write("Developed for the Innomatics Code4EdTech Hackathon.")

# --- Main Application Logic ---
def main_app():
    st.title("🤖 Automated Resume Relevance Checker")
    st.markdown("##### Upload a Job Description and one or more resumes to get an AI-powered relevance analysis.")
    st.markdown("---")
    
    jd_text = st.text_area("Paste the Job Description here:", height=200)
    uploaded_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    if st.button("Analyze Resumes", type="primary"):
        if not jd_text or not uploaded_files:
            st.warning("Please provide both a Job Description and at least one resume.")
            return
        if not GOOGLE_API_KEY:
            st.error("Google API Key not found. Please set it in your .env file.")
            return

        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
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

                    prompt_template = """
                    You are an expert HR Technology Analyst specializing in resume evaluation.
                    Your task is to provide a detailed, data-driven analysis of a resume against a job description.

                    CONTEXT:
                    Job Description: {jd}
                    Resume Text: {resume}
                    Hard Skills Found in Resume: {skills}

                    INSTRUCTIONS:
                    Based on ALL the provided context, perform the following actions:
                    1.  **Calculate Relevance Score:** Determine a score from 0-100. Base this on the presence of required skills, the depth of relevant experience described, and the overall alignment with the job description. The hard skills list is a good starting point, but you must analyze the semantic meaning of the resume text for a complete picture.
                    2.  **Give a Verdict:** Based on the score, provide a verdict: 'High Suitability' (score > 75), 'Medium Suitability' (score 45-75), or 'Low Suitability' (score < 45).
                    3.  **Identify Missing Skills:** List the most critical skills mentioned in the job description that are NOT found or strongly implied in the resume.
                    4.  **Generate Candidate Feedback:** Write a brief, constructive paragraph for the candidate. Start by acknowledging their strengths, then suggest specific ways they can improve their resume to better align with this job role. Mention 2-3 key areas from the JD they should highlight more. Be encouraging and professional.

                    Please format your entire output as a single JSON object with the following keys: "relevance_score", "verdict", "missing_skills", "candidate_feedback".
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
                    
                    st.write("**Missing Skills:**")
                    st.write(", ".join(analysis['missing_skills']) if analysis['missing_skills'] else "None specified.")

                    st.write("**Feedback for Candidate:**")
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
                    st.error(f"An error occurred while analyzing {uploaded_file.name}: {e}")
                
                finally:
                    progress_bar.progress((i + 1) / total_files)
                    st.markdown("---")
        
        st.success("Analysis complete for all resumes!")


# --- History Page Logic ---
def history_page():
    st.title("📄 Analysis History")
    
    conn = sqlite3.connect('resume_analysis.db')
    try:
        df = pd.read_sql_query("SELECT * FROM analysis_results ORDER BY id DESC", conn)
    except pd.io.sql.DatabaseError:
        st.warning("No analysis history found.")
        df = pd.DataFrame() # Create empty dataframe
    conn.close()

    if not df.empty:
        # Filters
        st.sidebar.subheader("Filter History")
        jd_options = ["All"] + list(df['jd_text'].unique())
        selected_jd = st.sidebar.selectbox("Filter by Job Description", options=jd_options)
        
        verdict_options = ["All"] + list(df['verdict'].unique())
        selected_verdict = st.sidebar.selectbox("Filter by Verdict", options=verdict_options)

        score_range = st.sidebar.slider(
            "Filter by Score",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )

        # Apply filters
        filtered_df = df.copy()
        if selected_jd != "All":
            filtered_df = filtered_df[filtered_df['jd_text'] == selected_jd]
        if selected_verdict != "All":
            filtered_df = filtered_df[filtered_df['verdict'] == selected_verdict]
        
        filtered_df = filtered_df[
            (filtered_df['relevance_score'] >= score_range[0]) &
            (filtered_df['relevance_score'] <= score_range[1])
        ]

        st.write(f"Displaying {len(filtered_df)} of {len(df)} total records.")

        # Display data
        for index, row in filtered_df.iterrows():
            with st.expander(f"**{row['resume_filename']}** | Score: {row['relevance_score']}% | Verdict: {row['verdict']}"):
                st.text_area("Job Description", value=row['jd_text'], height=150, disabled=True, key=f"jd_{row['id']}")
                st.write(f"**Missing Skills:** {row['missing_skills']}")
                st.info(f"**Candidate Feedback:** {row['candidate_feedback']}")
                
                # Delete button
                if st.button("Delete Record", key=f"delete_{row['id']}"):
                    conn = sqlite3.connect('resume_analysis.db')
                    c = conn.cursor()
                    c.execute("DELETE FROM analysis_results WHERE id = ?", (row['id'],))
                    conn.commit()
                    conn.close()
                    st.success(f"Deleted record for {row['resume_filename']}.")
                    st.rerun() # Refresh the page to show updated list
    else:
        st.info("No analysis has been performed yet. Go to 'Analyze Resumes' to get started.")


# --- Page Routing ---
if page == "Analyze Resumes":
    main_app()
else:
    history_page()

