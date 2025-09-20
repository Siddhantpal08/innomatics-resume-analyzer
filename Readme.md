🤖 Automated Resume Relevance Checker
A Submission for the Code4Edtech Challenge by Innomatics Research Labs
This project is a sophisticated, AI-powered system designed to solve a critical problem for the Innomatics Placement Team: the manual, time-consuming, and inconsistent process of resume evaluation. This application automates the entire workflow, providing a scalable, consistent, and insightful analysis of a candidate's suitability for a given job role.

✨ Live Demo
[Link to Deployed Application on Streamlit Cloud] (<- You will paste your URL here after deployment)

🎯 The Problem
The Innomatics placement team receives thousands of applications for dozens of job roles weekly. The manual process of sifting through resumes to find qualified candidates is a major bottleneck, leading to:

Delays: Slow shortlisting processes for hiring companies.

Inconsistency: Different evaluators can interpret requirements differently, leading to biased or inconsistent results.

High Workload: Placement staff spend more time on manual screening and less time on high-value tasks like student guidance and interview preparation.

This project directly addresses these challenges by building the automated system proposed in the hackathon problem statement.

🚀 Our Solution: A Hybrid AI Scoring Engine
I developed a multi-stage Hybrid Scoring Engine that combines the speed of traditional NLP with the deep contextual understanding of modern Large Language Models (LLMs). This ensures both accuracy and efficiency.

The analysis pipeline works in three stages:

Hard Match Analysis: The system first performs a high-speed keyword analysis using spaCy, a production-grade NLP library. It parses the Job Description to create a checklist of required skills and then rapidly scans each resume to calculate an initial keyword match score.

Semantic Match Analysis: To understand the context beyond keywords, the system uses a pretrained Sentence-BERT model (all-MiniLM-L6-v2) to generate vector embeddings. The resume is broken into logical chunks, embedded, and stored in a ChromaDB vector database. The core requirements of the Job Description are then used to perform a similarity search, instantly finding the most conceptually relevant sections of the resume, even if they don't use the exact same terminology.

LLM Synthesis & Feedback Generation: Finally, the pre-processed data from the first two stages is synthesized by a powerful LLM (Google's Gemini 1.5 Flash). The LLM is given a specific persona—an expert HR Analyst—and is tasked with generating the final, comprehensive report based only on the provided data. This includes:

A final Relevance Score (0-100).

A clear Verdict (High, Medium, or Low Suitability).

A list of Identified Gaps (missing skills, etc.).

Personalized, Actionable Feedback for the student.

🛠️ Tech Stack
This project was built using a modern, robust, and scalable tech stack as recommended in the problem statement.

Language: Python

Web Framework: Streamlit

AI Orchestration: LangChain

LLM: Google Gemini 1.5 Flash

NLP (Hard Match): spaCy

Embeddings (Semantic Match): Hugging Face Sentence-BERT (all-MiniLM-L6-v2)

Vector Database: ChromaDB

Database: SQLite

📋 Features
Bulk Resume Processing: Upload and analyze multiple resumes (PDF & DOCX) simultaneously.

Persistent Dashboard: All analysis results are saved to a local SQLite database.

Advanced Filtering: The dashboard allows the placement team to search and filter results by Job Description, AI Verdict, and Score.

Data Management: Functionality to delete old or irrelevant records directly from the dashboard.

Polished UI/UX: A clean, intuitive, and professional user interface with custom loaders and animations.

⚙️ How to Run Locally
Clone the repository:

git clone [YOUR_GITHUB_REPO_URL]
cd [repository-name]

Create and activate a virtual environment:

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create a .env file in the root directory and add your Google Gemini API key:

GOOGLE_API_KEY="YOUR_API_KEY_HERE"

Run the Streamlit app:

streamlit run app.py
