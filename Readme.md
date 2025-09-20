🤖 Automated Resume Relevance Checker
A Hackathon Submission for the Code4EdTech Challenge by Innomatics Research Labs
This project is a comprehensive, AI-powered system designed to automate and scale the resume evaluation process for the Innomatics Placement Team. It directly addresses the challenges of manual, time-consuming, and inconsistent resume screening by providing instant, data-driven analysis of a candidate's relevance to a specific job description.

Developed by: Siddhant Pal

🚀 Live Application
[Link to Deployed Application on Streamlit Cloud] <!-- THIS WILL BE FILLED IN AFTER DEPLOYMENT -->

🎯 The Problem
The Innomatics placement team receives thousands of applications for dozens of job roles weekly across multiple locations. The manual process of reviewing each resume against a job description (JD) is a significant bottleneck, leading to:

Delays: Slow shortlisting of qualified candidates.

Inconsistency: Subjective judgments from different evaluators.

High Workload: Reduced capacity for placement staff to focus on high-value tasks like student guidance and interview preparation.

This project was built to solve these problems with a robust, scalable, and intelligent solution.

✨ Our Solution: A Hybrid AI Architecture
This system is not just a simple AI wrapper; it's a sophisticated, multi-stage pipeline designed for both accuracy and explainability.

Hard Match Analysis: The system first performs a high-speed, objective screening using a custom dictionary of over 50 key technical skills. It leverages spaCy's efficient PhraseMatcher to instantly identify which required skills are present in the resume and which are missing. This provides a foundational, keyword-based score.

Semantic Match Analysis: To understand the context and meaning beyond keywords, the system then employs a pretrained Sentence-BERT model (all-MiniLM-L6-v2). It breaks the resume into meaningful chunks, converts them into vector embeddings, and stores them in a local ChromaDB vector database. The key requirements from the job description are then used to perform a similarity search, retrieving the most conceptually relevant sections of the candidate's experience, even if they don't use the exact same wording.

LLM Synthesis & Feedback Generation: The final and most powerful stage feeds the pre-analyzed data from the first two stages into a large language model (Google's Gemini 1.5 Flash). The LLM is given a specific persona—an expert HR Technology Analyst—and a structured task: to synthesize the hard skill matches, semantic relevance, and missing keywords into a final, holistic report.

This hybrid approach ensures the final output is grounded in objective data while being enriched with nuanced, AI-driven insights.

🏆 Core Features
Bulk Resume Processing: Upload and analyze multiple resumes (PDF & DOCX) against a single job description in one go.

Hybrid Scoring: Generates a weighted Relevance Score (0-100) based on both keyword matches and semantic context.

Automated Verdict: Provides a clear verdict of High, Medium, or Low Suitability for each candidate.

Actionable Feedback:

Highlights specific skill gaps and missing qualifications for the placement team.

Generates personalized, constructive feedback for each student on how to improve their resume for that specific role.

Persistent Dashboard: All analysis results are saved to a local SQLite database. The interactive dashboard allows the placement team to view, search, and filter all historical results by Job Description, Verdict, and Score.

Full Data Management: The dashboard includes the ability to permanently delete records that are no longer needed.

🛠️ Tech Stack
Language: Python

Web Framework: Streamlit

Database: SQLite

Data Handling: Pandas

AI Orchestration: LangChain

LLM: Google Gemini 1.5 Flash

Hard Match NLP: spaCy

Semantic Search:

Embeddings: Sentence-BERT (all-MiniLM-L6-v2) via Hugging Face

Vector Database: ChromaDB

Structured Output: Pydantic

⚙️ How to Run Locally
Clone the repository:

git clone [https://github.com/Siddhantpal08/innomatics-resume-analyzer.git](https://github.com/Siddhantpal08/innomatics-resume-analyzer.git)
cd innomatics-resume-analyzer

Create and activate a virtual environment:

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create a .env file in the project root and add your Google API key:

GOOGLE_API_KEY="YOUR_API_KEY_HERE"

Run the Streamlit app:

streamlit run app.py

This project was built with passion and a deep understanding of the problem statement to provide a truly valuable tool for the Innomatics ecosystem.
