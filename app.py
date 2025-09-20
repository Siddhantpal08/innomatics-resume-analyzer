

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
