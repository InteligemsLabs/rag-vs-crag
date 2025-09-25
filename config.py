"""
Configuration file for RAG vs CRAG comparison project.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")          # For embeddings only
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # For chat completions
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# OpenRouter settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Sample data
SAMPLE_URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Deep_learning"
]

SAMPLE_QUESTIONS = [
    "What is artificial intelligence?",
    "What are the key principles of corporate governance and risk management?",
    "How do financial derivatives work in modern markets?",
    "How does machine learning work?",
    "What are the applications of natural language processing?",
    "Explain deep learning concepts",
    "What are the compliance requirements for data protection regulations?",
    "How does insider trading law apply to corporate executives?"
]

# Model configurations (OpenRouter compatible)
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding via OpenRouter
CHAT_MODEL = "openai/gpt-4o-mini"  # OpenRouter format for OpenAI models
TEMPERATURE = 0

# Retrieval settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 10

# Resource limits for demo
MAX_URLS = 5
MAX_CHUNKS = 2000

# CRAG specific settings
RELEVANCE_THRESHOLD = 0.7
WEB_SEARCH_RESULTS = 3


class Constants:
    """Application constants and messages."""
    
    # Error messages
    ERROR_MESSAGES = {
        'NO_DOCS_LOADED': "No documents loaded. Please load documents first.",
        'INSUFFICIENT_INFO': "I don't have enough relevant information to answer this question.",
        'LOAD_FAILED': "No documents were loaded successfully",
        'EMPTY_QUESTION': "Question cannot be empty",
        'QUESTION_TOO_LONG': "Question too long (max {max_length} characters)"
    }
    
    # UI constants
    DOC_PREVIEW_LENGTH = 100
    ANSWER_PREVIEW_LENGTH = 300
    MAX_QUESTION_LENGTH = 1000
    
    # System identifiers
    SYSTEM_TYPES = {
        'RAG': 'Basic RAG',
        'CRAG': 'CRAG'
    }
