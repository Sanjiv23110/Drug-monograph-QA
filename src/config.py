"""Configuration settings for the Medical Document RAG System."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, VECTOR_DB_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BIOGPT_MODEL = "microsoft/biogpt"
PRIMARY_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"  # PubMedBERT as primary
FALLBACK_MODEL = "microsoft/biogpt"  # BioGPT as fallback

# Extractive QA model (biomedical SQuAD fine-tuned)
QA_MODEL_NAME = "ktrapeznikov/biobert_v1.1_pubmed_squad_v2"

# PDF Processing
MAX_PDF_SIZE_MB = 50
SUPPORTED_PDF_FORMATS = ['.pdf']

# Text Processing - Optimized for better accuracy
CHUNK_SIZE = 768              # Increased for more context
CHUNK_OVERLAP = 100           # Increased for better continuity
MIN_CHUNK_SIZE = 150          # Increased for more substantial chunks

# Vector Database
FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner Product for cosine similarity
EMBEDDING_DIMENSION = 768  # PubMedBERT embedding dimension

# ElasticSearch
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_INDEX_NAME = "medical_documents"

# Retrieval - Enhanced for better accuracy
TOP_K_FAISS = 20             # Increased for more semantic results
TOP_K_ES = 20                # Increased for more keyword results
RERANK_TOP_K = 10            # Increased for better final results

# RAG Settings - Optimized for better answers
MAX_CONTEXT_LENGTH = 4096    # Increased for more context
TEMPERATURE = 0.3            # Increased for more natural responses
MAX_NEW_TOKENS = 1024        # Increased for longer, detailed answers

# Extractive QA settings
WINDOW_SIZE_TOKENS = 512
WINDOW_STRIDE = 160
MAX_ANSWER_LENGTH = 150
MAX_SPANS_FOR_COMPREHENSIVE = 5

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
