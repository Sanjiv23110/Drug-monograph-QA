"""Embedding generation and vector database management."""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

from config import (
    PUBMEDBERT_MODEL, 
    FAISS_INDEX_TYPE, 
    EMBEDDING_DIMENSION,
    VECTOR_DB_DIR,
    TOP_K_FAISS
)

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation and FAISS vector database operations."""
    
    def __init__(self, model_name: str = PUBMEDBERT_MODEL):
        """Initialize the embedding manager with PubMedBERT."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading PubMedBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize FAISS index
        self.index = None
        self.id_to_metadata = {}
        self.metadata_file = VECTOR_DB_DIR / "metadata.pkl"
        self.index_file = VECTOR_DB_DIR / "faiss_index.bin"
        
        # Load existing index if available
        self.load_index()
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Use CLS token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Normalize embeddings for cosine similarity
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def create_index(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """Create FAISS index from embeddings."""
        logger.info(f"Creating FAISS index with {len(embeddings)} embeddings")
        
        # Create index
        if FAISS_INDEX_TYPE == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        elif FAISS_INDEX_TYPE == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        else:
            raise ValueError(f"Unsupported FAISS index type: {FAISS_INDEX_TYPE}")
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.id_to_metadata = {i: metadata for i, metadata in enumerate(metadata_list)}
        
        # Save index and metadata
        self.save_index()
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
    
    def add_to_index(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """Add new embeddings to existing index."""
        if self.index is None:
            self.create_index(embeddings, metadata_list)
            return
        
        logger.info(f"Adding {len(embeddings)} embeddings to existing index")
        
        # Get current size
        current_size = self.index.ntotal
        
        # Add embeddings
        self.index.add(embeddings.astype('float32'))
        
        # Update metadata
        for i, metadata in enumerate(metadata_list):
            self.id_to_metadata[current_size + i] = metadata
        
        # Save updated index
        self.save_index()
        
        logger.info(f"Index now contains {self.index.ntotal} vectors")
    
    def search(self, query_text: str, k: int = TOP_K_FAISS) -> List[Tuple[float, Dict]]:
        """Search for similar documents using semantic similarity."""
        if self.index is None:
            logger.warning("No index available for search")
            return []
        
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query_text])
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.id_to_metadata:
                results.append((float(score), self.id_to_metadata[idx]))
        
        return results
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        VECTOR_DB_DIR.mkdir(exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            logger.info(f"Saved FAISS index to {self.index_file}")
        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)
        logger.info(f"Saved metadata to {self.metadata_file}")
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                
                with open(self.metadata_file, 'rb') as f:
                    self.id_to_metadata = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self.index = None
                self.id_to_metadata = {}
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        if self.index is None:
            return {"status": "no_index", "total_vectors": 0}
        
        return {
            "status": "active",
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "index_type": FAISS_INDEX_TYPE,
            "model_name": self.model_name
        }
    
    def clear_index(self):
        """Clear the current index and metadata."""
        self.index = None
        self.id_to_metadata = {}
        
        # Remove files
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        logger.info("Index cleared")
