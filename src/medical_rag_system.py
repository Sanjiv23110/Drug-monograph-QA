"""Main Medical RAG System that integrates all components."""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

from pdf_processor import PDFProcessor, ProcessedDocument
from embedding_manager import EmbeddingManager
from elasticsearch_manager import ElasticSearchManager
from rag_pipeline import MedicalRAGPipeline
from config import (
    DATA_DIR, 
    LOGS_DIR,
    TOP_K_FAISS,
    TOP_K_ES,
    RERANK_TOP_K
)

logger = logging.getLogger(__name__)

class MedicalRAGSystem:
    """Main system that orchestrates the medical document RAG pipeline."""
    
    def __init__(self, 
                 spacy_model: str = "en_core_web_sm",
                 pubmedbert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 biogpt_model: str = "microsoft/biogpt",
                 es_host: str = "localhost",
                 es_port: int = 9200):
        """Initialize the Medical RAG System."""
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Initializing Medical RAG System")
        
        # Initialize components
        self.pdf_processor = PDFProcessor(spacy_model=spacy_model)
        self.embedding_manager = EmbeddingManager(model_name=pubmedbert_model)
        self.es_manager = ElasticSearchManager(host=es_host, port=es_port)
        self.rag_pipeline = MedicalRAGPipeline()  # Uses PRIMARY_MODEL from config
        
        # System state
        self.processed_documents = {}
        self.is_initialized = True
        
        logger.info("Medical RAG System initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        LOGS_DIR.mkdir(exist_ok=True)
        
        log_file = LOGS_DIR / f"medical_rag_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def add_documents(self, pdf_paths: Union[str, Path, List[Union[str, Path]]]) -> Dict:
        """Add PDF documents to the system."""
        if isinstance(pdf_paths, (str, Path)):
            pdf_paths = [pdf_paths]
        
        results = {
            'successful': [],
            'failed': [],
            'total_processed': 0,
            'total_chunks': 0
        }
        
        logger.info(f"Processing {len(pdf_paths)} PDF documents")
        
        all_chunks = []
        all_metadata = []
        
        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                results['failed'].append({
                    'path': str(pdf_path),
                    'error': 'File not found'
                })
                continue
            
            try:
                # Process PDF
                processed_doc = self.pdf_processor.process_pdf(pdf_path)
                
                # Store processed document
                doc_id = str(uuid.uuid4())
                self.processed_documents[doc_id] = processed_doc
                
                # Create chunks with metadata
                chunks = self.pdf_processor.create_chunks_with_boundaries(processed_doc.full_text)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    
                    chunk_metadata = {
                        'chunk_id': chunk_id,
                        'document_id': doc_id,
                        'filename': processed_doc.filename,
                        'section_title': 'Document',  # Will be enhanced with section detection
                        'content': chunk['text'],
                        'page_number': 1,  # Will be enhanced with actual page mapping
                        'start_char': chunk['start_char'],
                        'end_char': chunk['end_char'],
                        'chunk_length': chunk['length'],
                        'is_bold': False,
                        'font_size': None,
                        'metadata': processed_doc.metadata
                    }
                    
                    all_chunks.append(chunk['text'])
                    all_metadata.append(chunk_metadata)
                
                results['successful'].append({
                    'path': str(pdf_path),
                    'document_id': doc_id,
                    'chunks_created': len(chunks)
                })
                results['total_chunks'] += len(chunks)
                
                logger.info(f"Successfully processed {pdf_path.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results['failed'].append({
                    'path': str(pdf_path),
                    'error': str(e)
                })
        
        # Generate embeddings and add to vector database
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            embeddings = self.embedding_manager.generate_embeddings(all_chunks)
            
            # Add to FAISS index
            if self.embedding_manager.index is None:
                self.embedding_manager.create_index(embeddings, all_metadata)
            else:
                self.embedding_manager.add_to_index(embeddings, all_metadata)
            
            # Index in ElasticSearch
            self.es_manager.create_index()
            self.es_manager.index_documents(all_metadata)
            
            logger.info("Successfully indexed all documents")
        
        results['total_processed'] = len(results['successful'])
        return results
    
    def search_documents(self, query: str, top_k: int = 5) -> Dict:
        """Search documents using both semantic and keyword search."""
        logger.info(f"Searching for: {query}")
        
        # Semantic search using FAISS
        semantic_results = self.embedding_manager.search(query, k=TOP_K_FAISS)
        
        # Keyword search using ElasticSearch
        keyword_results = self.es_manager.search_keywords(query, k=TOP_K_ES)
        
        # Combine and rerank results
        combined_results = self.es_manager.search_semantic_keywords(
            query, semantic_results, k=top_k
        )
        
        return {
            'query': query,
            'semantic_results': semantic_results,
            'keyword_results': keyword_results,
            'combined_results': combined_results,
            'total_results': len(combined_results)
        }
    
    def ask_question(self, question: str) -> Dict:
        """Answer a medical question using the RAG pipeline."""
        logger.info(f"Answering question: {question}")
        
        # Search for relevant documents
        search_results = self.search_documents(question, top_k=RERANK_TOP_K)
        
        # Extract semantic and keyword results
        semantic_results = search_results['semantic_results']
        keyword_results = search_results['keyword_results']
        
        # Generate answer using RAG pipeline
        answer_result = self.rag_pipeline.answer_question(
            question, semantic_results, keyword_results
        )
        
        # Add search metadata
        answer_result['search_metadata'] = {
            'total_semantic_results': len(semantic_results),
            'total_keyword_results': len(keyword_results),
            'search_query': question
        }
        
        return answer_result
    
    def batch_ask_questions(self, questions: List[str]) -> List[Dict]:
        """Answer multiple questions in batch."""
        logger.info(f"Answering {len(questions)} questions in batch")
        
        # Search for all questions
        search_results_list = []
        for question in questions:
            search_results = self.search_documents(question, top_k=RERANK_TOP_K)
            search_results_list.append(search_results)
        
        # Extract results for batch processing
        semantic_results_list = [sr['semantic_results'] for sr in search_results_list]
        keyword_results_list = [sr['keyword_results'] for sr in search_results_list]
        
        # Generate answers using batch processing
        answers = self.rag_pipeline.batch_answer_questions(
            questions, semantic_results_list, keyword_results_list
        )
        
        return answers
    
    def get_system_stats(self) -> Dict:
        """Get system statistics and status."""
        faiss_stats = self.embedding_manager.get_index_stats()
        es_stats = self.es_manager.get_index_stats()
        
        return {
            'system_status': 'active' if self.is_initialized else 'inactive',
            'processed_documents': len(self.processed_documents),
            'faiss_index': faiss_stats,
            'elasticsearch_index': es_stats,
            'components': {
                'pdf_processor': 'active',
                'embedding_manager': 'active' if faiss_stats['status'] == 'active' else 'inactive',
                'elasticsearch_manager': 'active' if es_stats['status'] == 'active' else 'inactive',
                'rag_pipeline': 'active'
            }
        }
    
    def clear_all_data(self):
        """Clear all indexed data and processed documents."""
        logger.info("Clearing all system data")
        
        # Clear FAISS index
        self.embedding_manager.clear_index()
        
        # Clear ElasticSearch index
        self.es_manager.clear_all_documents()
        
        # Clear processed documents
        self.processed_documents.clear()
        
        logger.info("All system data cleared")
    
    def export_document_data(self, document_id: str) -> Optional[Dict]:
        """Export processed document data."""
        if document_id not in self.processed_documents:
            logger.warning(f"Document {document_id} not found")
            return None
        
        doc = self.processed_documents[document_id]
        return {
            'document_id': document_id,
            'filename': doc.filename,
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'page_number': section.page_number,
                    'is_bold': section.is_bold,
                    'font_size': section.font_size
                }
                for section in doc.sections
            ],
            'metadata': doc.metadata,
            'full_text': doc.full_text
        }
    
    def get_available_documents(self) -> List[Dict]:
        """Get list of available processed documents."""
        return [
            {
                'document_id': doc_id,
                'filename': doc.filename,
                'metadata': doc.metadata
            }
            for doc_id, doc in self.processed_documents.items()
        ]
