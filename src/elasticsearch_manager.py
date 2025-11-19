"""ElasticSearch management for keyword-based retrieval."""

import logging
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, ConnectionError
import json

from config import ES_HOST, ES_PORT, ES_INDEX_NAME, TOP_K_ES

logger = logging.getLogger(__name__)

class ElasticSearchManager:
    """Manages ElasticSearch operations for keyword-based document retrieval."""
    
    def __init__(self, host: str = ES_HOST, port: int = ES_PORT):
        """Initialize ElasticSearch connection."""
        self.host = host
        self.port = port
        self.index_name = ES_INDEX_NAME
        
        try:
            self.client = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
            self._test_connection()
            logger.info(f"Connected to ElasticSearch at {host}:{port}")
        except ConnectionError as e:
            logger.error(f"Failed to connect to ElasticSearch: {e}")
            self.client = None
    
    def _test_connection(self):
        """Test ElasticSearch connection."""
        if self.client:
            try:
                info = self.client.info()
                logger.info(f"ElasticSearch cluster: {info['cluster_name']}")
            except Exception as e:
                logger.error(f"ElasticSearch connection test failed: {e}")
                self.client = None
    
    def create_index(self, mapping: Optional[Dict] = None):
        """Create ElasticSearch index with custom mapping."""
        if not self.client:
            logger.error("ElasticSearch client not available")
            return False
        
        # Default mapping for medical documents
        if mapping is None:
            mapping = {
                "mappings": {
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "section_title": {"type": "text", "analyzer": "standard"},
                        "content": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "english": {"type": "text", "analyzer": "english"}
                            }
                        },
                        "page_number": {"type": "integer"},
                        "start_char": {"type": "integer"},
                        "end_char": {"type": "integer"},
                        "chunk_length": {"type": "integer"},
                        "is_bold": {"type": "boolean"},
                        "font_size": {"type": "float"},
                        "metadata": {"type": "object"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "medical_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
                            }
                        }
                    }
                }
            }
        
        try:
            if self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return True
            
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created ElasticSearch index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def index_documents(self, documents: List[Dict]):
        """Index documents in ElasticSearch."""
        if not self.client:
            logger.error("ElasticSearch client not available")
            return False
        
        if not self.client.indices.exists(index=self.index_name):
            self.create_index()
        
        logger.info(f"Indexing {len(documents)} documents in ElasticSearch")
        
        try:
            # Bulk index documents
            actions = []
            for doc in documents:
                action = {
                    "_index": self.index_name,
                    "_id": doc.get("chunk_id"),
                    "_source": doc
                }
                actions.append(action)
            
            from elasticsearch.helpers import bulk
            success, failed = bulk(self.client, actions, chunk_size=100)
            
            if failed:
                logger.warning(f"Failed to index {len(failed)} documents")
            else:
                logger.info(f"Successfully indexed {success} documents")
            
            # Refresh index
            self.client.indices.refresh(index=self.index_name)
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    def search_keywords(self, query: str, k: int = TOP_K_ES, filters: Optional[Dict] = None) -> List[Dict]:
        """Search documents using BM25 keyword matching."""
        if not self.client:
            logger.error("ElasticSearch client not available")
            return []
        
        try:
            # Build search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "content^2",  # Boost content field
                                        "section_title^1.5",  # Boost section titles
                                        "content.english"  # Use English analyzer
                                    ],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ]
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        },
                        "section_title": {
                            "fragment_size": 50,
                            "number_of_fragments": 1
                        }
                    }
                },
                "size": k
            }
            
            # Add filters if provided
            if filters:
                search_body["query"]["bool"]["filter"] = []
                for field, value in filters.items():
                    if isinstance(value, list):
                        search_body["query"]["bool"]["filter"].append({
                            "terms": {field: value}
                        })
                    else:
                        search_body["query"]["bool"]["filter"].append({
                            "term": {field: value}
                        })
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'score': hit['_score'],
                    'document': hit['_source'],
                    'highlights': hit.get('highlight', {})
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_semantic_keywords(self, query: str, semantic_results: List, k: int = TOP_K_ES) -> List[Dict]:
        """Combine semantic and keyword search results."""
        # Get keyword search results
        keyword_results = self.search_keywords(query, k=k)
        
        # Create a combined ranking
        combined_results = {}
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result['document']['chunk_id']
            combined_results[chunk_id] = {
                'keyword_score': result['score'],
                'semantic_score': 0,
                'document': result['document'],
                'highlights': result['highlights']
            }
        
        # Add semantic results
        for semantic_score, metadata in semantic_results:
            chunk_id = metadata.get('chunk_id')
            if chunk_id:
                if chunk_id in combined_results:
                    combined_results[chunk_id]['semantic_score'] = semantic_score
                else:
                    combined_results[chunk_id] = {
                        'keyword_score': 0,
                        'semantic_score': semantic_score,
                        'document': metadata,
                        'highlights': {}
                    }
        
        # Rank by combined score
        ranked_results = []
        for chunk_id, data in combined_results.items():
            # Simple linear combination of scores
            combined_score = (data['keyword_score'] * 0.6 + 
                            data['semantic_score'] * 0.4)
            
            ranked_results.append({
                'chunk_id': chunk_id,
                'combined_score': combined_score,
                'keyword_score': data['keyword_score'],
                'semantic_score': data['semantic_score'],
                'document': data['document'],
                'highlights': data['highlights']
            })
        
        # Sort by combined score
        ranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ranked_results[:k]
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the ElasticSearch index."""
        if not self.client:
            return {"status": "not_connected"}
        
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"status": "no_index"}
            
            stats = self.client.indices.stats(index=self.index_name)
            return {
                "status": "active",
                "total_documents": stats['indices'][self.index_name]['total']['docs']['count'],
                "index_size": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def delete_index(self):
        """Delete the ElasticSearch index."""
        if not self.client:
            logger.error("ElasticSearch client not available")
            return False
        
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted index: {self.index_name}")
                return True
            else:
                logger.info(f"Index {self.index_name} does not exist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False
    
    def clear_all_documents(self):
        """Clear all documents from the index."""
        if not self.client:
            logger.error("ElasticSearch client not available")
            return False
        
        try:
            # Delete all documents but keep the index structure
            query = {"query": {"match_all": {}}}
            self.client.delete_by_query(index=self.index_name, body=query)
            self.client.indices.refresh(index=self.index_name)
            logger.info("Cleared all documents from index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return False
