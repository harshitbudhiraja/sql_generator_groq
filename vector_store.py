# vector_store.py
# Module for handling vector embeddings and similarity search

import os
import logging
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vector retrieval parameters
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_DB_PATH = 'vector_db'
SIMILARITY_THRESHOLD = 0.8  # Minimum similarity score to use retrieved results
TOP_K = 3  # Number of similar examples to retrieve

class VectorStore:
    """Vector database for storing and retrieving similar queries"""
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        """Initialize the vector store with the embedding model"""
        self.model = SentenceTransformer(model_name)
        self.embedding_size = self.model.get_sentence_embedding_dimension()
        
        # For NL to SQL
        self.nl_to_sql_index = None
        self.nl_to_sql_data = []
        
        # For SQL correction
        self.incorrect_sql_index = None
        self.incorrect_sql_data = []
        
        # Create directory for vector DB if it doesn't exist
        Path(VECTOR_DB_PATH).mkdir(exist_ok=True)
    
    def embed_text(self, text):
        """Generate embeddings for the text"""
        return self.model.encode(text)
    
    def build_nl_to_sql_index(self, data):
        """Build index for NL to SQL conversion data"""
        if not data:
            logger.warning("No data provided to build NL to SQL index")
            return
        
        nl_queries = [item['NL'] for item in data]
        embeddings = self.model.encode(nl_queries)
        
        # Create FAISS index
        self.nl_to_sql_index = faiss.IndexFlatL2(self.embedding_size)
        faiss.normalize_L2(embeddings)
        self.nl_to_sql_index.add(embeddings)
        
        # Store the original data
        self.nl_to_sql_data = data
        
        # Save the index and data
        self._save_index('nl_to_sql')
        
        logger.info(f"Built NL to SQL index with {len(data)} examples")
    
    def build_sql_correction_index(self, data):
        """Build index for SQL correction data"""
        if not data:
            logger.warning("No data provided to build SQL correction index")
            return
        
        incorrect_queries = [item['IncorrectQuery'] for item in data]
        embeddings = self.model.encode(incorrect_queries)
        
        # Create FAISS index
        self.incorrect_sql_index = faiss.IndexFlatL2(self.embedding_size)
        faiss.normalize_L2(embeddings)
        self.incorrect_sql_index.add(embeddings)
        
        # Store the original data
        self.incorrect_sql_data = data
        
        # Save the index and data
        self._save_index('sql_correction')
        
        logger.info(f"Built SQL correction index with {len(data)} examples")
    
    def find_similar_nl_queries(self, nl_query, top_k=TOP_K):
        """Find similar natural language queries and their SQL equivalents"""
        if self.nl_to_sql_index is None:
            logger.warning("NL to SQL index not built yet")
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([nl_query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar queries
        distances, indices = self.nl_to_sql_index.search(query_embedding, top_k)
        
        # Prepare results with similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.nl_to_sql_data):
                similarity = 1 - distances[0][i]  # Convert L2 distance to similarity score
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append({
                        'nl_query': self.nl_to_sql_data[idx]['NL'],
                        'sql_query': self.nl_to_sql_data[idx]['Query'],
                        'similarity': float(similarity)
                    })
        
        return results
    
    def find_similar_incorrect_queries(self, incorrect_query, top_k=TOP_K):
        """Find similar incorrect SQL queries and their corrections"""
        if self.incorrect_sql_index is None:
            logger.warning("SQL correction index not built yet")
            return []
        
        # Generate embedding for the query
        query_embedding = self.model.encode([incorrect_query])
        faiss.normalize_L2(query_embedding)
        
        # Search for similar queries
        distances, indices = self.incorrect_sql_index.search(query_embedding, top_k)
        
        # Prepare results with similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.incorrect_sql_data):
                similarity = 1 - distances[0][i]  # Convert L2 distance to similarity score
                if similarity >= SIMILARITY_THRESHOLD:
                    results.append({
                        'incorrect_query': self.incorrect_sql_data[idx]['IncorrectQuery'],
                        'correct_query': self.incorrect_sql_data[idx]['CorrectQuery'],
                        'similarity': float(similarity)
                    })
        
        return results
    
    def _save_index(self, index_type):
        """Save the index and data to disk"""
        if index_type == 'nl_to_sql':
            # Save FAISS index
            faiss.write_index(self.nl_to_sql_index, f"{VECTOR_DB_PATH}/nl_to_sql.index")
            # Save data
            with open(f"{VECTOR_DB_PATH}/nl_to_sql_data.pkl", 'wb') as f:
                pickle.dump(self.nl_to_sql_data, f)
        elif index_type == 'sql_correction':
            # Save FAISS index
            faiss.write_index(self.incorrect_sql_index, f"{VECTOR_DB_PATH}/incorrect_sql.index")
            # Save data
            with open(f"{VECTOR_DB_PATH}/incorrect_sql_data.pkl", 'wb') as f:
                pickle.dump(self.incorrect_sql_data, f)
    
    def load_indexes(self):
        """Load indexes and data from disk if they exist"""
        try:
            # Try to load NL to SQL index
            if os.path.exists(f"{VECTOR_DB_PATH}/nl_to_sql.index") and os.path.exists(f"{VECTOR_DB_PATH}/nl_to_sql_data.pkl"):
                self.nl_to_sql_index = faiss.read_index(f"{VECTOR_DB_PATH}/nl_to_sql.index")
                with open(f"{VECTOR_DB_PATH}/nl_to_sql_data.pkl", 'rb') as f:
                    self.nl_to_sql_data = pickle.load(f)
                logger.info(f"Loaded NL to SQL index with {len(self.nl_to_sql_data)} examples")
            
            # Try to load SQL correction index
            if os.path.exists(f"{VECTOR_DB_PATH}/incorrect_sql.index") and os.path.exists(f"{VECTOR_DB_PATH}/incorrect_sql_data.pkl"):
                self.incorrect_sql_index = faiss.read_index(f"{VECTOR_DB_PATH}/incorrect_sql.index")
                with open(f"{VECTOR_DB_PATH}/incorrect_sql_data.pkl", 'rb') as f:
                    self.incorrect_sql_data = pickle.load(f)
                logger.info(f"Loaded SQL correction index with {len(self.incorrect_sql_data)} examples")
            
            return True
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            return False