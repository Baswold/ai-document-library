"""
Advanced Semantic Search Engine for AI Document Library
Uses vector embeddings for intelligent document retrieval
"""

import numpy as np
import sqlite3
import pickle
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearchEngine:
    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic search engine
        
        Args:
            db_path: Path to SQLite database
            model_name: Sentence transformer model name
        """
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunk_ids = []
        self.embedding_cache = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database tables
        self.init_embedding_tables()
        
    def init_embedding_tables(self):
        """Create tables for storing embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for storing document embeddings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_id INTEGER,
                embedding BLOB,
                embedding_model TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id),
                FOREIGN KEY (chunk_id) REFERENCES document_chunks (id)
            )
        ''')
        
        # Table for document relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc1_id INTEGER,
                doc2_id INTEGER,
                relationship_type TEXT,
                similarity_score REAL,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc1_id) REFERENCES documents (id),
                FOREIGN KEY (doc2_id) REFERENCES documents (id)
            )
        ''')
        
        # Table for search analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                results_count INTEGER,
                avg_similarity REAL,
                search_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            self.logger.info(f"Loading sentence transformer model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                # Fallback to smaller model
                self.model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name)
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text"""
        self.load_model()
        
        # Clean and truncate text
        text = text.strip()[:512]  # Limit to 512 chars for efficiency
        
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        embedding = self.model.encode([text])[0]
        self.embedding_cache[text] = embedding
        return embedding
        
    def store_embedding(self, document_id: int, chunk_id: int, text: str):
        """Store embedding for a document chunk"""
        try:
            embedding = self.generate_embedding(text)
            embedding_blob = pickle.dumps(embedding)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO document_embeddings 
                (document_id, chunk_id, embedding, embedding_model)
                VALUES (?, ?, ?, ?)
            ''', (document_id, chunk_id, embedding_blob, self.model_name))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing embedding: {e}")
            return False
            
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        self.logger.info("Building FAISS index...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, embedding FROM document_embeddings
            WHERE embedding_model = ?
        ''', (self.model_name,))
        
        embeddings = []
        chunk_ids = []
        
        for chunk_id, embedding_blob in cursor.fetchall():
            embedding = pickle.loads(embedding_blob)
            embeddings.append(embedding)
            chunk_ids.append(chunk_id)
            
        conn.close()
        
        if not embeddings:
            self.logger.warning("No embeddings found for indexing")
            return False
            
        # Create FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        self.chunk_ids = chunk_ids
        self.logger.info(f"FAISS index built with {len(embeddings)} embeddings")
        return True
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform semantic search using embeddings
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        if self.index is None:
            if not self.build_faiss_index():
                return []
                
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Search in FAISS index
        similarities, indices = self.index.search(query_vector, min(top_k, len(self.chunk_ids)))
        
        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # No more results
                break
                
            chunk_id = self.chunk_ids[idx]
            
            # Get chunk details
            cursor.execute('''
                SELECT dc.chunk_text, dc.document_id, d.filename, d.summary
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.id = ?
            ''', (chunk_id,))
            
            row = cursor.fetchone()
            if row:
                results.append({
                    'chunk_id': chunk_id,
                    'chunk_text': row[0],
                    'document_id': row[1],
                    'filename': row[2],
                    'summary': row[3],
                    'similarity_score': float(similarity),
                    'rank': i + 1
                })
                
        conn.close()
        
        # Store search analytics
        self.store_search_analytics(query, len(results), similarities[0])
        
        return results
        
    def hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict]:
        """
        Combine semantic search with keyword search
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic vs keyword search (0-1)
        """
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k * 2)
        
        # Get keyword results (existing functionality)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine and rerank results
        combined_results = self.combine_search_results(
            semantic_results, keyword_results, semantic_weight
        )
        
        return combined_results[:top_k]
        
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Traditional keyword search in document chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT dc.id, dc.chunk_text, dc.document_id, d.filename, d.summary
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.chunk_text LIKE ?
            ORDER BY dc.id DESC
            LIMIT ?
        ''', (f'%{query}%', top_k))
        
        results = []
        for i, row in enumerate(cursor.fetchall()):
            results.append({
                'chunk_id': row[0],
                'chunk_text': row[1],
                'document_id': row[2],
                'filename': row[3],
                'summary': row[4],
                'similarity_score': 1.0,  # Placeholder
                'rank': i + 1
            })
            
        conn.close()
        return results
        
    def combine_search_results(self, semantic_results: List[Dict], 
                             keyword_results: List[Dict], 
                             semantic_weight: float) -> List[Dict]:
        """Combine and rerank semantic and keyword search results"""
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['chunk_id']
            combined[chunk_id] = result.copy()
            combined[chunk_id]['semantic_score'] = result['similarity_score']
            combined[chunk_id]['keyword_score'] = 0.0
            
        # Add keyword results
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined:
                combined[chunk_id]['keyword_score'] = 1.0
            else:
                combined[chunk_id] = result.copy()
                combined[chunk_id]['semantic_score'] = 0.0
                combined[chunk_id]['keyword_score'] = 1.0
                
        # Calculate combined scores
        for chunk_id in combined:
            semantic_score = combined[chunk_id]['semantic_score']
            keyword_score = combined[chunk_id]['keyword_score']
            combined[chunk_id]['combined_score'] = (
                semantic_weight * semantic_score + 
                (1 - semantic_weight) * keyword_score
            )
            
        # Sort by combined score
        results = list(combined.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
        
    def find_similar_documents(self, document_id: int, top_k: int = 5) -> List[Dict]:
        """Find documents similar to a given document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all chunks for the source document
        cursor.execute('''
            SELECT chunk_text FROM document_chunks
            WHERE document_id = ?
        ''', (document_id,))
        
        chunks = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not chunks:
            return []
            
        # Create document embedding by averaging chunk embeddings
        doc_embedding = np.mean([self.generate_embedding(chunk) for chunk in chunks], axis=0)
        
        # Search for similar chunks
        if self.index is None:
            self.build_faiss_index()
            
        query_vector = np.array([doc_embedding]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        similarities, indices = self.index.search(query_vector, top_k * 3)
        
        # Group by document and calculate average similarity
        doc_similarities = {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                break
                
            chunk_id = self.chunk_ids[idx]
            cursor.execute('''
                SELECT dc.document_id, d.filename, d.summary
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.id = ?
            ''', (chunk_id,))
            
            row = cursor.fetchone()
            if row and row[0] != document_id:  # Exclude self
                doc_id = row[0]
                if doc_id not in doc_similarities:
                    doc_similarities[doc_id] = {
                        'document_id': doc_id,
                        'filename': row[1],
                        'summary': row[2],
                        'similarities': []
                    }
                doc_similarities[doc_id]['similarities'].append(float(similarity))
                
        conn.close()
        
        # Calculate average similarities and sort
        results = []
        for doc_data in doc_similarities.values():
            avg_similarity = np.mean(doc_data['similarities'])
            results.append({
                'document_id': doc_data['document_id'],
                'filename': doc_data['filename'],
                'summary': doc_data['summary'],
                'similarity_score': avg_similarity,
                'chunk_count': len(doc_data['similarities'])
            })
            
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
        
    def process_new_documents(self, force_rebuild: bool = False):
        """Process embeddings for documents that don't have them yet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find chunks without embeddings
        cursor.execute('''
            SELECT dc.id, dc.document_id, dc.chunk_text
            FROM document_chunks dc
            LEFT JOIN document_embeddings de ON dc.id = de.chunk_id 
                AND de.embedding_model = ?
            WHERE de.id IS NULL
        ''', (self.model_name,))
        
        chunks_to_process = cursor.fetchall()
        conn.close()
        
        if not chunks_to_process:
            self.logger.info("All chunks already have embeddings")
            return True
            
        self.logger.info(f"Processing embeddings for {len(chunks_to_process)} chunks...")
        
        processed = 0
        for chunk_id, document_id, chunk_text in chunks_to_process:
            if self.store_embedding(document_id, chunk_id, chunk_text):
                processed += 1
                if processed % 10 == 0:
                    self.logger.info(f"Processed {processed}/{len(chunks_to_process)} embeddings")
                    
        self.logger.info(f"Generated embeddings for {processed} chunks")
        
        # Rebuild index if requested or if new embeddings were added
        if force_rebuild or processed > 0:
            self.build_faiss_index()
            
        return True
        
    def store_search_analytics(self, query: str, results_count: int, similarities: np.ndarray):
        """Store search analytics for performance monitoring"""
        try:
            avg_similarity = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO search_analytics 
                (query, results_count, avg_similarity, search_time)
                VALUES (?, ?, ?, ?)
            ''', (query, results_count, avg_similarity, 0.0))  # TODO: Add timing
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing search analytics: {e}")
            
    def get_search_analytics(self, days: int = 7) -> Dict:
        """Get search analytics for the past N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_searches,
                AVG(results_count) as avg_results,
                AVG(avg_similarity) as avg_similarity,
                COUNT(DISTINCT query) as unique_queries
            FROM search_analytics
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_searches': stats[0] or 0,
            'avg_results_per_search': stats[1] or 0,
            'avg_similarity_score': stats[2] or 0,
            'unique_queries': stats[3] or 0
        }