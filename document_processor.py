"""
Document processing and AI cataloging system with semantic search integration
"""

import os
import sqlite3
import requests
import json
from pathlib import Path
import PyPDF2
import docx
import re
from typing import List, Dict, Optional
from semantic_search import SemanticSearchEngine

class DocumentProcessor:
    def __init__(self, db_path: str, config: dict):
        self.db_path = db_path
        self.config = config
        self.semantic_search = SemanticSearchEngine(db_path)
        
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.pdf':
                return self.extract_pdf_text(file_path)
            elif file_path.suffix.lower() == '.docx':
                return self.extract_docx_text(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                return self.extract_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    def extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            # Fallback for problematic PDFs
            return f"PDF text extraction failed: {str(e)}"
    
    def extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            return f"DOCX text extraction failed: {str(e)}"
    
    def extract_text_file(self, file_path: Path) -> str:
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                return f"Text file reading failed: {str(e)}"
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better RAG performance"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def analyze_document_with_ai(self, text: str, filename: str) -> Dict[str, str]:
        """Use AI to analyze document and extract metadata"""
        # Truncate text for analysis (use first 2000 characters)
        analysis_text = text[:2000] + "..." if len(text) > 2000 else text
        
        if self.config['ai_type'] == 'local':
            return self.analyze_with_ollama(analysis_text, filename)
        else:
            return self.analyze_with_api(analysis_text, filename)
    
    def analyze_with_ollama(self, text: str, filename: str) -> Dict[str, str]:
        """Analyze document using local Ollama model"""
        prompt = f"""Analyze this document and provide:
1. A brief summary (2-3 sentences)
2. Key topics/themes (comma-separated list)
3. Document type (report, article, manual, etc.)

Document filename: {filename}
Document content:
{text}

Respond in this exact format:
SUMMARY: [your summary here]
TOPICS: [topic1, topic2, topic3]
TYPE: [document type]"""
        
        try:
            url = f"{self.config['ollama_url']}/api/generate"
            payload = {
                "model": self.config.get('local_model', 'gemma2:2b'),
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            analysis_text = result.get('response', '')
            
            return self.parse_analysis_response(analysis_text)
            
        except Exception as e:
            return {
                'summary': f'Document: {filename}',
                'topics': 'document',
                'doc_type': 'unknown'
            }
    
    def analyze_with_api(self, text: str, filename: str) -> Dict[str, str]:
        """Analyze document using cloud API"""
        # Placeholder for API implementation
        # This would implement OpenAI/Anthropic API calls
        return {
            'summary': f'Document analysis via API for {filename}',
            'topics': 'api, analysis',
            'doc_type': 'document'
        }
    
    def parse_analysis_response(self, response: str) -> Dict[str, str]:
        """Parse AI response into structured data"""
        result = {
            'summary': '',
            'topics': '',
            'doc_type': 'document'
        }
        
        # Extract summary
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?:\n|TOPICS:|TYPE:|$)', response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            result['summary'] = summary_match.group(1).strip()
        
        # Extract topics
        topics_match = re.search(r'TOPICS:\s*(.+?)(?:\n|TYPE:|$)', response, re.IGNORECASE)
        if topics_match:
            result['topics'] = topics_match.group(1).strip()
        
        # Extract type
        type_match = re.search(r'TYPE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if type_match:
            result['doc_type'] = type_match.group(1).strip()
        
        return result
    
    def process_document(self, document_id: int, file_path: str) -> bool:
        """Process a single document: extract text, analyze, and store chunks"""
        try:
            # Extract text
            text = self.extract_text_from_file(file_path)
            
            if not text or text.startswith("Error"):
                return False
            
            # Get AI analysis
            filename = Path(file_path).name
            analysis = self.analyze_document_with_ai(text, filename)
            
            # Update document record
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE documents 
                SET summary = ?, topics = ?, doc_type = ?, processed = TRUE
                WHERE id = ?
            ''', (analysis['summary'], analysis['topics'], analysis['doc_type'], document_id))
            
            # Store text chunks and generate embeddings
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                cursor.execute('''
                    INSERT INTO document_chunks (document_id, chunk_text, chunk_index)
                    VALUES (?, ?, ?)
                ''', (document_id, chunk, i))
                
                # Get the chunk ID for embedding generation
                chunk_id = cursor.lastrowid
                
                # Generate embedding for this chunk (in background)
                try:
                    self.semantic_search.store_embedding(document_id, chunk_id, chunk)
                except Exception as e:
                    print(f"Warning: Failed to generate embedding for chunk {chunk_id}: {e}")
            
            conn.commit()
            conn.close()
            
            # Rebuild search index if this is a new document
            try:
                self.semantic_search.build_faiss_index()
            except Exception as e:
                print(f"Warning: Failed to rebuild search index: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error processing document {document_id}: {str(e)}")
            return False
    
    def search_documents(self, query: str, limit: int = 5, use_semantic: bool = True) -> List[Dict]:
        """Search documents using semantic search or fallback to keyword search"""
        if use_semantic:
            try:
                # Use semantic search for better results
                semantic_results = self.semantic_search.hybrid_search(query, limit)
                
                # Convert to expected format
                results = []
                seen_docs = set()
                
                for result in semantic_results:
                    doc_id = result['document_id']
                    if doc_id not in seen_docs:
                        results.append({
                            'id': doc_id,
                            'filename': result['filename'],
                            'summary': result['summary'],
                            'topics': '',  # Will be filled from database
                            'doc_type': '',  # Will be filled from database
                            'similarity_score': result['similarity_score']
                        })
                        seen_docs.add(doc_id)
                        
                # Fill in missing details
                if results:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    for result in results:
                        cursor.execute('''
                            SELECT topics, doc_type FROM documents WHERE id = ?
                        ''', (result['id'],))
                        row = cursor.fetchone()
                        if row:
                            result['topics'] = row[0] or ''
                            result['doc_type'] = row[1] or ''
                    
                    conn.close()
                
                return results[:limit]
                
            except Exception as e:
                print(f"Semantic search failed, falling back to keyword search: {e}")
        
        # Fallback to original keyword search
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT d.id, d.filename, d.summary, d.topics, d.doc_type
            FROM documents d
            WHERE d.processed = TRUE
            AND (d.summary LIKE ? OR d.topics LIKE ? OR d.filename LIKE ?)
            ORDER BY d.added_date DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'filename': row[1],
                'summary': row[2],
                'topics': row[3],
                'doc_type': row[4],
                'similarity_score': 1.0  # Default score for keyword search
            })
        
        conn.close()
        return results
    
    def get_relevant_chunks(self, document_ids: List[int], query: str, limit: int = 3) -> List[str]:
        """Get relevant text chunks from specific documents using semantic search"""
        if not document_ids:
            return []
        
        try:
            # Use semantic search to find most relevant chunks
            semantic_results = self.semantic_search.semantic_search(query, limit * 3)
            
            # Filter for requested documents and get top chunks
            relevant_chunks = []
            for result in semantic_results:
                if result['document_id'] in document_ids:
                    relevant_chunks.append(result['chunk_text'])
                    if len(relevant_chunks) >= limit:
                        break
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Semantic chunk search failed, falling back to keyword search: {e}")
            
            # Fallback to original keyword search
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in document_ids])
            cursor.execute(f'''
                SELECT chunk_text
                FROM document_chunks
                WHERE document_id IN ({placeholders})
                AND chunk_text LIKE ?
                LIMIT ?
            ''', document_ids + [f'%{query}%', limit])
            
            chunks = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return chunks