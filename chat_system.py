"""
Chat system for AI Document Library
"""

import requests
import json
import sqlite3
from typing import List, Dict, Optional
from document_processor import DocumentProcessor

class ChatSystem:
    def __init__(self, db_path: str, config: dict):
        self.db_path = db_path
        self.config = config
        self.doc_processor = DocumentProcessor(db_path, config)
    
    def process_message(self, user_message: str) -> str:
        """Process user message and generate AI response"""
        try:
            # Search for relevant documents
            relevant_docs = self.doc_processor.search_documents(user_message, limit=3)
            
            if relevant_docs:
                # Get relevant chunks from found documents
                doc_ids = [doc['id'] for doc in relevant_docs]
                relevant_chunks = self.doc_processor.get_relevant_chunks(doc_ids, user_message, limit=3)
                
                # Build context for AI
                context = self.build_context(user_message, relevant_docs, relevant_chunks)
                
                # Generate AI response
                if self.config['ai_type'] == 'local':
                    response = self.generate_ollama_response(context)
                else:
                    response = self.generate_api_response(context)
                
                # Store chat history
                self.store_chat_history(user_message, response, relevant_docs)
                
                return response
            else:
                # No relevant documents found
                return self.generate_no_docs_response(user_message)
                
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def build_context(self, user_message: str, relevant_docs: List[Dict], 
                     relevant_chunks: List[str]) -> str:
        """Build context for AI response including relevant document information"""
        context = f"User question: {user_message}\n\n"
        
        if relevant_docs:
            context += "Relevant documents found:\n"
            for doc in relevant_docs:
                context += f"- {doc['filename']}: {doc['summary']}\n"
                context += f"  Topics: {doc['topics']}\n\n"
        
        if relevant_chunks:
            context += "Relevant content excerpts:\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                # Truncate long chunks
                display_chunk = chunk[:500] + "..." if len(chunk) > 500 else chunk
                context += f"{i}. {display_chunk}\n\n"
        
        return context
    
    def generate_ollama_response(self, context: str) -> str:
        """Generate response using local Ollama model"""
        prompt = f"""You are an AI assistant helping users understand their document collection. 
Based on the context below, provide a helpful and accurate response to the user's question.

If relevant documents are found, use the information from them to answer the question.
If content excerpts are provided, reference them in your response.
Be conversational and helpful.

{context}

Please provide a helpful response:"""
        
        try:
            url = f"{self.config['ollama_url']}/api/generate"
            payload = {
                "model": self.config.get('local_model', 'gemma2:2b'),
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', 'Sorry, I could not generate a response.')
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_api_response(self, context: str) -> str:
        """Generate response using cloud API"""
        # Placeholder for API implementation
        return f"API response would be generated here based on: {context[:100]}..."
    
    def generate_no_docs_response(self, user_message: str) -> str:
        """Generate response when no relevant documents are found"""
        if self.config['ai_type'] == 'local':
            prompt = f"""The user asked: "{user_message}"

No relevant documents were found in their library. Provide a helpful response explaining this and suggest they might need to add relevant documents to their library or rephrase their question.

Keep it conversational and helpful."""
            
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
                return result.get('response', 
                    "I couldn't find any relevant documents in your library for that question. Try adding more documents or rephrasing your question.")
                
            except Exception as e:
                return "I couldn't find any relevant documents in your library for that question. Try adding more documents or rephrasing your question."
        else:
            return "I couldn't find any relevant documents in your library for that question. Try adding more documents or rephrasing your question."
    
    def store_chat_history(self, user_message: str, ai_response: str, 
                          relevant_docs: List[Dict]) -> None:
        """Store chat exchange in database"""
        try:
            doc_refs = json.dumps([doc['filename'] for doc in relevant_docs]) if relevant_docs else None
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history (user_message, ai_response, relevant_docs)
                VALUES (?, ?, ?)
            ''', (user_message, ai_response, doc_refs))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing chat history: {str(e)}")
    
    def get_chat_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent chat history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_message, ai_response, timestamp, relevant_docs
                FROM chat_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'user_message': row[0],
                    'ai_response': row[1],
                    'timestamp': row[2],
                    'relevant_docs': json.loads(row[3]) if row[3] else []
                })
            
            conn.close()
            return history
            
        except Exception as e:
            print(f"Error retrieving chat history: {str(e)}")
            return []