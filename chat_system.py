"""
Chat system for AI Document Library with file system search capabilities
"""

import requests
import json
import sqlite3
import re
from typing import List, Dict, Optional
from document_processor import DocumentProcessor
from file_system_cataloger import FileSystemCatalog

class ChatSystem:
    def __init__(self, db_path: str, config: dict):
        self.db_path = db_path
        self.config = config
        self.doc_processor = DocumentProcessor(db_path, config)
        self.file_catalog = FileSystemCatalog(db_path)
    
    def detect_search_intent(self, message: str) -> Optional[Dict]:
        """Detect if user wants to search files with glob or grep patterns"""
        message_lower = message.lower()

        # Detect glob search patterns
        glob_patterns = [
            r'find.*\*\.[\w]+',  # "find *.py"
            r'search.*\*\.[\w]+',  # "search *.txt"
            r'show.*\*\.[\w]+',  # "show *.md"
            r'list.*\*\.[\w]+',  # "list *.pdf"
        ]

        for pattern in glob_patterns:
            if re.search(pattern, message_lower):
                # Extract the glob pattern
                match = re.search(r'\*\.[\w]+', message)
                if match:
                    return {'type': 'glob', 'pattern': match.group()}

        # Detect grep/content search
        grep_keywords = [
            'grep', 'search for', 'find files containing', 'files with',
            'search content', 'find text', 'where does', 'which files have'
        ]

        for keyword in grep_keywords:
            if keyword in message_lower:
                # Try to extract the search term
                # Look for quoted text or key phrases
                quote_match = re.search(r'["\']([^"\']+)["\']', message)
                if quote_match:
                    return {'type': 'grep', 'pattern': quote_match.group(1)}

                # Otherwise use the whole message as query
                return {'type': 'grep', 'pattern': message}

        return None

    def process_message(self, user_message: str) -> str:
        """Process user message and generate AI response"""
        try:
            # Check if this is a file system search query
            search_intent = self.detect_search_intent(user_message)

            if search_intent:
                if search_intent['type'] == 'glob':
                    return self.handle_glob_search(search_intent['pattern'])
                elif search_intent['type'] == 'grep':
                    return self.handle_grep_search(search_intent['pattern'], user_message)

            # Regular document search
            relevant_docs = self.doc_processor.search_documents(user_message, limit=3)

            # Also search file catalog
            file_results = self.file_catalog.combined_search(user_message, limit=5)

            if relevant_docs or file_results:
                # Get relevant chunks from found documents
                doc_ids = [doc['id'] for doc in relevant_docs]
                relevant_chunks = self.doc_processor.get_relevant_chunks(doc_ids, user_message, limit=3)

                # Build context for AI
                context = self.build_context(user_message, relevant_docs, relevant_chunks, file_results)

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

    def handle_glob_search(self, pattern: str) -> str:
        """Handle glob pattern file search"""
        results = self.file_catalog.glob_search(pattern, limit=50)

        if not results:
            return f"No files found matching pattern: {pattern}"

        response = f"Found {len(results)} files matching '{pattern}':\n\n"
        for i, result in enumerate(results[:20], 1):  # Show first 20
            size_kb = result['file_size'] / 1024
            response += f"{i}. {result['filename']}\n"
            response += f"   Path: {result['filepath']}\n"
            response += f"   Size: {size_kb:.1f} KB\n\n"

        if len(results) > 20:
            response += f"\n... and {len(results) - 20} more files."

        return response

    def handle_grep_search(self, pattern: str, original_message: str) -> str:
        """Handle grep/content search"""
        # Extract just the search term
        search_term = pattern

        # Remove common prefixes from the search term
        for prefix in ['search for', 'find files containing', 'files with', 'grep', 'where does']:
            search_term = search_term.replace(prefix, '').strip()

        search_term = search_term.strip('"\'')

        if len(search_term) < 2:
            return "Please provide a search term with at least 2 characters."

        results = self.file_catalog.grep_search(search_term, limit=30, regex=False)

        if not results:
            return f"No files found containing: {search_term}"

        response = f"Found '{search_term}' in {len(results)} files:\n\n"
        for i, result in enumerate(results[:15], 1):  # Show first 15
            response += f"{i}. {result['filename']} ({result['match_count']} matches)\n"
            response += f"   Path: {result['filepath']}\n"

            # Show first match context
            if result['matches']:
                first_match = result['matches'][0]
                response += f"   Line {first_match['line_number']}: {first_match['line_content'][:100]}\n\n"

        if len(results) > 15:
            response += f"\n... and {len(results) - 15} more files."

        return response
    
    def build_context(self, user_message: str, relevant_docs: List[Dict],
                     relevant_chunks: List[str], file_results: List[Dict] = None) -> str:
        """Build context for AI response including relevant document information and file system results"""
        context = f"User question: {user_message}\n\n"

        if relevant_docs:
            context += "Relevant documents from library:\n"
            for doc in relevant_docs:
                context += f"- {doc['filename']}: {doc['summary']}\n"
                context += f"  Topics: {doc['topics']}\n\n"

        if file_results:
            context += "Related files from file system:\n"
            for file_info in file_results[:5]:
                context += f"- {file_info['filename']}\n"
                context += f"  Path: {file_info['filepath']}\n"
                if file_info.get('matched_in_content') and file_info.get('content_snippet'):
                    context += f"  Content: ...{file_info['content_snippet']}...\n"
                context += "\n"

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