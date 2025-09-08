"""
Smart Document Recommendation System for AI Document Library
Provides intelligent document suggestions based on user behavior and content analysis
"""

import sqlite3
import json
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SmartRecommendationEngine:
    def __init__(self, db_path: str, semantic_search_engine=None, relationship_mapper=None):
        """
        Initialize recommendation engine
        
        Args:
            db_path: Path to SQLite database
            semantic_search_engine: Semantic search engine for similarity
            relationship_mapper: Document relationship mapper
        """
        self.db_path = db_path
        self.semantic_search = semantic_search_engine
        self.relationship_mapper = relationship_mapper
        self.logger = logging.getLogger(__name__)
        
        # Initialize recommendation tracking tables
        self.init_recommendation_tables()
        
    def init_recommendation_tables(self):
        """Create tables for tracking recommendations and user behavior"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User interaction tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                interaction_type TEXT,
                interaction_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Recommendation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                recommended_doc_id INTEGER,
                recommendation_type TEXT,
                confidence_score REAL,
                user_action TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id),
                FOREIGN KEY (recommended_doc_id) REFERENCES documents (id)
            )
        ''')
        
        # User preferences learned from behavior
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_type TEXT,
                preference_value TEXT,
                confidence REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Document access patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_access_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                access_count INTEGER DEFAULT 1,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_duration INTEGER,
                access_context TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def track_user_interaction(self, document_id: int, interaction_type: str, 
                             interaction_data: Dict = None, session_id: str = None):
        """Track user interactions with documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_interactions 
            (document_id, interaction_type, interaction_data, session_id)
            VALUES (?, ?, ?, ?)
        ''', (
            document_id, interaction_type, 
            json.dumps(interaction_data) if interaction_data else None,
            session_id
        ))
        
        # Update access patterns
        cursor.execute('''
            INSERT OR REPLACE INTO document_access_patterns 
            (document_id, access_count, last_accessed)
            VALUES (?, 
                    COALESCE((SELECT access_count FROM document_access_patterns WHERE document_id = ?), 0) + 1,
                    CURRENT_TIMESTAMP)
        ''', (document_id, document_id))
        
        conn.commit()
        conn.close()
        
    def get_recommendations(self, context: Dict = None, limit: int = 5) -> List[Dict]:
        """
        Get document recommendations based on various factors
        
        Args:
            context: Current context (current_document, recent_queries, etc.)
            limit: Number of recommendations to return
            
        Returns:
            List of recommended documents with scores and explanations
        """
        recommendations = []
        
        # Get different types of recommendations
        content_based = self.get_content_based_recommendations(context, limit * 2)
        collaborative = self.get_collaborative_recommendations(limit * 2)
        trending = self.get_trending_recommendations(limit)
        gap_analysis = self.get_knowledge_gap_recommendations(context, limit)
        temporal = self.get_temporal_recommendations(limit)
        
        # Combine and rank recommendations
        all_recommendations = {}
        
        # Add content-based recommendations (highest weight)
        for rec in content_based:
            doc_id = rec['document_id']
            all_recommendations[doc_id] = {
                **rec,
                'combined_score': rec['score'] * 0.4,
                'reasons': [rec['reason']]
            }
            
        # Add collaborative recommendations
        for rec in collaborative:
            doc_id = rec['document_id']
            if doc_id in all_recommendations:
                all_recommendations[doc_id]['combined_score'] += rec['score'] * 0.3
                all_recommendations[doc_id]['reasons'].append(rec['reason'])
            else:
                all_recommendations[doc_id] = {
                    **rec,
                    'combined_score': rec['score'] * 0.3,
                    'reasons': [rec['reason']]
                }
                
        # Add trending recommendations
        for rec in trending:
            doc_id = rec['document_id']
            if doc_id in all_recommendations:
                all_recommendations[doc_id]['combined_score'] += rec['score'] * 0.15
                all_recommendations[doc_id]['reasons'].append(rec['reason'])
            else:
                all_recommendations[doc_id] = {
                    **rec,
                    'combined_score': rec['score'] * 0.15,
                    'reasons': [rec['reason']]
                }
                
        # Add gap analysis recommendations
        for rec in gap_analysis:
            doc_id = rec['document_id']
            if doc_id in all_recommendations:
                all_recommendations[doc_id]['combined_score'] += rec['score'] * 0.1
                all_recommendations[doc_id]['reasons'].append(rec['reason'])
            else:
                all_recommendations[doc_id] = {
                    **rec,
                    'combined_score': rec['score'] * 0.1,
                    'reasons': [rec['reason']]
                }
                
        # Add temporal recommendations
        for rec in temporal:
            doc_id = rec['document_id']
            if doc_id in all_recommendations:
                all_recommendations[doc_id]['combined_score'] += rec['score'] * 0.05
                all_recommendations[doc_id]['reasons'].append(rec['reason'])
            else:
                all_recommendations[doc_id] = {
                    **rec,
                    'combined_score': rec['score'] * 0.05,
                    'reasons': [rec['reason']]
                }
                
        # Sort by combined score and return top recommendations
        sorted_recommendations = sorted(
            all_recommendations.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        # Format final recommendations
        final_recommendations = []
        for rec in sorted_recommendations[:limit]:
            # Combine reasons intelligently
            unique_reasons = []
            for reason in rec['reasons']:
                if reason not in unique_reasons:
                    unique_reasons.append(reason)
                    
            final_recommendations.append({
                'document_id': rec['document_id'],
                'filename': rec['filename'],
                'summary': rec.get('summary', ''),
                'score': rec['combined_score'],
                'primary_reason': unique_reasons[0] if unique_reasons else 'Similar content',
                'all_reasons': unique_reasons,
                'recommendation_type': 'combined'
            })
            
        return final_recommendations
        
    def get_content_based_recommendations(self, context: Dict = None, limit: int = 10) -> List[Dict]:
        """Get recommendations based on content similarity"""
        recommendations = []
        
        if not context or not context.get('current_document'):
            # Use recent chat history to infer interests
            recent_queries = self.get_recent_queries(limit=5)
            if recent_queries:
                query_text = ' '.join(recent_queries)
                return self.get_recommendations_for_query(query_text, limit)
            else:
                return self.get_popular_recommendations(limit)
                
        current_doc_id = context['current_document']
        
        if self.semantic_search:
            # Use semantic similarity
            similar_docs = self.semantic_search.find_similar_documents(current_doc_id, limit)
            
            for doc in similar_docs:
                recommendations.append({
                    'document_id': doc['document_id'],
                    'filename': doc['filename'],
                    'summary': doc['summary'],
                    'score': doc['similarity_score'],
                    'reason': f"Similar content ({doc['similarity_score']:.2f} similarity)"
                })
        else:
            # Fallback to topic-based similarity
            recommendations = self.get_topic_based_recommendations(current_doc_id, limit)
            
        return recommendations
        
    def get_collaborative_recommendations(self, limit: int = 10) -> List[Dict]:
        """Get recommendations based on user behavior patterns"""
        recommendations = []
        
        # Find documents frequently accessed together
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get documents accessed in same sessions
        cursor.execute('''
            SELECT ui1.document_id, ui2.document_id, COUNT(*) as co_occurrence
            FROM user_interactions ui1
            JOIN user_interactions ui2 ON ui1.session_id = ui2.session_id
            WHERE ui1.document_id != ui2.document_id
            AND ui1.session_id IS NOT NULL
            GROUP BY ui1.document_id, ui2.document_id
            HAVING co_occurrence > 1
            ORDER BY co_occurrence DESC
            LIMIT ?
        ''', (limit,))
        
        co_accessed = cursor.fetchall()
        
        # Get recently accessed documents to base recommendations on
        cursor.execute('''
            SELECT DISTINCT document_id FROM user_interactions
            WHERE timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp DESC
            LIMIT 5
        ''')
        
        recent_docs = [row[0] for row in cursor.fetchall()]
        
        # Find documents co-accessed with recent ones
        for recent_doc in recent_docs:
            for doc1, doc2, count in co_accessed:
                if doc1 == recent_doc and doc2 not in recent_docs:
                    # Get document details
                    cursor.execute('''
                        SELECT filename, summary FROM documents WHERE id = ?
                    ''', (doc2,))
                    doc_info = cursor.fetchone()
                    
                    if doc_info:
                        recommendations.append({
                            'document_id': doc2,
                            'filename': doc_info[0],
                            'summary': doc_info[1] or '',
                            'score': min(count / 5.0, 1.0),  # Normalize score
                            'reason': f"Frequently accessed with {self.get_document_name(recent_doc)}"
                        })
                        
        conn.close()
        return recommendations[:limit]
        
    def get_trending_recommendations(self, limit: int = 5) -> List[Dict]:
        """Get recommendations based on trending/recently active documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get documents with high recent activity
        cursor.execute('''
            SELECT d.id, d.filename, d.summary, COUNT(*) as recent_activity
            FROM documents d
            JOIN user_interactions ui ON d.id = ui.document_id
            WHERE ui.timestamp >= datetime('now', '-3 days')
            GROUP BY d.id, d.filename, d.summary
            ORDER BY recent_activity DESC
            LIMIT ?
        ''', (limit,))
        
        recommendations = []
        for doc_id, filename, summary, activity_count in cursor.fetchall():
            recommendations.append({
                'document_id': doc_id,
                'filename': filename,
                'summary': summary or '',
                'score': min(activity_count / 10.0, 1.0),  # Normalize
                'reason': f"Trending ({activity_count} recent interactions)"
            })
            
        conn.close()
        return recommendations
        
    def get_knowledge_gap_recommendations(self, context: Dict = None, limit: int = 5) -> List[Dict]:
        """Recommend documents that fill knowledge gaps"""
        recommendations = []
        
        # Analyze user's questions to find gaps
        recent_queries = self.get_recent_queries(limit=10)
        failed_queries = self.get_failed_queries(limit=5)
        
        # Combine all queries to understand information needs
        all_queries = recent_queries + failed_queries
        
        if not all_queries:
            return recommendations
            
        # Find documents that might answer unanswered questions
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for query in failed_queries:
            # Search for documents with keywords from failed queries
            keywords = self.extract_keywords(query)
            
            for keyword in keywords:
                cursor.execute('''
                    SELECT d.id, d.filename, d.summary, d.topics
                    FROM documents d
                    LEFT JOIN document_chunks dc ON d.id = dc.document_id
                    WHERE (d.summary LIKE ? OR d.topics LIKE ? OR dc.chunk_text LIKE ?)
                    AND d.id NOT IN (
                        SELECT DISTINCT document_id FROM user_interactions
                        WHERE timestamp >= datetime('now', '-7 days')
                    )
                    LIMIT 3
                ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))
                
                for doc_id, filename, summary, topics in cursor.fetchall():
                    recommendations.append({
                        'document_id': doc_id,
                        'filename': filename,
                        'summary': summary or '',
                        'score': 0.8,
                        'reason': f"May answer your question about '{keyword}'"
                    })
                    
        conn.close()
        
        # Remove duplicates and return top recommendations
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['document_id'] not in seen:
                seen.add(rec['document_id'])
                unique_recommendations.append(rec)
                
        return unique_recommendations[:limit]
        
    def get_temporal_recommendations(self, limit: int = 5) -> List[Dict]:
        """Get recommendations based on temporal patterns"""
        recommendations = []
        
        # Get documents from same time period in previous weeks/months
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = datetime.now()
        current_hour = current_time.hour
        current_weekday = current_time.weekday()
        
        # Find documents accessed at similar times
        cursor.execute('''
            SELECT d.id, d.filename, d.summary, COUNT(*) as pattern_match
            FROM documents d
            JOIN user_interactions ui ON d.id = ui.document_id
            WHERE cast(strftime('%H', ui.timestamp) as integer) BETWEEN ? AND ?
            AND cast(strftime('%w', ui.timestamp) as integer) = ?
            AND ui.timestamp < datetime('now', '-1 day')
            GROUP BY d.id, d.filename, d.summary
            ORDER BY pattern_match DESC
            LIMIT ?
        ''', (current_hour - 1, current_hour + 1, current_weekday, limit))
        
        for doc_id, filename, summary, pattern_count in cursor.fetchall():
            recommendations.append({
                'document_id': doc_id,
                'filename': filename,
                'summary': summary or '',
                'score': min(pattern_count / 5.0, 0.8),
                'reason': f"You often access this around this time"
            })
            
        conn.close()
        return recommendations
        
    def get_recent_queries(self, limit: int = 10) -> List[str]:
        """Get recent chat queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message FROM chat_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        queries = [row[0] for row in cursor.fetchall()]
        conn.close()
        return queries
        
    def get_failed_queries(self, limit: int = 5) -> List[str]:
        """Get queries that didn't return good results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Look for responses that indicate no good results
        cursor.execute('''
            SELECT user_message FROM chat_history
            WHERE ai_response LIKE '%no relevant documents%'
            OR ai_response LIKE '%couldn''t find%'
            OR ai_response LIKE '%no documents%'
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        failed_queries = [row[0] for row in cursor.fetchall()]
        conn.close()
        return failed_queries
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they',
                     'this', 'have', 'from', 'not', 'but', 'had', 'what', 'can', 'out', 'other',
                     'were', 'all', 'your', 'when', 'time', 'how', 'about', 'could', 'there'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return most common keywords
        return list(Counter(keywords).keys())[:5]
        
    def get_recommendations_for_query(self, query: str, limit: int = 5) -> List[Dict]:
        """Get recommendations based on a specific query"""
        if self.semantic_search:
            # Use semantic search
            results = self.semantic_search.semantic_search(query, limit)
            
            recommendations = []
            for result in results:
                recommendations.append({
                    'document_id': result['document_id'],
                    'filename': result['filename'],
                    'summary': result['summary'],
                    'score': result['similarity_score'],
                    'reason': f"Relevant to your interests ({result['similarity_score']:.2f} match)"
                })
            return recommendations
        else:
            return self.get_popular_recommendations(limit)
            
    def get_popular_recommendations(self, limit: int = 5) -> List[Dict]:
        """Get popular/frequently accessed documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT d.id, d.filename, d.summary, dap.access_count
            FROM documents d
            LEFT JOIN document_access_patterns dap ON d.id = dap.document_id
            ORDER BY COALESCE(dap.access_count, 0) DESC
            LIMIT ?
        ''', (limit,))
        
        recommendations = []
        for doc_id, filename, summary, access_count in cursor.fetchall():
            recommendations.append({
                'document_id': doc_id,
                'filename': filename,
                'summary': summary or '',
                'score': min((access_count or 1) / 10.0, 1.0),
                'reason': f"Popular document ({access_count or 1} accesses)"
            })
            
        conn.close()
        return recommendations
        
    def get_topic_based_recommendations(self, document_id: int, limit: int = 5) -> List[Dict]:
        """Get recommendations based on topic similarity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get topics for the source document
        cursor.execute('SELECT topics FROM documents WHERE id = ?', (document_id,))
        source_topics = cursor.fetchone()
        
        if not source_topics or not source_topics[0]:
            return []
            
        source_topic_list = [t.strip().lower() for t in source_topics[0].split(',') if t.strip()]
        
        recommendations = []
        
        # Find documents with overlapping topics
        cursor.execute('''
            SELECT id, filename, summary, topics
            FROM documents
            WHERE id != ? AND topics IS NOT NULL
        ''', (document_id,))
        
        for doc_id, filename, summary, topics in cursor.fetchall():
            if not topics:
                continue
                
            doc_topics = [t.strip().lower() for t in topics.split(',') if t.strip()]
            common_topics = set(source_topic_list).intersection(set(doc_topics))
            
            if common_topics:
                similarity = len(common_topics) / len(set(source_topic_list).union(set(doc_topics)))
                
                recommendations.append({
                    'document_id': doc_id,
                    'filename': filename,
                    'summary': summary or '',
                    'score': similarity,
                    'reason': f"Shares topics: {', '.join(list(common_topics)[:3])}"
                })
                
        conn.close()
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:limit]
        
    def get_document_name(self, document_id: int) -> str:
        """Get document name by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT filename FROM documents WHERE id = ?', (document_id,))
        result = cursor.fetchone()
        
        conn.close()
        return result[0] if result else f"Document {document_id}"
        
    def record_recommendation_feedback(self, document_id: int, recommended_doc_id: int, 
                                     action: str, recommendation_type: str):
        """Record user action on recommendation for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recommendation_history 
            (document_id, recommended_doc_id, recommendation_type, user_action)
            VALUES (?, ?, ?, ?)
        ''', (document_id, recommended_doc_id, recommendation_type, action))
        
        conn.commit()
        conn.close()
        
        # Update user preferences based on feedback
        self.update_user_preferences(action, recommendation_type)
        
    def update_user_preferences(self, action: str, recommendation_type: str):
        """Update learned user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Positive feedback increases preference confidence
        confidence_change = 0.1 if action in ['clicked', 'viewed', 'liked'] else -0.05
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (preference_type, preference_value, confidence, last_updated)
            VALUES (?, ?, 
                    COALESCE((SELECT confidence FROM user_preferences 
                             WHERE preference_type = ? AND preference_value = ?), 0.5) + ?,
                    CURRENT_TIMESTAMP)
        ''', (
            'recommendation_type', recommendation_type, 
            'recommendation_type', recommendation_type,
            confidence_change
        ))
        
        conn.commit()
        conn.close()
        
    def get_recommendation_analytics(self) -> Dict:
        """Get analytics on recommendation performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recommendation click-through rates
        cursor.execute('''
            SELECT recommendation_type, 
                   COUNT(*) as total_recommendations,
                   SUM(CASE WHEN user_action IN ('clicked', 'viewed') THEN 1 ELSE 0 END) as positive_actions
            FROM recommendation_history
            GROUP BY recommendation_type
        ''')
        
        performance = {}
        for rec_type, total, positive in cursor.fetchall():
            ctr = (positive / total) if total > 0 else 0
            performance[rec_type] = {
                'total_recommendations': total,
                'positive_actions': positive,
                'click_through_rate': ctr
            }
            
        # Get most effective recommendation types
        cursor.execute('''
            SELECT preference_value, confidence 
            FROM user_preferences
            WHERE preference_type = 'recommendation_type'
            ORDER BY confidence DESC
        ''')
        
        preferences = cursor.fetchall()
        
        conn.close()
        
        return {
            'performance_by_type': performance,
            'user_preferences': preferences
        }