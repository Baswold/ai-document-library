"""
Document Relationship Mapper for AI Document Library
Analyzes and visualizes connections between documents
"""

import sqlite3
import json
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import logging
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentRelationshipMapper:
    def __init__(self, db_path: str, semantic_search_engine=None):
        """
        Initialize relationship mapper
        
        Args:
            db_path: Path to SQLite database
            semantic_search_engine: Optional semantic search engine for similarity
        """
        self.db_path = db_path
        self.semantic_search = semantic_search_engine
        self.logger = logging.getLogger(__name__)
        self.graph = nx.Graph()
        
        # Initialize relationship tracking tables
        self.init_relationship_tables()
        
    def init_relationship_tables(self):
        """Create tables for storing document relationships"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Document relationships table (already created in semantic_search.py)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc1_id INTEGER,
                doc2_id INTEGER,
                relationship_type TEXT,
                similarity_score REAL,
                shared_entities TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc1_id) REFERENCES documents (id),
                FOREIGN KEY (doc2_id) REFERENCES documents (id)
            )
        ''')
        
        # Document entities table (people, organizations, concepts)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                entity_text TEXT,
                entity_type TEXT,
                confidence REAL DEFAULT 1.0,
                positions TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        ''')
        
        # Document citations/references table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_doc_id INTEGER,
                target_doc_id INTEGER,
                citation_text TEXT,
                citation_type TEXT,
                FOREIGN KEY (source_doc_id) REFERENCES documents (id),
                FOREIGN KEY (target_doc_id) REFERENCES documents (id)
            )
        ''')
        
        # Document clusters table for grouping related documents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_name TEXT,
                cluster_description TEXT,
                document_ids TEXT,
                cluster_score REAL,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def analyze_all_relationships(self, force_rebuild: bool = False):
        """Analyze relationships between all documents"""
        if force_rebuild:
            self.clear_existing_relationships()
            
        self.logger.info("Starting comprehensive document relationship analysis...")
        
        # Get all processed documents
        documents = self.get_all_documents()
        total_docs = len(documents)
        
        if total_docs < 2:
            self.logger.info("Need at least 2 documents for relationship analysis")
            return
            
        self.logger.info(f"Analyzing relationships for {total_docs} documents")
        
        # Extract entities from all documents
        self.extract_document_entities(documents)
        
        # Analyze different types of relationships
        relationships_found = 0
        
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                # Skip self-comparison
                if doc1['id'] == doc2['id']:
                    continue
                    
                # Check for existing relationship
                if self.relationship_exists(doc1['id'], doc2['id']):
                    continue
                    
                # Analyze semantic similarity
                semantic_rel = self.analyze_semantic_similarity(doc1, doc2)
                if semantic_rel:
                    self.store_relationship(doc1['id'], doc2['id'], semantic_rel)
                    relationships_found += 1
                    
                # Analyze entity overlap
                entity_rel = self.analyze_entity_overlap(doc1['id'], doc2['id'])
                if entity_rel:
                    self.store_relationship(doc1['id'], doc2['id'], entity_rel)
                    relationships_found += 1
                    
                # Analyze temporal relationships
                temporal_rel = self.analyze_temporal_relationship(doc1, doc2)
                if temporal_rel:
                    self.store_relationship(doc1['id'], doc2['id'], temporal_rel)
                    relationships_found += 1
                    
                # Analyze topic similarity
                topic_rel = self.analyze_topic_similarity(doc1, doc2)
                if topic_rel:
                    self.store_relationship(doc1['id'], doc2['id'], topic_rel)
                    relationships_found += 1
                    
            # Progress logging
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{total_docs} documents")
                
        # Create document clusters
        self.create_document_clusters()
        
        self.logger.info(f"Analysis complete. Found {relationships_found} relationships")
        
    def get_all_documents(self) -> List[Dict]:
        """Get all processed documents with their content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT d.id, d.filename, d.summary, d.topics, d.doc_type, d.added_date,
                   GROUP_CONCAT(dc.chunk_text, ' ') as full_text
            FROM documents d
            LEFT JOIN document_chunks dc ON d.id = dc.document_id
            WHERE d.processed = TRUE
            GROUP BY d.id, d.filename, d.summary, d.topics, d.doc_type, d.added_date
        ''')
        
        documents = []
        for row in cursor.fetchall():
            documents.append({
                'id': row[0],
                'filename': row[1],
                'summary': row[2] or '',
                'topics': row[3] or '',
                'doc_type': row[4] or '',
                'added_date': row[5],
                'full_text': row[6] or ''
            })
            
        conn.close()
        return documents
        
    def extract_document_entities(self, documents: List[Dict]):
        """Extract entities (people, organizations, concepts) from documents"""
        for doc in documents:
            # Check if entities already extracted
            if self.entities_exist(doc['id']):
                continue
                
            entities = self.extract_entities_from_text(doc['full_text'], doc['summary'])
            self.store_document_entities(doc['id'], entities)
            
    def extract_entities_from_text(self, text: str, summary: str) -> List[Dict]:
        """Extract entities from document text using pattern matching"""
        entities = []
        
        # Combine text and summary for analysis
        full_content = f"{summary} {text}"
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, full_content)
        for email in set(emails):
            entities.append({
                'text': email,
                'type': 'email',
                'confidence': 0.9,
                'positions': [m.start() for m in re.finditer(re.escape(email), full_content)]
            })
            
        # Extract potential names (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        names = re.findall(name_pattern, full_content)
        for name in set(names):
            # Filter out common false positives
            if not any(word in name.lower() for word in ['the', 'and', 'this', 'that', 'with']):
                entities.append({
                    'text': name,
                    'type': 'person',
                    'confidence': 0.7,
                    'positions': [m.start() for m in re.finditer(re.escape(name), full_content)]
                })
                
        # Extract organizations (words with Inc, Corp, LLC, etc.)
        org_pattern = r'\b[A-Z][A-Za-z\s&]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|Association|Foundation)\b'
        orgs = re.findall(org_pattern, full_content)
        for org in set(orgs):
            entities.append({
                'text': org,
                'type': 'organization',
                'confidence': 0.8,
                'positions': [m.start() for m in re.finditer(re.escape(org), full_content)]
            })
            
        # Extract dates
        date_pattern = r'\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2}, \d{4})\b'
        dates = re.findall(date_pattern, full_content)
        for date in set(dates):
            entities.append({
                'text': date,
                'type': 'date',
                'confidence': 0.9,
                'positions': [m.start() for m in re.finditer(re.escape(date), full_content)]
            })
            
        # Extract phone numbers
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        phones = re.findall(phone_pattern, full_content)
        for phone in set(phones):
            entities.append({
                'text': phone,
                'type': 'phone',
                'confidence': 0.9,
                'positions': [m.start() for m in re.finditer(re.escape(phone), full_content)]
            })
            
        return entities
        
    def entities_exist(self, document_id: int) -> bool:
        """Check if entities already extracted for document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM document_entities WHERE document_id = ?', (document_id,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count > 0
        
    def store_document_entities(self, document_id: int, entities: List[Dict]):
        """Store extracted entities in database"""
        if not entities:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for entity in entities:
            cursor.execute('''
                INSERT INTO document_entities 
                (document_id, entity_text, entity_type, confidence, positions)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                document_id, entity['text'], entity['type'], 
                entity['confidence'], json.dumps(entity['positions'])
            ))
            
        conn.commit()
        conn.close()
        
    def analyze_semantic_similarity(self, doc1: Dict, doc2: Dict) -> Optional[Dict]:
        """Analyze semantic similarity between two documents"""
        if not self.semantic_search:
            return None
            
        try:
            # Use semantic search to find similarity
            similar_docs = self.semantic_search.find_similar_documents(doc1['id'], top_k=10)
            
            for similar_doc in similar_docs:
                if similar_doc['document_id'] == doc2['id']:
                    if similar_doc['similarity_score'] > 0.7:  # High similarity threshold
                        return {
                            'type': 'semantic_similarity',
                            'score': similar_doc['similarity_score'],
                            'description': f"High semantic similarity ({similar_doc['similarity_score']:.2f})"
                        }
                    elif similar_doc['similarity_score'] > 0.5:  # Medium similarity threshold
                        return {
                            'type': 'content_similarity', 
                            'score': similar_doc['similarity_score'],
                            'description': f"Content similarity ({similar_doc['similarity_score']:.2f})"
                        }
                        
        except Exception as e:
            self.logger.error(f"Error in semantic similarity analysis: {e}")
            
        return None
        
    def analyze_entity_overlap(self, doc1_id: int, doc2_id: int) -> Optional[Dict]:
        """Analyze entity overlap between two documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get entities for both documents
        cursor.execute('''
            SELECT entity_text, entity_type FROM document_entities WHERE document_id = ?
        ''', (doc1_id,))
        entities1 = set((row[0].lower(), row[1]) for row in cursor.fetchall())
        
        cursor.execute('''
            SELECT entity_text, entity_type FROM document_entities WHERE document_id = ?
        ''', (doc2_id,))
        entities2 = set((row[0].lower(), row[1]) for row in cursor.fetchall())
        
        conn.close()
        
        if not entities1 or not entities2:
            return None
            
        # Calculate overlap
        common_entities = entities1.intersection(entities2)
        
        if len(common_entities) > 0:
            overlap_ratio = len(common_entities) / min(len(entities1), len(entities2))
            
            if overlap_ratio > 0.3:  # Significant overlap
                return {
                    'type': 'entity_overlap',
                    'score': overlap_ratio,
                    'shared_entities': list(common_entities),
                    'description': f"Shared entities: {', '.join([e[0] for e in list(common_entities)[:3]])}"
                }
                
        return None
        
    def analyze_temporal_relationship(self, doc1: Dict, doc2: Dict) -> Optional[Dict]:
        """Analyze temporal relationships between documents"""
        try:
            date1 = datetime.fromisoformat(doc1['added_date'].replace('Z', '+00:00'))
            date2 = datetime.fromisoformat(doc2['added_date'].replace('Z', '+00:00'))
            
            time_diff = abs((date1 - date2).days)
            
            if time_diff <= 1:  # Same day or consecutive days
                return {
                    'type': 'temporal_close',
                    'score': 1.0 - (time_diff / 7),  # Higher score for closer dates
                    'description': f"Created within {time_diff} day(s)"
                }
            elif time_diff <= 7:  # Same week
                return {
                    'type': 'temporal_week',
                    'score': 1.0 - (time_diff / 30),
                    'description': f"Created within same week ({time_diff} days apart)"
                }
                
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {e}")
            
        return None
        
    def analyze_topic_similarity(self, doc1: Dict, doc2: Dict) -> Optional[Dict]:
        """Analyze topic similarity between documents"""
        topics1 = set(topic.strip().lower() for topic in doc1['topics'].split(',') if topic.strip())
        topics2 = set(topic.strip().lower() for topic in doc2['topics'].split(',') if topic.strip())
        
        if not topics1 or not topics2:
            return None
            
        common_topics = topics1.intersection(topics2)
        
        if len(common_topics) > 0:
            similarity = len(common_topics) / len(topics1.union(topics2))
            
            if similarity > 0.3:  # Significant topic overlap
                return {
                    'type': 'topic_similarity',
                    'score': similarity,
                    'shared_topics': list(common_topics),
                    'description': f"Shared topics: {', '.join(list(common_topics)[:3])}"
                }
                
        return None
        
    def relationship_exists(self, doc1_id: int, doc2_id: int) -> bool:
        """Check if relationship already exists between two documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM document_relationships 
            WHERE (doc1_id = ? AND doc2_id = ?) OR (doc1_id = ? AND doc2_id = ?)
        ''', (doc1_id, doc2_id, doc2_id, doc1_id))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
        
    def store_relationship(self, doc1_id: int, doc2_id: int, relationship: Dict):
        """Store relationship between two documents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        shared_entities = json.dumps(relationship.get('shared_entities', []))
        
        cursor.execute('''
            INSERT INTO document_relationships 
            (doc1_id, doc2_id, relationship_type, similarity_score, shared_entities)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            doc1_id, doc2_id, relationship['type'], 
            relationship['score'], shared_entities
        ))
        
        conn.commit()
        conn.close()
        
    def clear_existing_relationships(self):
        """Clear existing relationship data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM document_relationships')
        cursor.execute('DELETE FROM document_entities')
        cursor.execute('DELETE FROM document_clusters')
        
        conn.commit()
        conn.close()
        
    def create_document_clusters(self):
        """Create clusters of related documents"""
        # Get all relationships
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT doc1_id, doc2_id, similarity_score, relationship_type
            FROM document_relationships
            WHERE similarity_score > 0.5
        ''')
        
        relationships = cursor.fetchall()
        
        # Build graph for clustering
        G = nx.Graph()
        for doc1_id, doc2_id, score, rel_type in relationships:
            G.add_edge(doc1_id, doc2_id, weight=score, type=rel_type)
            
        # Find communities/clusters
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(G))
            
            # Store clusters
            for i, community in enumerate(communities):
                if len(community) >= 2:  # Only store clusters with multiple documents
                    # Get cluster description
                    doc_ids = list(community)
                    cluster_description = self.generate_cluster_description(doc_ids)
                    
                    cursor.execute('''
                        INSERT INTO document_clusters 
                        (cluster_name, cluster_description, document_ids, cluster_score)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        f"Cluster {i+1}", cluster_description,
                        json.dumps(doc_ids), len(community)
                    ))
                    
        except ImportError:
            self.logger.warning("NetworkX community detection not available")
            
        conn.commit()
        conn.close()
        
    def generate_cluster_description(self, doc_ids: List[int]) -> str:
        """Generate description for a document cluster"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get topics and types for cluster documents
        placeholders = ','.join(['?' for _ in doc_ids])
        cursor.execute(f'''
            SELECT topics, doc_type FROM documents 
            WHERE id IN ({placeholders})
        ''', doc_ids)
        
        all_topics = []
        doc_types = []
        
        for topics, doc_type in cursor.fetchall():
            if topics:
                all_topics.extend([t.strip() for t in topics.split(',') if t.strip()])
            if doc_type:
                doc_types.append(doc_type)
                
        conn.close()
        
        # Find most common topics and types
        topic_counts = Counter(all_topics)
        type_counts = Counter(doc_types)
        
        top_topics = [topic for topic, _ in topic_counts.most_common(3)]
        top_type = type_counts.most_common(1)[0][0] if type_counts else "documents"
        
        description = f"{len(doc_ids)} {top_type}"
        if top_topics:
            description += f" related to: {', '.join(top_topics)}"
            
        return description
        
    def visualize_relationships(self, output_path: str = None, interactive: bool = True):
        """Create visualization of document relationships"""
        # Get relationship data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT dr.doc1_id, dr.doc2_id, dr.similarity_score, dr.relationship_type,
                   d1.filename as doc1_name, d2.filename as doc2_name
            FROM document_relationships dr
            JOIN documents d1 ON dr.doc1_id = d1.id
            JOIN documents d2 ON dr.doc2_id = d2.id
            WHERE dr.similarity_score > 0.4
        ''')
        
        relationships = cursor.fetchall()
        conn.close()
        
        if not relationships:
            self.logger.info("No relationships found for visualization")
            return
            
        # Build graph
        G = nx.Graph()
        
        for doc1_id, doc2_id, score, rel_type, doc1_name, doc2_name in relationships:
            G.add_node(doc1_id, label=doc1_name[:20] + "..." if len(doc1_name) > 20 else doc1_name)
            G.add_node(doc2_id, label=doc2_name[:20] + "..." if len(doc2_name) > 20 else doc2_name)
            G.add_edge(doc1_id, doc2_id, weight=score, type=rel_type)
            
        if interactive:
            self.create_interactive_visualization(G, output_path)
        else:
            self.create_static_visualization(G, output_path)
            
    def create_interactive_visualization(self, G: nx.Graph, output_path: str = None):
        """Create interactive Plotly visualization"""
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"Relationship: {edge[2]['type']}<br>Score: {edge[2]['weight']:.2f}")
            
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract nodes
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[1]['label'])
            
            # Count connections
            connections = len(list(G.neighbors(node[0])))
            node_info.append(f"Document: {node[1]['label']}<br>Connections: {connections}")
            
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[len(list(G.neighbors(node))) for node in G.nodes()],
                size=10,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    xanchor="left",
                    title="Node Connections"
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Document Relationship Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Document relationships based on content similarity, shared entities, and temporal proximity",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        if output_path:
            fig.write_html(output_path)
            self.logger.info(f"Interactive visualization saved to {output_path}")
        else:
            fig.show()
            
    def create_static_visualization(self, G: nx.Graph, output_path: str = None):
        """Create static matplotlib visualization"""
        plt.figure(figsize=(12, 8))
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw edges with varying thickness based on similarity
        edges = G.edges(data=True)
        weights = [edge[2]['weight'] * 5 for edge in edges]  # Scale for visibility
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray')
        
        # Draw nodes with size based on number of connections
        node_sizes = [len(list(G.neighbors(node))) * 100 + 200 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
        
        # Draw labels
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Document Relationship Network", size=16)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Static visualization saved to {output_path}")
        else:
            plt.show()
            
    def get_relationship_summary(self) -> Dict:
        """Get summary of document relationships"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count relationships by type
        cursor.execute('''
            SELECT relationship_type, COUNT(*), AVG(similarity_score)
            FROM document_relationships
            GROUP BY relationship_type
        ''')
        
        relationship_stats = {}
        for rel_type, count, avg_score in cursor.fetchall():
            relationship_stats[rel_type] = {
                'count': count,
                'avg_score': avg_score
            }
            
        # Get cluster information
        cursor.execute('SELECT COUNT(*) FROM document_clusters')
        cluster_count = cursor.fetchone()[0]
        
        # Get most connected documents
        cursor.execute('''
            SELECT d.filename, COUNT(*) as connection_count
            FROM document_relationships dr
            JOIN documents d ON (d.id = dr.doc1_id OR d.id = dr.doc2_id)
            GROUP BY d.id, d.filename
            ORDER BY connection_count DESC
            LIMIT 5
        ''')
        
        top_connected = cursor.fetchall()
        
        conn.close()
        
        return {
            'relationship_types': relationship_stats,
            'cluster_count': cluster_count,
            'most_connected_documents': top_connected
        }