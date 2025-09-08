"""
Email Importer for AI Document Library
Supports Gmail (.mbox), Outlook (.pst), and standard email formats
"""

import mailbox
import email
import sqlite3
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import logging
from email.header import decode_header
from email.utils import parsedate_to_datetime
import hashlib

class EmailImporter:
    def __init__(self, db_path: str):
        """
        Initialize email importer
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_email_tables()
        
    def init_email_tables(self):
        """Create tables for storing email data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main emails table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                subject TEXT,
                sender_name TEXT,
                sender_email TEXT,
                recipient_emails TEXT,
                date_sent DATETIME,
                body_text TEXT,
                body_html TEXT,
                has_attachments BOOLEAN DEFAULT FALSE,
                thread_id TEXT,
                folder TEXT,
                email_hash TEXT UNIQUE,
                imported_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Email attachments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id INTEGER,
                filename TEXT,
                content_type TEXT,
                size_bytes INTEGER,
                attachment_data BLOB,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
        ''')
        
        # Email threads table for conversation tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT UNIQUE,
                subject TEXT,
                participants TEXT,
                message_count INTEGER DEFAULT 1,
                first_message_date DATETIME,
                last_message_date DATETIME
            )
        ''')
        
        # Email labels/tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id INTEGER,
                label TEXT,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def import_mbox_file(self, mbox_path: str, folder_name: str = "Imported") -> Dict[str, int]:
        """
        Import emails from .mbox file (Gmail Takeout format)
        
        Args:
            mbox_path: Path to .mbox file
            folder_name: Name to assign to imported emails
            
        Returns:
            Dictionary with import statistics
        """
        stats = {
            'total_emails': 0,
            'imported_emails': 0,
            'skipped_emails': 0,
            'errors': 0
        }
        
        try:
            mbox = mailbox.mbox(mbox_path)
            stats['total_emails'] = len(mbox)
            
            self.logger.info(f"Importing {stats['total_emails']} emails from {mbox_path}")
            
            for i, message in enumerate(mbox):
                try:
                    if self.import_email_message(message, folder_name):
                        stats['imported_emails'] += 1
                    else:
                        stats['skipped_emails'] += 1
                        
                    # Progress logging
                    if (i + 1) % 100 == 0:
                        self.logger.info(f"Processed {i + 1}/{stats['total_emails']} emails")
                        
                except Exception as e:
                    self.logger.error(f"Error processing email {i}: {e}")
                    stats['errors'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error opening mbox file {mbox_path}: {e}")
            stats['errors'] += 1
            
        return stats
        
    def import_email_message(self, message: email.message.Message, folder: str) -> bool:
        """
        Import a single email message
        
        Args:
            message: Email message object
            folder: Folder name for organization
            
        Returns:
            True if imported, False if skipped (duplicate)
        """
        try:
            # Extract basic email data
            email_data = self.extract_email_data(message, folder)
            
            # Check if email already exists
            if self.email_exists(email_data['email_hash']):
                return False
                
            # Store email in database
            email_id = self.store_email(email_data)
            
            if not email_id:
                return False
                
            # Process attachments
            if email_data['has_attachments']:
                self.process_attachments(message, email_id)
                
            # Update thread information
            self.update_thread_info(email_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing email: {e}")
            return False
            
    def extract_email_data(self, message: email.message.Message, folder: str) -> Dict:
        """Extract structured data from email message"""
        
        # Get subject
        subject = self.decode_header_value(message.get('Subject', ''))
        
        # Get sender information
        sender_raw = message.get('From', '')
        sender_name, sender_email = self.parse_email_address(sender_raw)
        
        # Get recipients
        recipients = []
        for header in ['To', 'Cc', 'Bcc']:
            if message.get(header):
                recipients.extend(self.parse_email_addresses(message.get(header)))
        
        # Get date
        date_sent = None
        date_header = message.get('Date')
        if date_header:
            try:
                date_sent = parsedate_to_datetime(date_header)
            except:
                pass
                
        # Get message ID and generate thread ID
        message_id = message.get('Message-ID', '')
        thread_id = self.generate_thread_id(message)
        
        # Extract body content
        body_text, body_html = self.extract_body_content(message)
        
        # Check for attachments
        has_attachments = self.has_attachments(message)
        
        # Generate hash for duplicate detection
        email_content = f"{message_id}{subject}{sender_email}{date_sent}"
        email_hash = hashlib.md5(email_content.encode()).hexdigest()
        
        return {
            'message_id': message_id,
            'subject': subject,
            'sender_name': sender_name,
            'sender_email': sender_email,
            'recipient_emails': json.dumps(recipients),
            'date_sent': date_sent,
            'body_text': body_text,
            'body_html': body_html,
            'has_attachments': has_attachments,
            'thread_id': thread_id,
            'folder': folder,
            'email_hash': email_hash
        }
        
    def decode_header_value(self, header_value: str) -> str:
        """Decode email header values that might be encoded"""
        if not header_value:
            return ''
            
        decoded_parts = decode_header(header_value)
        decoded_string = ''
        
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                try:
                    if encoding:
                        decoded_string += part.decode(encoding)
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                except:
                    decoded_string += part.decode('utf-8', errors='ignore')
            else:
                decoded_string += str(part)
                
        return decoded_string.strip()
        
    def parse_email_address(self, email_str: str) -> tuple:
        """Parse email string to extract name and email"""
        if not email_str:
            return '', ''
            
        # Handle format: "Name <email@domain.com>" or just "email@domain.com"
        email_match = re.search(r'<([^>]+)>', email_str)
        if email_match:
            email = email_match.group(1)
            name = email_str.replace(f'<{email}>', '').strip().strip('"')
        else:
            email = email_str.strip()
            name = ''
            
        return name, email
        
    def parse_email_addresses(self, email_str: str) -> List[str]:
        """Parse multiple email addresses from a string"""
        if not email_str:
            return []
            
        # Split by comma and parse each
        addresses = []
        for addr in email_str.split(','):
            _, email = self.parse_email_address(addr.strip())
            if email:
                addresses.append(email)
                
        return addresses
        
    def generate_thread_id(self, message: email.message.Message) -> str:
        """Generate thread ID for email conversation tracking"""
        # Try to get thread ID from headers
        thread_id = message.get('X-GM-THRID')  # Gmail thread ID
        if thread_id:
            return thread_id
            
        # Use In-Reply-To or References for threading
        in_reply_to = message.get('In-Reply-To')
        if in_reply_to:
            return hashlib.md5(in_reply_to.encode()).hexdigest()
            
        references = message.get('References')
        if references:
            return hashlib.md5(references.encode()).hexdigest()
            
        # Generate thread ID from subject (cleaned)
        subject = self.decode_header_value(message.get('Subject', ''))
        cleaned_subject = re.sub(r'^(RE:|FW:|FWD:)\s*', '', subject, flags=re.IGNORECASE)
        return hashlib.md5(cleaned_subject.encode()).hexdigest()
        
    def extract_body_content(self, message: email.message.Message) -> tuple:
        """Extract text and HTML body content from email"""
        body_text = ''
        body_html = ''
        
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = part.get('Content-Disposition', '')
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                    
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        content = payload.decode(charset, errors='ignore')
                        
                        if content_type == 'text/plain':
                            body_text += content
                        elif content_type == 'text/html':
                            body_html += content
                except:
                    continue
        else:
            # Single part message
            try:
                payload = message.get_payload(decode=True)
                if payload:
                    charset = message.get_content_charset() or 'utf-8'
                    content = payload.decode(charset, errors='ignore')
                    
                    if message.get_content_type() == 'text/html':
                        body_html = content
                    else:
                        body_text = content
            except:
                pass
                
        return body_text.strip(), body_html.strip()
        
    def has_attachments(self, message: email.message.Message) -> bool:
        """Check if email has attachments"""
        if message.is_multipart():
            for part in message.walk():
                content_disposition = part.get('Content-Disposition', '')
                if 'attachment' in content_disposition:
                    return True
        return False
        
    def process_attachments(self, message: email.message.Message, email_id: int):
        """Process and store email attachments"""
        if not message.is_multipart():
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for part in message.walk():
            content_disposition = part.get('Content-Disposition', '')
            if 'attachment' not in content_disposition:
                continue
                
            filename = part.get_filename()
            if not filename:
                continue
                
            try:
                # Get attachment data
                payload = part.get_payload(decode=True)
                content_type = part.get_content_type()
                size_bytes = len(payload) if payload else 0
                
                # Store attachment info (optionally store data)
                cursor.execute('''
                    INSERT INTO email_attachments 
                    (email_id, filename, content_type, size_bytes, attachment_data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (email_id, filename, content_type, size_bytes, payload))
                
            except Exception as e:
                self.logger.error(f"Error processing attachment {filename}: {e}")
                
        conn.commit()
        conn.close()
        
    def email_exists(self, email_hash: str) -> bool:
        """Check if email already exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM emails WHERE email_hash = ?', (email_hash,))
        exists = cursor.fetchone() is not None
        
        conn.close()
        return exists
        
    def store_email(self, email_data: Dict) -> Optional[int]:
        """Store email data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO emails (
                    message_id, subject, sender_name, sender_email, recipient_emails,
                    date_sent, body_text, body_html, has_attachments, thread_id,
                    folder, email_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                email_data['message_id'], email_data['subject'], email_data['sender_name'],
                email_data['sender_email'], email_data['recipient_emails'], email_data['date_sent'],
                email_data['body_text'], email_data['body_html'], email_data['has_attachments'],
                email_data['thread_id'], email_data['folder'], email_data['email_hash']
            ))
            
            email_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return email_id
            
        except Exception as e:
            self.logger.error(f"Error storing email: {e}")
            return None
            
    def update_thread_info(self, email_data: Dict):
        """Update email thread information"""
        if not email_data['thread_id']:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if thread exists
        cursor.execute('SELECT id, message_count FROM email_threads WHERE thread_id = ?', 
                      (email_data['thread_id'],))
        thread = cursor.fetchone()
        
        if thread:
            # Update existing thread
            cursor.execute('''
                UPDATE email_threads 
                SET message_count = message_count + 1,
                    last_message_date = ?
                WHERE thread_id = ?
            ''', (email_data['date_sent'], email_data['thread_id']))
        else:
            # Create new thread
            participants = [email_data['sender_email']]
            recipients = json.loads(email_data['recipient_emails'])
            participants.extend(recipients)
            
            cursor.execute('''
                INSERT INTO email_threads 
                (thread_id, subject, participants, first_message_date, last_message_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                email_data['thread_id'], email_data['subject'], 
                json.dumps(list(set(participants))),
                email_data['date_sent'], email_data['date_sent']
            ))
            
        conn.commit()
        conn.close()
        
    def search_emails(self, query: str, limit: int = 20) -> List[Dict]:
        """Search emails by content, subject, or sender"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, subject, sender_name, sender_email, date_sent, 
                   body_text, folder, has_attachments
            FROM emails
            WHERE subject LIKE ? OR body_text LIKE ? OR sender_name LIKE ?
            ORDER BY date_sent DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'subject': row[1],
                'sender_name': row[2],
                'sender_email': row[3],
                'date_sent': row[4],
                'body_preview': row[5][:200] + '...' if row[5] else '',
                'folder': row[6],
                'has_attachments': row[7]
            })
            
        conn.close()
        return results
        
    def get_email_threads(self, limit: int = 50) -> List[Dict]:
        """Get email conversation threads"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT thread_id, subject, participants, message_count, 
                   first_message_date, last_message_date
            FROM email_threads
            ORDER BY last_message_date DESC
            LIMIT ?
        ''', (limit,))
        
        threads = []
        for row in cursor.fetchall():
            threads.append({
                'thread_id': row[0],
                'subject': row[1],
                'participants': json.loads(row[2]),
                'message_count': row[3],
                'first_message_date': row[4],
                'last_message_date': row[5]
            })
            
        conn.close()
        return threads
        
    def get_import_stats(self) -> Dict:
        """Get email import statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM emails')
        total_emails = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM email_threads')
        total_threads = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM email_attachments')
        total_attachments = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT sender_email) FROM emails')
        unique_senders = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_emails': total_emails,
            'total_threads': total_threads,
            'total_attachments': total_attachments,
            'unique_senders': unique_senders
        }