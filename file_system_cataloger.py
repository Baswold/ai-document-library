"""
File System Cataloger - Automatically index and search the entire computer
Supports glob patterns, grep-style content search, and smart filtering
"""

import os
import sqlite3
import hashlib
import mimetypes
import fnmatch
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
from datetime import datetime
import threading
import time


class FileSystemCatalog:
    """Manages cataloging and searching of the entire file system"""

    # Directories to skip by default (system/temp directories)
    DEFAULT_SKIP_DIRS = {
        # System directories
        '/proc', '/sys', '/dev', '/run', '/tmp', '/var/tmp',
        # Common OS-specific
        '/boot', '/lost+found', '/snap', '/swapfile',
        # Windows
        'C:\\Windows', 'C:\\Program Files', 'C:\\Program Files (x86)',
        'C:\\ProgramData', 'C:\\$Recycle.Bin',
        # macOS
        '/System', '/Library', '/private', '/Volumes',
        # Hidden and cache directories
        '.git', '.svn', '.hg', '__pycache__', 'node_modules', '.cache',
        '.npm', '.gradle', '.m2', '.cargo', 'venv', '.venv', 'env',
        # Browser caches
        '.mozilla', '.chrome', '.firefox', 'AppData', 'Application Data',
    }

    # File extensions to skip (binary/compiled files)
    DEFAULT_SKIP_EXTENSIONS = {
        '.pyc', '.pyo', '.so', '.dll', '.dylib', '.exe', '.bin',
        '.o', '.obj', '.class', '.jar', '.war', '.iso', '.img',
        '.vmdk', '.vdi', '.qcow2',
    }

    # Maximum file size to index content (10 MB)
    MAX_CONTENT_SIZE = 10 * 1024 * 1024

    # Supported text file extensions for content indexing
    TEXT_EXTENSIONS = {
        '.txt', '.md', '.rst', '.log', '.csv', '.json', '.xml', '.yaml', '.yml',
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.sh', '.bash',
        '.html', '.htm', '.css', '.scss', '.sass', '.less', '.sql',
        '.r', '.m', '.scala', '.pl', '.lua', '.vim', '.el',
        '.toml', '.ini', '.cfg', '.conf', '.properties',
    }

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_catalog_database()
        self._scanning = False
        self._scan_thread = None
        self._stop_scan = False
        self.scan_progress_callback = None

    def init_catalog_database(self):
        """Initialize database for file system catalog"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # File catalog table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                directory TEXT NOT NULL,
                extension TEXT,
                file_size INTEGER,
                mime_type TEXT,
                file_hash TEXT,
                content_indexed BOOLEAN DEFAULT FALSE,
                file_content TEXT,
                last_modified DATETIME,
                last_scanned DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_readable BOOLEAN DEFAULT TRUE
            )
        ''')

        # Create indexes for faster searching
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_filename
            ON file_catalog(filename)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_directory
            ON file_catalog(directory)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_extension
            ON file_catalog(extension)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_filepath
            ON file_catalog(filepath)
        ''')

        # Scan configuration table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_directories TEXT,
                skip_directories TEXT,
                skip_extensions TEXT,
                last_full_scan DATETIME
            )
        ''')

        conn.commit()
        conn.close()

    def should_skip_directory(self, dir_path: str, custom_skip: Optional[Set[str]] = None) -> bool:
        """Check if a directory should be skipped"""
        skip_dirs = self.DEFAULT_SKIP_DIRS.copy()
        if custom_skip:
            skip_dirs.update(custom_skip)

        dir_path_obj = Path(dir_path)

        # Check exact matches
        if str(dir_path) in skip_dirs:
            return True

        # Check if any parent directory is in skip list
        for skip_dir in skip_dirs:
            try:
                if str(dir_path_obj).startswith(str(skip_dir)):
                    return True
                # Check directory name matches (for things like node_modules anywhere)
                if dir_path_obj.name == skip_dir or dir_path_obj.name == skip_dir.lstrip('/').lstrip('.'):
                    return True
            except:
                pass

        # Skip hidden directories
        if dir_path_obj.name.startswith('.') and dir_path_obj.name != '.':
            return True

        return False

    def should_skip_file(self, file_path: str, custom_skip_ext: Optional[Set[str]] = None) -> bool:
        """Check if a file should be skipped"""
        skip_ext = self.DEFAULT_SKIP_EXTENSIONS.copy()
        if custom_skip_ext:
            skip_ext.update(custom_skip_ext)

        file_path_obj = Path(file_path)

        # Check extension
        if file_path_obj.suffix.lower() in skip_ext:
            return True

        # Skip very large files
        try:
            if file_path_obj.stat().st_size > 100 * 1024 * 1024:  # 100 MB
                return True
        except:
            return True

        return False

    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate MD5 hash of file for change detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None

    def extract_text_content(self, file_path: str) -> Optional[str]:
        """Extract text content from file if it's a text file"""
        file_path_obj = Path(file_path)

        # Only index text files
        if file_path_obj.suffix.lower() not in self.TEXT_EXTENSIONS:
            return None

        # Check file size
        try:
            if file_path_obj.stat().st_size > self.MAX_CONTENT_SIZE:
                return None
        except:
            return None

        # Try to read as text
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    return content
            except:
                continue

        return None

    def scan_directory(self,
                      root_directory: str,
                      custom_skip_dirs: Optional[Set[str]] = None,
                      custom_skip_ext: Optional[Set[str]] = None,
                      progress_callback: Optional[Callable] = None) -> Dict[str, int]:
        """
        Scan a directory tree and catalog all files
        Returns stats about the scan
        """
        stats = {
            'files_scanned': 0,
            'files_indexed': 0,
            'files_skipped': 0,
            'directories_scanned': 0,
            'errors': 0
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            for root, dirs, files in os.walk(root_directory):
                # Check if scan should stop
                if self._stop_scan:
                    break

                # Filter out directories to skip
                dirs[:] = [d for d in dirs if not self.should_skip_directory(
                    os.path.join(root, d), custom_skip_dirs)]

                stats['directories_scanned'] += 1

                for filename in files:
                    if self._stop_scan:
                        break

                    file_path = os.path.join(root, filename)

                    # Skip certain files
                    if self.should_skip_file(file_path, custom_skip_ext):
                        stats['files_skipped'] += 1
                        continue

                    try:
                        stats['files_scanned'] += 1

                        # Get file info
                        file_stat = os.stat(file_path)
                        file_size = file_stat.st_size
                        last_modified = datetime.fromtimestamp(file_stat.st_mtime)

                        # Get MIME type
                        mime_type, _ = mimetypes.guess_type(file_path)

                        # Calculate hash
                        file_hash = self.calculate_file_hash(file_path)

                        # Extract content if text file
                        content = self.extract_text_content(file_path)
                        content_indexed = content is not None

                        # Insert or update in database
                        cursor.execute('''
                            INSERT OR REPLACE INTO file_catalog
                            (filepath, filename, directory, extension, file_size,
                             mime_type, file_hash, content_indexed, file_content,
                             last_modified, last_scanned, is_readable)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            file_path, filename, root,
                            Path(file_path).suffix.lower(),
                            file_size, mime_type, file_hash, content_indexed, content,
                            last_modified, datetime.now(), True
                        ))

                        stats['files_indexed'] += 1

                        # Progress callback
                        if progress_callback and stats['files_scanned'] % 100 == 0:
                            progress_callback(stats)

                    except Exception as e:
                        stats['errors'] += 1
                        print(f"Error indexing {file_path}: {e}")

                # Commit periodically
                if stats['files_scanned'] % 1000 == 0:
                    conn.commit()

            conn.commit()

        finally:
            conn.close()

        return stats

    def scan_multiple_directories(self,
                                  directories: List[str],
                                  progress_callback: Optional[Callable] = None) -> Dict[str, int]:
        """Scan multiple directories and return combined stats"""
        total_stats = {
            'files_scanned': 0,
            'files_indexed': 0,
            'files_skipped': 0,
            'directories_scanned': 0,
            'errors': 0
        }

        for directory in directories:
            if self._stop_scan:
                break

            stats = self.scan_directory(directory, progress_callback=progress_callback)

            for key in total_stats:
                total_stats[key] += stats[key]

        return total_stats

    def start_background_scan(self,
                            directories: List[str],
                            progress_callback: Optional[Callable] = None):
        """Start scanning in background thread"""
        if self._scanning:
            return False

        self._scanning = True
        self._stop_scan = False

        def scan_worker():
            try:
                stats = self.scan_multiple_directories(directories, progress_callback)
                if progress_callback:
                    progress_callback(stats, completed=True)
            finally:
                self._scanning = False

        self._scan_thread = threading.Thread(target=scan_worker, daemon=True)
        self._scan_thread.start()
        return True

    def stop_scan(self):
        """Stop ongoing scan"""
        self._stop_scan = True
        if self._scan_thread:
            self._scan_thread.join(timeout=5)
        self._scanning = False

    def glob_search(self, pattern: str, limit: int = 100) -> List[Dict]:
        """
        Search files using glob patterns
        Examples: '*.py', '**/*.txt', '/home/user/*.md'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all files from catalog
        cursor.execute('''
            SELECT id, filepath, filename, directory, extension,
                   file_size, mime_type, last_modified
            FROM file_catalog
            ORDER BY last_modified DESC
        ''')

        results = []
        for row in cursor.fetchall():
            filepath = row[1]

            # Use fnmatch for glob pattern matching
            if fnmatch.fnmatch(filepath, pattern) or fnmatch.fnmatch(row[2], pattern):
                results.append({
                    'id': row[0],
                    'filepath': filepath,
                    'filename': row[2],
                    'directory': row[3],
                    'extension': row[4],
                    'file_size': row[5],
                    'mime_type': row[6],
                    'last_modified': row[7]
                })

                if len(results) >= limit:
                    break

        conn.close()
        return results

    def grep_search(self, pattern: str, limit: int = 100, regex: bool = True,
                   case_sensitive: bool = False) -> List[Dict]:
        """
        Search file contents using regex patterns (like grep)

        Args:
            pattern: Search pattern (regex or plain text)
            limit: Maximum number of results
            regex: Use regex matching (True) or plain text (False)
            case_sensitive: Case sensitive matching
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Only search files with indexed content
        cursor.execute('''
            SELECT id, filepath, filename, directory, file_content, extension
            FROM file_catalog
            WHERE content_indexed = TRUE AND file_content IS NOT NULL
            ORDER BY last_modified DESC
        ''')

        results = []

        # Compile regex pattern if needed
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled_pattern = re.compile(pattern, flags)
            except re.error:
                # Invalid regex, treat as plain text
                regex = False

        for row in cursor.fetchall():
            file_id, filepath, filename, directory, content, extension = row

            # Search in content
            matches = []
            if regex:
                matches = list(compiled_pattern.finditer(content))
            else:
                # Plain text search
                search_content = content if case_sensitive else content.lower()
                search_pattern = pattern if case_sensitive else pattern.lower()

                start = 0
                while True:
                    pos = search_content.find(search_pattern, start)
                    if pos == -1:
                        break
                    matches.append((pos, pos + len(search_pattern)))
                    start = pos + 1

            if matches:
                # Extract context around matches
                match_contexts = []
                for match in matches[:5]:  # Limit to 5 matches per file
                    if regex:
                        pos = match.start()
                        matched_text = match.group(0)
                    else:
                        pos, end_pos = match
                        matched_text = content[pos:end_pos]

                    # Get line containing the match
                    line_start = content.rfind('\n', 0, pos) + 1
                    line_end = content.find('\n', pos)
                    if line_end == -1:
                        line_end = len(content)

                    line = content[line_start:line_end]
                    line_num = content[:pos].count('\n') + 1

                    match_contexts.append({
                        'line_number': line_num,
                        'line_content': line,
                        'matched_text': matched_text
                    })

                results.append({
                    'id': file_id,
                    'filepath': filepath,
                    'filename': filename,
                    'directory': directory,
                    'extension': extension,
                    'match_count': len(matches),
                    'matches': match_contexts
                })

                if len(results) >= limit:
                    break

        conn.close()
        return results

    def combined_search(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Combined search: filename, path, and content
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Search in filename, path, and content
        cursor.execute('''
            SELECT id, filepath, filename, directory, extension,
                   file_size, mime_type, last_modified,
                   content_indexed, file_content
            FROM file_catalog
            WHERE filename LIKE ?
               OR filepath LIKE ?
               OR (content_indexed = TRUE AND file_content LIKE ?)
            ORDER BY last_modified DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))

        results = []
        for row in cursor.fetchall():
            result = {
                'id': row[0],
                'filepath': row[1],
                'filename': row[2],
                'directory': row[3],
                'extension': row[4],
                'file_size': row[5],
                'mime_type': row[6],
                'last_modified': row[7],
                'content_indexed': row[8]
            }

            # Add match context if found in content
            if row[8] and row[9] and query.lower() in row[9].lower():
                result['matched_in_content'] = True
                # Extract snippet
                content = row[9]
                pos = content.lower().find(query.lower())
                start = max(0, pos - 50)
                end = min(len(content), pos + len(query) + 50)
                result['content_snippet'] = content[start:end]
            else:
                result['matched_in_content'] = False

            results.append(result)

        conn.close()
        return results

    def get_catalog_stats(self) -> Dict:
        """Get statistics about the catalog"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM file_catalog')
        total_files = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM file_catalog WHERE content_indexed = TRUE')
        indexed_files = cursor.fetchone()[0]

        cursor.execute('SELECT SUM(file_size) FROM file_catalog')
        total_size = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(DISTINCT directory) FROM file_catalog')
        total_dirs = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT extension) FROM file_catalog')
        total_extensions = cursor.fetchone()[0]

        conn.close()

        return {
            'total_files': total_files,
            'content_indexed_files': indexed_files,
            'total_size_bytes': total_size,
            'total_directories': total_dirs,
            'total_extensions': total_extensions
        }
