import os
import fnmatch
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Processes repository files to extract content and metadata.
    """
    
    DEFAULT_TEXT_EXTENSIONS = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
        '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
        '.html', '.css', '.scss', '.less', '.xml', '.json', '.yaml', '.yml',
        '.md', '.txt', '.rst', '.tex', '.org', '.sql', '.sh', '.bash'
    }
    
    DEFAULT_EXCLUDE_DIRS = {
        '.git', '__pycache__', 'node_modules', 'vendor', 'dist', 'build',
        'target', '.idea', '.vscode', 'coverage', '.pytest_cache'
    }
    
    DEFAULT_EXCLUDE_FILES = {
        'package-lock.json', 'yarn.lock', '*.pyc', '*.so', '*.dll', '*.exe'
    }
    
    def __init__(self):
        self.text_extensions = self.DEFAULT_TEXT_EXTENSIONS.copy()
        self.exclude_dirs = self.DEFAULT_EXCLUDE_DIRS.copy()
        self.exclude_files = self.DEFAULT_EXCLUDE_FILES.copy()
    
    def process_repository(self, repo_path: str, 
                          extensions: Optional[Set[str]] = None,
                          exclude_dirs: Optional[Set[str]] = None,
                          exclude_files: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(repo_path):
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        target_extensions = extensions or self.text_extensions
        excluded_dirs = exclude_dirs or self.exclude_dirs
        excluded_files = exclude_files or self.exclude_files
        
        logger.info(f"Processing repository: {repo_path}")
        
        documents = []
        processed_files = 0
        skipped_files = 0
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                if self._should_exclude_file(file, relative_path, excluded_files):
                    skipped_files += 1
                    continue
                
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext not in target_extensions:
                    skipped_files += 1
                    continue
                
                document = self._process_file(file_path, relative_path, repo_path)
                if document:
                    documents.append(document)
                    processed_files += 1
        
        logger.info(f"File processing complete: {processed_files} files processed, {skipped_files} files skipped")
        return documents
    
    def _should_exclude_file(self, filename: str, relative_path: str, excluded_files: Set[str]) -> bool:
        for pattern in excluded_files:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(relative_path, pattern):
                return True
        return False
    
    def _process_file(self, file_path: str, relative_path: str, repo_root: str) -> Optional[Dict[str, Any]]:
        try:
            if self._is_binary_file(file_path):
                logger.debug(f"Skipping binary file: {relative_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return None
            
            return {
                'file_path': relative_path,
                'absolute_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_extension': os.path.splitext(file_path)[1],
                'content': content,
                'content_length': len(content),
                'repo_root': repo_root
            }
            
        except Exception as e:
            logger.warning(f"Failed to process file {relative_path}: {e}")
            return None
    
    def _is_binary_file(self, file_path: str) -> bool:
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\0' in chunk:
                    return True
                text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
                return bool(chunk.translate(None, text_chars))
        except Exception:
            return True