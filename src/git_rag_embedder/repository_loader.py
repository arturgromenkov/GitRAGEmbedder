import os
import tempfile
import subprocess
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RepositoryLoader:
    """
    Handles loading of code repositories from both local paths and Git URLs.
    """
    
    def __init__(self):
        self.temp_dirs = []
    
    def load_repository(self, source: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load repository from either local path or Git URL.
        """
        logger.info(f"Loading repository from: {source}")
        
        if self._is_git_url(source):
            return self._clone_repository(source, local_path)
        elif self._is_local_directory(source):
            return self._load_local_repository(source)
        else:
            raise ValueError(f"Invalid repository source: {source}")
    
    def _is_git_url(self, source: str) -> bool:
        git_indicators = ['git@', 'https://', 'http://', '.git']
        return any(indicator in source for indicator in git_indicators)
    
    def _is_local_directory(self, source: str) -> bool:
        return os.path.isdir(source)
    
    def _clone_repository(self, git_url: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        if local_path is None:
            local_path = tempfile.mkdtemp(prefix="gitrag_")
            self.temp_dirs.append(local_path)
        
        try:
            cmd = ['git', 'clone', '--depth', '1', '--single-branch', git_url, local_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned repository to {local_path}")
                return {
                    'source_type': 'git',
                    'source_url': git_url,
                    'local_path': local_path,
                    'cloned': True
                }
            else:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git clone failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Git command not found.")
    
    def _load_local_repository(self, local_path: str) -> Dict[str, Any]:
        if not os.path.isdir(local_path):
            raise ValueError(f"Local path does not exist: {local_path}")
        
        git_dir = os.path.join(local_path, '.git')
        is_git_repo = os.path.isdir(git_dir)
        
        logger.info(f"Loaded local repository from {local_path}")
        
        return {
            'source_type': 'local',
            'local_path': local_path,
            'is_git_repo': is_git_repo,
            'cloned': False
        }
    
    def cleanup(self):
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")
        
        self.temp_dirs.clear()