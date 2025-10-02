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
    
    This class provides a unified interface to access repository contents
    regardless of whether the source is a local directory or remote Git repository.
    """
    
    def __init__(self):
        """Initialize the repository loader."""
        self.temp_dirs = []  # Track temp dirs for cleanup if needed
    
    def load_repository(self, source: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load repository from either local path or Git URL.
        
        Args:
            source: Either a local directory path or Git repository URL
            local_path: Optional local path for cloning (uses temp dir if None)
            
        Returns:
            Dictionary containing repository metadata and local path
            
        Raises:
            ValueError: If source is invalid or repository cannot be accessed
        """
        logger.info(f"Loading repository from: {source}")
        
        if self._is_git_url(source):
            return self._clone_repository(source, local_path)
        elif self._is_local_directory(source):
            return self._load_local_repository(source)
        else:
            raise ValueError(f"Invalid repository source: {source}. Must be local path or Git URL.")
    
    def _is_git_url(self, source: str) -> bool:
        """
        Check if the source string is a Git repository URL.
        
        Args:
            source: Source string to check
            
        Returns:
            True if source appears to be a Git URL
        """
        git_indicators = ['git@', 'https://', 'http://', '.git']
        return any(indicator in source for indicator in git_indicators)
    
    def _is_local_directory(self, source: str) -> bool:
        """
        Check if the source string is a valid local directory.
        
        Args:
            source: Source string to check
            
        Returns:
            True if source is an existing local directory
        """
        return os.path.isdir(source)
    
    def _clone_repository(self, git_url: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Clone a Git repository to local directory.
        
        Args:
            git_url: Git repository URL to clone
            local_path: Local directory to clone into (uses temp dir if None)
            
        Returns:
            Dictionary with repository metadata and local path
            
        Raises:
            RuntimeError: If git clone fails
        """
        if local_path is None:
            # Create temporary directory for repository
            local_path = tempfile.mkdtemp(prefix="gitrag_")
            self.temp_dirs.append(local_path)
        
        try:
            # Build git clone command with minimal history for efficiency
            cmd = ['git', 'clone', '--depth', '1', '--single-branch', git_url, local_path]
            
            # Execute clone command
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
            raise RuntimeError("Git command not found. Please install git to clone repositories.")
    
    def _load_local_repository(self, local_path: str) -> Dict[str, Any]:
        """
        Load repository from local directory.
        
        Args:
            local_path: Path to local repository directory
            
        Returns:
            Dictionary with repository metadata
            
        Raises:
            ValueError: If local_path is not a valid directory
        """
        if not os.path.isdir(local_path):
            raise ValueError(f"Local path does not exist or is not a directory: {local_path}")
        
        # Check if it's a Git repository
        git_dir = os.path.join(local_path, '.git')
        is_git_repo = os.path.isdir(git_dir)
        
        logger.info(f"Loaded local repository from {local_path} (Git repo: {is_git_repo})")
        
        return {
            'source_type': 'local',
            'local_path': local_path,
            'is_git_repo': is_git_repo,
            'cloned': False
        }
    
    def cleanup(self):
        """Clean up any temporary directories created during cloning."""
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
        
        self.temp_dirs.clear()