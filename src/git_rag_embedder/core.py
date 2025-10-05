import os
from typing import List, Dict, Any, Optional, Set
import logging

from .repository_loader import RepositoryLoader
from .file_processor import FileProcessor
from .chunker import Chunker
from .embeddings import EmbeddingGenerator
from .serializer import EmbeddingSerializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitRAGEmbedder:
    """
    Main class for converting code repositories into embeddings.
    
    Supports both local directories and Git repositories. Provides a complete
    pipeline from repository loading to embedding generation and serialization.
    """
    
    def __init__(self, 
                 encoding_name: str = "cl100k_base",
                 embedding_backend: str = "sentence_transformers",
                 embedding_model: Optional[str] = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize GitRAGEmbedder.
        
        Args:
            encoding_name: Tokenizer encoding to use for chunking
            embedding_backend: Backend for embeddings ('sentence_transformers', 'openai', 'huggingface')
            embedding_model: Specific model to use for embeddings
            openai_api_key: OpenAI API key (only for 'openai' backend)
        """
        self.repository_loader = RepositoryLoader()
        self.file_processor = FileProcessor()
        self.chunker = Chunker(encoding_name)
        self.serializer = EmbeddingSerializer()
        
        # Initialize embedding generator
        backend_kwargs = {}
        if embedding_backend == 'openai':
            backend_kwargs['api_key'] = openai_api_key or os.getenv('OPENAI_API_KEY')
            if embedding_model:
                backend_kwargs['model'] = embedding_model
        elif embedding_backend in ['sentence_transformers', 'huggingface']:
            if embedding_model:
                backend_kwargs['model_name'] = embedding_model
        
        self.embedding_generator = EmbeddingGenerator(
            backend=embedding_backend,
            **backend_kwargs
        )
        
        self.setup_components()
        
    def setup_components(self):
        """Initialize internal components."""
        logger.info("Initializing GitRAGEmbedder...")
        logger.info(f"Using embedding backend: {self.embedding_generator.backend_type}")
        logger.info(f"Embedding dimension: {self.embedding_generator.get_embedding_dimension()}")
    
    def load_repository(self, source: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load repository from local path or Git URL.
        
        Args:
            source: Local directory path or Git repository URL
            local_path: Optional local path for cloning (uses temp dir if None)
            
        Returns:
            Dictionary containing repository metadata and local path
        """
        return self.repository_loader.load_repository(source, local_path)
    
    def process_files(self, repo_path: str, 
                     extensions: Optional[Set[str]] = None,
                     exclude_dirs: Optional[Set[str]] = None,
                     exclude_files: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Process repository files and extract content.
        
        Args:
            repo_path: Path to local repository
            extensions: File extensions to process (None for default text files)
            exclude_dirs: Directories to exclude from processing
            exclude_files: File patterns to exclude from processing
            
        Returns:
            List of processed documents with metadata and content
        """
        return self.file_processor.process_repository(
            repo_path, extensions, exclude_dirs, exclude_files
        )
    
    def chunk_documents(self, documents: List[Dict[str, Any]], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200,
                       strategy: str = "code_aware") -> List[Dict[str, Any]]:
        """
        Split documents into chunks for embedding generation.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: Chunking strategy ('token', 'code_aware', 'semantic')
            
        Returns:
            List of document chunks with metadata
        """
        return self.chunker.chunk_documents(
            documents, chunk_size, chunk_overlap, strategy
        )
    
    def generate_embeddings(self, 
                          chunks: List[Dict[str, Any]],
                          batch_size: int = 32,
                          max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            batch_size: Batch size for embedding generation
            max_retries: Maximum retries for failed embedding requests
            
        Returns:
            List of chunks with generated embeddings
        """
        return self.embedding_generator.generate_embeddings(
            chunks, batch_size, max_retries
        )
    
    def process_repository(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Complete pipeline to convert repository to embeddings.
        
        Args:
            source: Local directory path or Git repository URL
            **kwargs: Additional parameters for processing
            
        Returns:
            List of embedded chunks ready for export
        """
        logger.info(f"Starting repository processing pipeline: {source}")
        
        # Step 1: Repository loading
        repo_info = self.load_repository(source, kwargs.get('local_path'))
        repo_path = repo_info['local_path']
        
        # Step 2: File processing and content extraction
        documents = self.process_files(
            repo_path,
            extensions=kwargs.get('extensions'),
            exclude_dirs=kwargs.get('exclude_dirs'),
            exclude_files=kwargs.get('exclude_files')
        )
        
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        logger.info(f"Processed {len(documents)} files")
        
        # Step 3: Document chunking
        chunks = self.chunk_documents(
            documents, 
            chunk_size=kwargs.get('chunk_size', 1000),
            chunk_overlap=kwargs.get('chunk_overlap', 200),
            strategy=kwargs.get('chunk_strategy', 'code_aware')
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 4: Embedding generation
        embedded_chunks = self.generate_embeddings(
            chunks,
            batch_size=kwargs.get('batch_size', 32),
            max_retries=kwargs.get('max_retries', 3)
        )
        
        logger.info(f"Processing complete. Generated {len(embedded_chunks)} embedded chunks")
        return embedded_chunks
    
    def save_embeddings(self, 
                       embedded_chunks: List[Dict[str, Any]], 
                       file_path: str,
                       format: str = 'json') -> str:
        """
        Save embedded chunks to file.
        
        Args:
            embedded_chunks: Embedded chunks to save
            file_path: Path where to save the file
            format: Serialization format ('json', 'pkl', 'npz')
            
        Returns:
            Path to saved file
        """
        return self.serializer.save_embeddings(embedded_chunks, file_path, format)
    
    def load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load embedded chunks from file.
        
        Args:
            file_path: Path to the saved embeddings file
            
        Returns:
            List of embedded chunks
        """
        return self.serializer.load_embeddings(file_path)
    
    def get_embedding_stats(self, embedded_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about embedded chunks.
        
        Args:
            embedded_chunks: Embedded chunks to analyze
            
        Returns:
            Dictionary with statistics
        """
        return self.serializer.get_embedding_stats(embedded_chunks)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the chunker's encoding.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.chunker.count_tokens(text)
    
    def embed_single_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_generator.embed_single_text(text)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding configuration.
        
        Returns:
            Dictionary with embedding information
        """
        return {
            'backend': self.embedding_generator.backend_type,
            'dimension': self.embedding_generator.get_embedding_dimension()
        }
    
    def cleanup(self):
        """Clean up any temporary resources."""
        self.repository_loader.cleanup()