import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Abstract base class for all embedding backends."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this backend."""
        pass


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI API-based embedding backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        """
        Initialize OpenAI embedding backend.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: OpenAI embedding model to use
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            self.model = model
            self.dimension = self._get_model_dimension(model)
            logger.info(f"Initialized OpenAI embeddings with model: {model}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def _get_model_dimension(self, model: str) -> int:
        """Get embedding dimension for known OpenAI models."""
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return model_dimensions.get(model, 1536)  # Default to 1536
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            # Return embeddings in the same order as input texts
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class SentenceTransformersBackend(EmbeddingBackend):
    """Local embedding backend using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformers backend.
        
        Args:
            model_name: Name of SentenceTransformers model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized SentenceTransformers with model: {model_name}")
        except ImportError:
            raise ImportError("SentenceTransformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformers: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using local SentenceTransformers model."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformers embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch using local SentenceTransformers model."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformers batch embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class HuggingFaceEmbeddingBackend(EmbeddingBackend):
    """Hugging Face transformers-based embedding backend."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Hugging Face embedding backend.
        
        Args:
            model_name: Name of Hugging Face model
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model_name = model_name
            
            # Get embedding dimension
            with torch.no_grad():
                dummy_input = self.tokenizer("test", return_tensors="pt")
                output = self.model(**dummy_input)
                self.dimension = output.last_hidden_state.size(-1)
            
            logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
        except ImportError:
            raise ImportError("Transformers or torch not installed. Run: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace model: {e}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using Hugging Face model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            return sentence_embedding[0].numpy().tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch using Hugging Face model."""
        try:
            import torch
            
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            return sentence_embeddings.numpy().tolist()
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class EmbeddingGenerator:
    """
    Main embedding generator that supports multiple backends.
    
    Handles batching, error handling, and provides a unified interface
    for different embedding providers.
    """
    
    def __init__(self, backend: str = "sentence_transformers", **backend_kwargs):
        """
        Initialize embedding generator with specified backend.
        
        Args:
            backend: One of 'openai', 'sentence_transformers', 'huggingface'
            **backend_kwargs: Arguments passed to the backend constructor
        """
        self.backend_type = backend
        self.backend = self._initialize_backend(backend, backend_kwargs)
        logger.info(f"Initialized EmbeddingGenerator with {backend} backend")
    
    def _initialize_backend(self, backend: str, backend_kwargs: dict) -> EmbeddingBackend:
        """Initialize the specified embedding backend."""
        backends = {
            'openai': OpenAIEmbeddingBackend,
            'sentence_transformers': SentenceTransformersBackend,
            'huggingface': HuggingFaceEmbeddingBackend,
        }
        
        if backend not in backends:
            raise ValueError(f"Unsupported backend: {backend}. Available: {list(backends.keys())}")
        
        return backends[backend](**backend_kwargs)
    
    def generate_embeddings(self, 
                          chunks: List[Dict[str, Any]],
                          batch_size: int = 32,
                          max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunks with content
            batch_size: Number of chunks to process in each batch
            max_retries: Maximum number of retries for failed requests
            
        Returns:
            List of chunks with added embedding fields
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.backend_type}")
        
        # Extract texts from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.backend.embed_batch(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to generate embeddings after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, all_embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk.update({
                'embedding': embedding,
                'embedding_dimension': len(embedding),
                'embedding_model': self.backend_type,
                'embedding_norm': float(np.linalg.norm(embedding))
            })
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the current backend."""
        return self.backend.get_embedding_dimension()
    
    def embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        return self.backend.embed_text(text)