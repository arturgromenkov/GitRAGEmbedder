import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        pass


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
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
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return model_dimensions.get(model, 1536)
    
    def embed_text(self, text: str) -> List[float]:
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
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class SentenceTransformersBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
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
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformers embedding failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformers batch embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self.dimension


class HuggingFaceEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model_name = model_name
            
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
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_text(self, text: str) -> List[float]:
        try:
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
    def __init__(self, backend: str = "sentence_transformers", **backend_kwargs):
        self.backend_type = backend
        self.backend = self._initialize_backend(backend, backend_kwargs)
        logger.info(f"Initialized EmbeddingGenerator with {backend} backend")
    
    def _initialize_backend(self, backend: str, backend_kwargs: dict) -> EmbeddingBackend:
        backends = {
            'openai': OpenAIEmbeddingBackend,
            'sentence_transformers': SentenceTransformersBackend,
            'huggingface': HuggingFaceEmbeddingBackend,
        }
        
        if backend not in backends:
            raise ValueError(f"Unsupported backend: {backend}")
        
        return backends[backend](**backend_kwargs)
    
    def generate_embeddings(self, 
                          chunks: List[Dict[str, Any]],
                          batch_size: int = 32,
                          max_retries: int = 3) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.backend_type}")
        
        texts = [chunk['content'] for chunk in chunks]
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
        return self.backend.get_embedding_dimension()
    
    def embed_single_text(self, text: str) -> List[float]:
        return self.backend.embed_text(text)