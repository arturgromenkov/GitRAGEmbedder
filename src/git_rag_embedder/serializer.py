import json
import pickle
import numpy as np
from typing import List, Dict, Any
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class EmbeddingSerializer:
    """
    Handles serialization and deserialization of embedded chunks.
    """
    
    def __init__(self):
        self.supported_formats = {'json', 'pkl', 'npz'}
    
    def save_embeddings(self, 
                       embedded_chunks: List[Dict[str, Any]], 
                       file_path: str,
                       format: str = 'json') -> str:
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            return self._save_json(embedded_chunks, file_path)
        elif format == 'pkl':
            return self._save_pickle(embedded_chunks, file_path)
        elif format == 'npz':
            return self._save_npz(embedded_chunks, file_path)
    
    def _save_json(self, embedded_chunks: List[Dict[str, Any]], file_path: str) -> str:
        serializable_chunks = []
        for chunk in embedded_chunks:
            serializable_chunk = chunk.copy()
            if 'embedding' in serializable_chunk and isinstance(serializable_chunk['embedding'], np.ndarray):
                serializable_chunk['embedding'] = serializable_chunk['embedding'].tolist()
            serializable_chunks.append(serializable_chunk)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(serializable_chunks)} embeddings to {file_path} (JSON)")
        return file_path
    
    def _save_pickle(self, embedded_chunks: List[Dict[str, Any]], file_path: str) -> str:
        with open(file_path, 'wb') as f:
            pickle.dump(embedded_chunks, f)
        
        logger.info(f"Saved {len(embedded_chunks)} embeddings to {file_path} (Pickle)")
        return file_path
    
    def _save_npz(self, embedded_chunks: List[Dict[str, Any]], file_path: str) -> str:
        if not embedded_chunks:
            raise ValueError("No embeddings to save")
        
        embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks])
        metadata = []
        for chunk in embedded_chunks:
            metadata_chunk = chunk.copy()
            metadata_chunk.pop('embedding', None)
            metadata.append(metadata_chunk)
        
        np.savez_compressed(file_path, embeddings=embeddings, metadata=metadata)
        
        logger.info(f"Saved {len(embedded_chunks)} embeddings to {file_path} (NPZ)")
        return file_path
    
    def load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            return self._load_json(file_path)
        elif file_extension == '.pkl':
            return self._load_pickle(file_path)
        elif file_extension == '.npz':
            return self._load_npz(file_path)
        else:
            raise ValueError(f"Unrecognized file format: {file_extension}")
    
    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            embedded_chunks = json.load(f)
        
        logger.info(f"Loaded {len(embedded_chunks)} embeddings from {file_path}")
        return embedded_chunks
    
    def _load_pickle(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'rb') as f:
            embedded_chunks = pickle.load(f)
        
        logger.info(f"Loaded {len(embedded_chunks)} embeddings from {file_path}")
        return embedded_chunks
    
    def _load_npz(self, file_path: str) -> List[Dict[str, Any]]:
        data = np.load(file_path, allow_pickle=True)
        embeddings = data['embeddings']
        metadata = data['metadata']
        
        embedded_chunks = []
        for i, meta in enumerate(metadata):
            chunk = meta.copy()
            chunk['embedding'] = embeddings[i]
            embedded_chunks.append(chunk)
        
        logger.info(f"Loaded {len(embedded_chunks)} embeddings from {file_path}")
        return embedded_chunks
    
    def get_embedding_stats(self, embedded_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not embedded_chunks:
            return {}
        
        embeddings = [chunk['embedding'] for chunk in embedded_chunks if 'embedding' in chunk]
        
        if not embeddings:
            return {}
        
        embedding_array = np.array(embeddings)
        
        return {
            'total_chunks': len(embedded_chunks),
            'embedding_dimension': embedding_array.shape[1],
            'mean_embedding_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
            'std_embedding_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings])),
            'file_types': list(set(chunk.get('file_extension', '') for chunk in embedded_chunks)),
            'total_files': len(set(chunk.get('file_path', '') for chunk in embedded_chunks)),
        }