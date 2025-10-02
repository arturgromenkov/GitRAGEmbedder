import tiktoken
from typing import List, Dict, Any, Optional, Callable
import logging
import re

logger = logging.getLogger(__name__)


class Chunker:
    """
    Handles intelligent chunking of code and text documents.
    
    Supports multiple chunking strategies optimized for different types of content:
    - Token-based chunking for general text
    - Code-aware chunking that respects language structure
    - Semantic chunking that preserves logical boundaries
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize chunker with specified token encoding.
        
        Args:
            encoding_name: Tokenizer encoding to use (cl100k_base works for most models)
        """
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            logger.info(f"Initialized chunker with encoding: {encoding_name}")
        except Exception as e:
            logger.error(f"Failed to initialize encoding {encoding_name}: {e}")
            raise
    
    def chunk_documents(self, 
                       documents: List[Dict[str, Any]], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200,
                       strategy: str = "code_aware") -> List[Dict[str, Any]]:
        """
        Split documents into chunks using specified strategy.
        
        Args:
            documents: List of documents with content and metadata
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: Chunking strategy ('token', 'code_aware', 'semantic')
            
        Returns:
            List of chunks with metadata and content
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        logger.info(f"Chunking {len(documents)} documents with {strategy} strategy")
        
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_single_document(doc, chunk_size, chunk_overlap, strategy)
            all_chunks.extend(chunks)
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _chunk_single_document(self, 
                             document: Dict[str, Any],
                             chunk_size: int,
                             chunk_overlap: int,
                             strategy: str) -> List[Dict[str, Any]]:
        """
        Chunk a single document based on the specified strategy.
        
        Args:
            document: Document dictionary with content and metadata
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: Chunking strategy to use
            
        Returns:
            List of chunks from the single document
        """
        content = document.get('content', '')
        file_extension = document.get('file_extension', '').lower()
        
        if not content.strip():
            return []
        
        # Choose chunking strategy based on file type and strategy parameter
        if strategy == "code_aware" and self._is_code_file(file_extension):
            return self._code_aware_chunking(document, chunk_size, chunk_overlap)
        elif strategy == "semantic":
            return self._semantic_chunking(document, chunk_size, chunk_overlap)
        else:
            return self._token_based_chunking(document, chunk_size, chunk_overlap)
    
    def _is_code_file(self, file_extension: str) -> bool:
        """
        Check if file extension corresponds to a programming language.
        
        Args:
            file_extension: File extension including dot (e.g., '.py', '.js')
            
        Returns:
            True if file is a code file
        """
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.r',
            '.sql', '.sh', '.bash', '.ps1', '.bat'
        }
        return file_extension in code_extensions
    
    def _token_based_chunking(self, 
                            document: Dict[str, Any],
                            chunk_size: int,
                            chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Simple token-based chunking that works for any text.
        
        Args:
            document: Document to chunk
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of token-based chunks
        """
        content = document['content']
        tokens = self.encoding.encode(content)
        
        if len(tokens) <= chunk_size:
            return [self._create_chunk(document, content, 0, len(content), 1, 1)]
        
        chunks = []
        start_idx = 0
        chunk_id = 1
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            # Convert tokens back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk = self._create_chunk(document, chunk_text, start_idx, end_idx, chunk_id, len(tokens))
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += chunk_size - chunk_overlap
            chunk_id += 1
            
            # Avoid infinite loop with very small chunks
            if start_idx >= len(tokens):
                break
        
        return chunks
    
    def _code_aware_chunking(self,
                           document: Dict[str, Any],
                           chunk_size: int,
                           chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Code-aware chunking that tries to preserve function/class boundaries.
        
        Args:
            document: Code document to chunk
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of code-aware chunks
        """
        content = document['content']
        file_extension = document.get('file_extension', '').lower()
        
        # First, try to split by logical code boundaries
        segments = self._split_code_by_boundaries(content, file_extension)
        
        # If logical splitting doesn't work or segments are too large, fall back to token chunking
        if not segments or max(len(seg) for seg in segments) > chunk_size * 4:
            return self._token_based_chunking(document, chunk_size, chunk_overlap)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 1
        
        for segment in segments:
            segment_tokens = len(self.encoding.encode(segment))
            
            # If adding this segment would exceed chunk size, finalize current chunk
            if current_tokens + segment_tokens > chunk_size and current_chunk:
                chunk = self._create_chunk(document, current_chunk.strip(), 0, 0, chunk_id, 0)
                chunks.append(chunk)
                
                # Start new chunk with overlap from previous chunk
                if chunk_overlap > 0 and chunks:
                    # For overlap, take the end of previous chunk
                    overlap_text = self._get_chunk_overlap(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + segment
                    current_tokens = len(self.encoding.encode(current_chunk))
                else:
                    current_chunk = segment
                    current_tokens = segment_tokens
                
                chunk_id += 1
            else:
                # Add segment to current chunk
                current_chunk += segment
                current_tokens += segment_tokens
        
        # Add the final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(document, current_chunk.strip(), 0, 0, chunk_id, 0)
            chunks.append(chunk)
        
        return chunks
    
    def _split_code_by_boundaries(self, content: str, file_extension: str) -> List[str]:
        """
        Split code content by logical boundaries (functions, classes, etc.).
        
        Args:
            content: Code content to split
            file_extension: File extension to determine language rules
            
        Returns:
            List of code segments
        """
        segments = []
        
        if file_extension == '.py':
            # Python: split by functions, classes, and top-level blocks
            segments = self._split_python_code(content)
        elif file_extension in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript: split by functions, classes, components
            segments = self._split_javascript_code(content)
        elif file_extension in ['.java', '.cpp', '.c', '.cs']:
            # C-like languages: split by classes, methods, functions
            segments = self._split_clike_code(content)
        else:
            # For other languages, use simple paragraph splitting
            segments = self._split_by_paragraphs(content)
        
        return [seg for seg in segments if seg.strip()]
    
    def _split_python_code(self, content: str) -> List[str]:
        """Split Python code by functions, classes, and imports."""
        segments = []
        current_segment = ""
        
        # Split by major constructs while keeping related blocks together
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for major constructs (class, def, import, from, decorators)
            if (stripped.startswith(('def ', 'class ', '@')) or 
                (stripped.startswith(('import ', 'from ')) and ' import ' in stripped)):
                
                # Save current segment if not empty
                if current_segment.strip():
                    segments.append(current_segment)
                    current_segment = ""
            
            current_segment += line + '\n'
            i += 1
        
        # Add the final segment
        if current_segment.strip():
            segments.append(current_segment)
        
        return segments if segments else [content]
    
    def _split_javascript_code(self, content: str) -> List[str]:
        """Split JavaScript/TypeScript code by functions, classes, and components."""
        segments = []
        
        # Simple approach: split by function, class, and component definitions
        patterns = [
            r'(export\s+)?(default\s+)?(class|function|const|let|var)\s+\w+',  # Class/function definitions
            r'export\s+default',  # Default exports
            r'import\s+.*?from',  # Import statements
        ]
        
        # Use the first pattern that produces meaningful splits
        for pattern in patterns:
            splits = re.split(f'({pattern})', content)
            if len(splits) > 1:
                # Reconstruct segments with the split delimiters
                reconstructed = []
                for i in range(0, len(splits), 2):
                    if i + 1 < len(splits):
                        segment = splits[i] + splits[i + 1]
                        if segment.strip():
                            reconstructed.append(segment)
                if reconstructed:
                    return reconstructed
        
        return [content]
    
    def _split_clike_code(self, content: str) -> List[str]:
        """Split C-like language code by classes and functions."""
        segments = []
        
        # Split by class and function definitions
        patterns = [
            r'(public|private|protected)?\s*(class|interface|struct)\s+\w+',  # Class definitions
            r'(public|private|protected)?\s*\w+\s+\w+\s*\(',  # Method definitions
            r'#include',  # Include statements
        ]
        
        for pattern in patterns:
            splits = re.split(f'({pattern})', content)
            if len(splits) > 1:
                reconstructed = []
                for i in range(0, len(splits), 2):
                    if i + 1 < len(splits):
                        segment = splits[i] + splits[i + 1]
                        if segment.strip():
                            reconstructed.append(segment)
                if reconstructed:
                    return reconstructed
        
        return [content]
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraphs or blank lines."""
        return [p for p in re.split(r'\n\s*\n', content) if p.strip()]
    
    def _semantic_chunking(self,
                          document: Dict[str, Any],
                          chunk_size: int,
                          chunk_overlap: int) -> List[Dict[str, Any]]:
        """
        Semantic chunking that preserves logical sections and headings.
        
        Args:
            document: Document to chunk
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of semantically coherent chunks
        """
        content = document['content']
        file_extension = document.get('file_extension', '').lower()
        
        # For markdown and text files, split by headings
        if file_extension in ['.md', '.rst', '.txt']:
            segments = self._split_by_headings(content)
        else:
            # For other files, use paragraph splitting as semantic boundaries
            segments = self._split_by_paragraphs(content)
        
        # If segments are too large, further split them
        refined_segments = []
        for segment in segments:
            segment_tokens = len(self.encoding.encode(segment))
            if segment_tokens > chunk_size:
                # Further split large segments using token-based approach
                sub_doc = document.copy()
                sub_doc['content'] = segment
                sub_chunks = self._token_based_chunking(sub_doc, chunk_size, chunk_overlap)
                refined_segments.extend([chunk['content'] for chunk in sub_chunks])
            else:
                refined_segments.append(segment)
        
        # Create chunks from segments
        chunks = []
        for i, segment in enumerate(refined_segments, 1):
            if segment.strip():
                chunk = self._create_chunk(document, segment.strip(), 0, 0, i, 0)
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_headings(self, content: str) -> List[str]:
        """Split markdown/text content by headings."""
        segments = []
        lines = content.split('\n')
        current_segment = ""
        
        for line in lines:
            # Check for markdown headings
            if re.match(r'^#+\s+', line.strip()):
                if current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
            
            current_segment += line + '\n'
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments if segments else [content]
    
    def _get_chunk_overlap(self, text: str, overlap_tokens: int) -> str:
        """
        Extract overlapping portion from the end of text.
        
        Args:
            text: Text to extract overlap from
            overlap_tokens: Number of tokens for overlap
            
        Returns:
            Overlap text
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        overlap_tokens = tokens[-overlap_tokens:]
        return self.encoding.decode(overlap_tokens)
    
    def _create_chunk(self, 
                     original_doc: Dict[str, Any],
                     content: str,
                     start_idx: int,
                     end_idx: int,
                     chunk_number: int,
                     total_tokens: int) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata.
        
        Args:
            original_doc: Original document metadata
            content: Chunk content
            start_idx: Start token index (if applicable)
            end_idx: End token index (if applicable)
            chunk_number: Chunk sequence number
            total_tokens: Total tokens in original document
            
        Returns:
            Chunk dictionary with metadata
        """
        chunk_tokens = len(self.encoding.encode(content))
        
        return {
            'content': content,
            'chunk_tokens': chunk_tokens,
            'chunk_number': chunk_number,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_document_tokens': total_tokens,
            
            # Preserve original document metadata
            'file_path': original_doc.get('file_path'),
            'absolute_path': original_doc.get('absolute_path'),
            'file_name': original_doc.get('file_name'),
            'file_extension': original_doc.get('file_extension'),
            'repo_root': original_doc.get('repo_root'),
            'source_type': original_doc.get('source_type'),
            
            # Chunk metadata
            'chunk_id': f"{original_doc.get('file_path', 'unknown')}_chunk_{chunk_number}",
        }
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the current encoding.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))