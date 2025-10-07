import tiktoken
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class Chunker:
    """
    Handles intelligent chunking of code and text documents.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
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
        content = document.get('content', '')
        file_extension = document.get('file_extension', '').lower()
        
        if not content.strip():
            return []
        
        if strategy == "code_aware" and self._is_code_file(file_extension):
            return self._code_aware_chunking(document, chunk_size, chunk_overlap)
        elif strategy == "semantic":
            return self._semantic_chunking(document, chunk_size, chunk_overlap)
        else:
            return self._token_based_chunking(document, chunk_size, chunk_overlap)
    
    def _is_code_file(self, file_extension: str) -> bool:
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
        content = document['content']
        tokens = self.encoding.encode(content, disallowed_special=())
        
        if len(tokens) <= chunk_size:
            return [self._create_chunk(document, content, 0, len(content), 1, 1)]
        
        chunks = []
        start_idx = 0
        chunk_id = 1
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunk = self._create_chunk(document, chunk_text, start_idx, end_idx, chunk_id, len(tokens))
            chunks.append(chunk)
            
            start_idx += chunk_size - chunk_overlap
            chunk_id += 1
            
            if start_idx >= len(tokens):
                break
        
        return chunks
    
    def _code_aware_chunking(self,
                           document: Dict[str, Any],
                           chunk_size: int,
                           chunk_overlap: int) -> List[Dict[str, Any]]:
        content = document['content']
        file_extension = document.get('file_extension', '').lower()
        
        segments = self._split_code_by_boundaries(content, file_extension)
        
        if not segments or max(len(seg) for seg in segments) > chunk_size * 4:
            return self._token_based_chunking(document, chunk_size, chunk_overlap)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 1
        
        for segment in segments:
            segment_tokens = len(self.encoding.encode(segment))
            
            if current_tokens + segment_tokens > chunk_size and current_chunk:
                chunk = self._create_chunk(document, current_chunk.strip(), 0, 0, chunk_id, 0)
                chunks.append(chunk)
                
                if chunk_overlap > 0 and chunks:
                    overlap_text = self._get_chunk_overlap(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + segment
                    current_tokens = len(self.encoding.encode(current_chunk))
                else:
                    current_chunk = segment
                    current_tokens = segment_tokens
                
                chunk_id += 1
            else:
                current_chunk += segment
                current_tokens += segment_tokens
        
        if current_chunk.strip():
            chunk = self._create_chunk(document, current_chunk.strip(), 0, 0, chunk_id, 0)
            chunks.append(chunk)
        
        return chunks
    
    def _split_code_by_boundaries(self, content: str, file_extension: str) -> List[str]:
        segments = []
        
        if file_extension == '.py':
            segments = self._split_python_code(content)
        elif file_extension in ['.js', '.jsx', '.ts', '.tsx']:
            segments = self._split_javascript_code(content)
        elif file_extension in ['.java', '.cpp', '.c', '.cs']:
            segments = self._split_clike_code(content)
        else:
            segments = self._split_by_paragraphs(content)
        
        return [seg for seg in segments if seg.strip()]
    
    def _split_python_code(self, content: str) -> List[str]:
        segments = []
        current_segment = ""
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if (stripped.startswith(('def ', 'class ', '@')) or 
                (stripped.startswith(('import ', 'from ')) and ' import ' in stripped)):
                if current_segment.strip():
                    segments.append(current_segment)
                    current_segment = ""
            
            current_segment += line + '\n'
            i += 1
        
        if current_segment.strip():
            segments.append(current_segment)
        
        return segments if segments else [content]
    
    def _split_javascript_code(self, content: str) -> List[str]:
        patterns = [
            r'(export\s+)?(default\s+)?(class|function|const|let|var)\s+\w+',
            r'export\s+default',
            r'import\s+.*?from',
        ]
        for pattern in patterns:
            splits = re.split(f'({pattern})', content)
            # Фильтруем None элементы
            splits = [s for s in splits if s is not None]
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
    
    def _split_clike_code(self, content: str) -> List[str]:
        patterns = [
            r'(public|private|protected)?\s*(class|interface|struct)\s+\w+',
            r'(public|private|protected)?\s*\w+\s+\w+\s*\(',
            r'#include',
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
        return [p for p in re.split(r'\n\s*\n', content) if p.strip()]
    
    def _semantic_chunking(self,
                          document: Dict[str, Any],
                          chunk_size: int,
                          chunk_overlap: int) -> List[Dict[str, Any]]:
        content = document['content']
        file_extension = document.get('file_extension', '').lower()
        
        if file_extension in ['.md', '.rst', '.txt']:
            segments = self._split_by_headings(content)
        else:
            segments = self._split_by_paragraphs(content)
        
        refined_segments = []
        for segment in segments:
            segment_tokens = len(self.encoding.encode(segment))
            if segment_tokens > chunk_size:
                sub_doc = document.copy()
                sub_doc['content'] = segment
                sub_chunks = self._token_based_chunking(sub_doc, chunk_size, chunk_overlap)
                refined_segments.extend([chunk['content'] for chunk in sub_chunks])
            else:
                refined_segments.append(segment)
        
        chunks = []
        for i, segment in enumerate(refined_segments, 1):
            if segment.strip():
                chunk = self._create_chunk(document, segment.strip(), 0, 0, i, 0)
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_headings(self, content: str) -> List[str]:
        segments = []
        lines = content.split('\n')
        current_segment = ""
        
        for line in lines:
            if re.match(r'^#+\s+', line.strip()):
                if current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
            
            current_segment += line + '\n'
        
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments if segments else [content]
    
    def _get_chunk_overlap(self, text: str, overlap_tokens: int) -> str:
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
        chunk_tokens = len(self.encoding.encode(content, disallowed_special=()))
        
        return {
            'content': content,
            'chunk_tokens': chunk_tokens,
            'chunk_number': chunk_number,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_document_tokens': total_tokens,
            'file_path': original_doc.get('file_path'),
            'absolute_path': original_doc.get('absolute_path'),
            'file_name': original_doc.get('file_name'),
            'file_extension': original_doc.get('file_extension'),
            'repo_root': original_doc.get('repo_root'),
            'source_type': original_doc.get('source_type'),
            'chunk_id': f"{original_doc.get('file_path', 'unknown')}_chunk_{chunk_number}",
        }
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))