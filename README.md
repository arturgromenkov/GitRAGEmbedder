# GitRAG Embedder: Git Repository to Embeddings Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![RAG](https://img.shields.io/badge/RAG-Embeddings-green)
![Git](https://img.shields.io/badge/Git-Repository%20Processing-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Overview

**GitRAG Embedder** is a focused library that converts Git repositories into embedding vectors. Pure processing pipeline - from repository cloning to embedding generation.

## ✨ Features

### 🔍 Repository Processing
- **Git Integration**: Clone and process any Git repository
- **Multi-Format Support**: Process `.py`, `.md`, `.rst`, `.txt` files
- **Smart Chunking**: Configurable text splitting with overlap
- **Selective Processing**: Filter by file type and directory
- **Batch Processing**: Efficient handling of large codebases

### 📊 Multiple Embedding Backends
- **OpenAI Embeddings**: Support for text-embedding-ada-002, text-embedding-3-small/large
- **Sentence Transformers**: Local models like all-MiniLM-L6-v2
- **HuggingFace Transformers**: Custom transformer models
- **Batch Processing**: Efficient batch embedding generation
- **Automatic Retries**: Built-in error handling and retries

## 📦 Installation

```bash
pip install git-rag-embedder
```

**Optional dependencies** (install as needed):
```bash
# For OpenAI embeddings
pip install openai

# For Sentence Transformers
pip install sentence-transformers

# For HuggingFace embeddings  
pip install transformers torch
```

## 🚀 Quick Start

### Basic Usage

```python
from git_rag_embedder import GitRAGEmbedder

# Initialize with default settings (Sentence Transformers)
embedder = GitRAGEmbedder()

# Process a repository into embeddings
embeddings = embedder.process_repository(
    "https://github.com/username/repository"
)

print(f"Generated {len(embeddings)} embedding vectors")
```

### Using Different Backends

```python
# OpenAI embeddings
embedder = GitRAGEmbedder(
    embedding_backend="openai",
    model="text-embedding-3-small",
    api_key="your-openai-key"
)

# Sentence Transformers
embedder = GitRAGEmbedder(
    embedding_backend="sentence_transformers", 
    model_name="all-MiniLM-L6-v2"
)

# HuggingFace Transformers
embedder = GitRAGEmbedder(
    embedding_backend="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Advanced Configuration

```python
embeddings = embedder.process_repository(
    repo_url="https://github.com/username/repository",
    chunk_size=1000,
    chunk_overlap=150,
    extensions={'.py', '.md', '.txt'},
    exclude_dirs={'tests', 'docs', 'node_modules'},
    batch_size=32,
    max_files=1000
)
```

## ⚙️ Configuration

### Processing Parameters

```python
# All configuration options
embeddings = embedder.process_repository(
    repo_url="https://github.com/user/repo",  # Git URL or local path
    chunk_size=1000,           # Characters per chunk
    chunk_overlap=150,         # Overlap between chunks  
    extensions={'.py', '.md'}, # File types to process
    exclude_dirs={'tests'},    # Directories to skip
    batch_size=32,             # Processing batch size
    max_files=1000            # Maximum files to process
)
```

### Available Embedding Models

**OpenAI Models:**
- `text-embedding-ada-002` (1536 dim)
- `text-embedding-3-small` (1536 dim) 
- `text-embedding-3-large` (3072 dim)

**Sentence Transformers Models:**
- `all-MiniLM-L6-v2` (384 dim)
- `all-mpnet-base-v2` (768 dim)
- `multi-qa-mpnet-base-dot-v1` (768 dim)

**HuggingFace Models:**
- Any sentence transformer compatible model

## 📊 Output Format

### Embedding Structure

Each embedding contains:

```python
{
    'content': 'def calculate_sum(a, b):\n    return a + b',
    'file_path': 'src/math_utils.py', 
    'file_extension': '.py',
    'embedding': [0.123, -0.456, 0.789, ...],  # Vector array
    'embedding_dimension': 384,
    'embedding_model': 'sentence_transformers',
    'embedding_norm': 1.234,  # L2 norm of the vector
    'metadata': {
        'chunk_size': 245,
        'token_count': 45
    }
}
```

## 🔧 API Reference

### GitRAGEmbedder Class

```python
class GitRAGEmbedder:
    def __init__(
        self,
        embedding_backend: str = "sentence_transformers",
        **backend_kwargs
    )
    
    def process_repository(
        self,
        repo_url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        extensions: Set[str] = None,
        exclude_dirs: Set[str] = None,
        batch_size: int = 32,
        max_files: int = 1000
    ) -> List[Dict[str, Any]]
```

### EmbeddingGenerator Class

```python
class EmbeddingGenerator:
    def __init__(self, backend: str = "sentence_transformers", **backend_kwargs)
    
    def generate_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]
    
    def embed_single_text(self, text: str) -> List[float]
    def get_embedding_dimension(self) -> int
```

### Available Backends

- `OpenAIEmbeddingBackend` - For OpenAI embedding API
- `SentenceTransformersBackend` - For local sentence transformer models  
- `HuggingFaceEmbeddingBackend` - For HuggingFace transformer models

## 🏗️ Advanced Usage

### Custom Processing Pipeline

```python
from git_rag_embedder import GitRAGEmbedder, EmbeddingGenerator

# Step-by-step processing
embedder = GitRAGEmbedder()

# Extract and chunk documents
documents = embedder.extract_documents("https://github.com/user/repo")

# Use different embedding backend for generation
embedding_gen = EmbeddingGenerator(
    backend="openai",
    model="text-embedding-3-small",
    api_key="your-key"
)

embeddings = embedding_gen.generate_embeddings(documents)
```

### Multiple Repositories

```python
repositories = [
    "https://github.com/org/repo1",
    "https://github.com/org/repo2", 
    "/path/to/local/repo"
]

all_embeddings = []
for repo in repositories:
    embeddings = embedder.process_repository(repo)
    all_embeddings.extend(embeddings)
```

### Save/Load Embeddings

```python
# Save for later use
embedder.save_embeddings(embeddings, "my_embeddings.json")

# Load saved embeddings
loaded_embeddings = embedder.load_embeddings("my_embeddings.json")
```

## 🔍 Backend Details

### OpenAI Backend
```python
backend = OpenAIEmbeddingBackend(
    api_key="your-key",  # Optional, uses OPENAI_API_KEY env var
    model="text-embedding-3-small"
)
```

### Sentence Transformers Backend
```python
backend = SentenceTransformersBackend(
    model_name="all-MiniLM-L6-v2"  # Any sentence-transformers model
)
```

### HuggingFace Backend
```python
backend = HuggingFaceEmbeddingBackend(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Any HF model
)
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/git-rag-embedder/issues)

---

**Focus**: Pure Git repository to embedding conversion pipeline. No search, no quality metrics, just embeddings.

---

*Star this repository if you find it helpful! ⭐*
