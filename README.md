# RAG Document Query System with LlamaIndex

A Retrieval-Augmented Generation (RAG) system that uses Claude Sonnet 4.5 and multilingual embeddings to query documents intelligently.

## Overview

This project implements a document question-answering system using:
- **LLM**: Claude Sonnet 4.5 (via Anthropic API)
- **Embeddings**: Multilingual E5-Large model
- **Framework**: LlamaIndex for orchestration
- **Storage**: Persistent vector index storage

## Features

-  Load and index documents from a local directory
-  Semantic search with multilingual support
-  Persistent index storage for faster subsequent runs
-  Claude-powered responses with context from your documents
-  Configurable similarity search (top-k retrieval)

## Prerequisites

- Python 3.8+
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install required packages:
```bash
pip install llama-index llama-index-llms-anthropic llama-index-embeddings-huggingface torch
```

## Configuration

1. **API Key**: Replace the placeholder API key in the code:
```python
Settings.llm = Anthropic(
    api_key="your-anthropic-api-key-here",  # Replace this
    model="claude-sonnet-4-5-20250929",
    temperature=0.3,
    max_tokens=1024
)
```

2. **Document Directory**: Place your documents in a `data/` folder in the project root
   - Supported formats: TXT, PDF, DOCX, and other formats supported by LlamaIndex

3. **Storage Path**: The vector index is saved to `./storage` by default

## Usage

Run the script:
```bash
python main.py
```

### How It Works

1. **Setup Phase**: Initializes Claude LLM and E5-Large embedding model
2. **Index Loading/Creation**: 
   - Checks for existing index in `./storage`
   - If found, loads from disk
   - If not found, creates new index from documents in `data/`
3. **Query Engine Creation**: Sets up retrieval system with top-3 similarity search
4. **Query Processing**: Sends query and receives context-aware response

### Example Query

The default query in the code:
```python
query = "Verim, Suç ve Ceza kitabına göre suç türleri nelerdir?"
```

Modify this line to ask your own questions about your documents.

## Project Structure
```
.
├── main.py              # Main script
├── data/                # Place your documents here
├── storage/             # Persistent vector index (auto-generated)
└── README.md           # This file
```

## Configuration Options

### LLM Settings
- **Model**: `claude-sonnet-4-5-20250929`
- **Temperature**: `0.3` (lower = more focused, higher = more creative)
- **Max Tokens**: `1024` (maximum response length)

### Query Engine Settings
- **similarity_top_k**: `3` (number of relevant document chunks to retrieve)
- **response_mode**: `compact` (concise responses)

Available response modes:
- `compact`: Concise answers
- `tree_summarize`: Hierarchical summarization
- `simple_summarize`: Simple concatenation and summarization

## Logging

The script includes detailed logging to track:
- Setup progress
- Document loading status
- Index creation/loading
- Query processing

## Error Handling

The system handles:
- Missing `data/` directory
- Empty document directory
- Index loading failures
- Automatic fallback to index creation

## Performance Notes

- **First Run**: Slower (creates embeddings for all documents)
- **Subsequent Runs**: Faster (loads pre-computed index from disk)
- **Index Storage**: Persists in `./storage` directory

## Multilingual Support

The E5-Large embedding model supports multiple languages, making this system suitable for:
- English documents
- Turkish documents (as shown in example)
- Many other languages

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Anthropic](https://www.anthropic.com/) for Claude API
- [HuggingFace](https://huggingface.co/) for E5-Large embeddings
