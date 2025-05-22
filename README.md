# Knowledge Graph Pipeline

A powerful pipeline for extracting structured information (knowledge graphs) from unstructured text using Large Language Models (LLMs).

## Overview

This project provides a flexible and extensible pipeline for converting unstructured text into knowledge graphs by extracting subject-predicate-object triples using state-of-the-art LLMs.

## Components

### 1. Main Pipeline (`src/pipeline.py`)
- Core `KnowledgeGraphPipeline` class that orchestrates the entire process
- Supports multiple LLM providers (OpenAI and Anthropic)
- Processing steps:
  1. Text chunking
  2. Triple extraction
  3. Normalization and deduplication
  4. Statistics and results generation

### 2. Project Structure
- `src/models/`: LLM client implementations (OpenAI and Anthropic)
- `src/processors/`: Text processing utilities
- `src/config/`: Configuration settings
- `src/utils/`: Utility functions

### 3. Dependencies
- `openai` and `anthropic`: LLM API access
- `pandas`: Data manipulation and display
- `networkx` and `ipycytoscape`: Graph visualization
- `ipywidgets`: Interactive widgets
- `python-dotenv`: Environment variable management

## Features

- Configurable LLM provider (OpenAI or Anthropic)
- Text chunking for handling long documents
- Triple extraction and normalization
- Error handling and reporting
- Detailed statistics and visualization capabilities

## Usage

See `example.py` for a complete usage example. The pipeline can be used as follows:

```python
from src.pipeline import KnowledgeGraphPipeline

# Initialize the pipeline
pipeline = KnowledgeGraphPipeline()

# Process text
success, result, error = pipeline.process_text(your_text)

# Display results
if success:
    pipeline.display_results(result)
else:
    print(f"Error processing text: {error}")
```

## Environment Setup

1. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LLM_PROVIDER=openai  # or anthropic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Example

The example in `example.py` demonstrates the pipeline's capabilities using a Marie Curie biography, showing how it can extract structured information and create a knowledge graph of relationships and facts.
