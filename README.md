# Knowledge Graph Pipeline

A powerful, modular pipeline for extracting structured information (knowledge graphs) from unstructured text using Large Language Models (LLMs).

## Overview

This project provides a flexible and extensible pipeline for converting unstructured text into knowledge graphs by extracting subject-predicate-object triples or JSON-LD structured data using state-of-the-art LLMs. The architecture is designed for maintainability, extensibility, and ease of testing.

## Architecture

### 1. Main Pipeline (`src/pipeline.py`)
- Core `KnowledgeGraphPipeline` class that orchestrates the entire process
- Uses composition over inheritance for better modularity
- Supports multiple LLM providers (OpenAI and Anthropic)
- Supports both triple extraction and JSON-LD extraction modes
- Processing steps:
  1. Text chunking
  2. Information extraction (triples or JSON-LD)
  3. Validation and normalization
  4. Deduplication and statistics generation

### 2. Prompt Management
- **Centralized Prompts**: System and user prompts are defined in `src/config/settings.py`
- **Dynamic Selection**: Prompts are selected based on extraction mode (triples vs JSON-LD)
- **Parameter Passing**: Prompts are passed as constructor parameters to LLM clients
- **Flexible Configuration**: Different prompts can be used for different use cases

### 3. Modular Components

#### Configuration Management (`src/config/`)
- `configuration.py`: Centralized configuration with validation
- `settings.py`: Default settings and prompt templates
- Environment-based configuration loading

#### LLM Clients (`src/models/`)
- `base_llm_client.py`: Abstract base class for LLM clients
- `openai_client.py`: OpenAI API client implementation
- `anthropic_client.py`: Anthropic API client implementation
- `response_parsers.py`: Modular response parsing for different extraction modes

#### Extractors (`src/extractors/`)
- `base_extractor.py`: Abstract base class for extractors
- `triple_extractor.py`: Subject-predicate-object triple extraction
- `jsonld_extractor.py`: JSON-LD structured data extraction
- `extractor_factory.py`: Factory for creating appropriate extractors

#### Processors (`src/processors/`)
- `text_processor.py`: Text chunking and processing utilities
- `ontology_processor.py`: Ontology handling and JSON-LD validation

#### Utilities (`src/utils/`)
- `logger.py`: Centralized logging system
- `display_manager.py`: Result display and formatting

#### Storage (`src/storage/`)
- `jsonld_graphdb_storage.py`: GraphDB storage backend
- Extensible storage abstraction for different backends

### 4. Dependencies
- `openai` and `anthropic`: LLM API access
- `pandas`: Data manipulation and display
- `owlready2` and `PyLD`: Ontology and JSON-LD processing
- `rdflib`: RDF graph operations
- `pymupdf4llm`: PDF text extraction
- `python-dotenv`: Environment variable management

## Features

- **Modular Architecture**: Clean separation of concerns with pluggable components
- **Multiple Extraction Modes**: Support for both triple extraction and JSON-LD extraction
- **Ontology Support**: JSON-LD extraction with ontology validation
- **Configurable LLM Provider**: OpenAI or Anthropic with easy extension
- **Text Processing**: Intelligent chunking for handling long documents
- **Validation & Normalization**: Built-in data validation and normalization
- **Error Handling**: Comprehensive error handling and logging
- **Storage Backends**: Extensible storage system (GraphDB support included)
- **Detailed Statistics**: Processing statistics and performance metrics

## Usage

### Basic Triple Extraction

```python
from src.pipeline import KnowledgeGraphPipeline
from src.config.configuration import Configuration

# Load configuration from environment
config = Configuration.from_env()

# Initialize the pipeline
pipeline = KnowledgeGraphPipeline(config)

# Process text
success, result, error = pipeline.process_text(your_text)

# Display results
if success:
    pipeline.display_results(result)
    pipeline.display_summary(result)
else:
    print(f"Error processing text: {error}")
```

### JSON-LD Extraction with Ontology

```python
from src.pipeline import KnowledgeGraphPipeline
from src.config.configuration import Configuration

# Create configuration for JSON-LD extraction
config = Configuration.from_env()
config.extraction.extraction_mode = "jsonld"
config.extraction.ontology_path = "path/to/your/ontology.owl"

# Initialize the pipeline
pipeline = KnowledgeGraphPipeline(config)

# Process text
success, result, error = pipeline.process_text(your_text)

# Display results
if success:
    pipeline.display_results(result)
    pipeline.display_summary(result)
else:
    print(f"Error processing text: {error}")
```

### PDF Processing

```python
from src.pipeline import KnowledgeGraphPipeline
from src.config.configuration import Configuration

# Load configuration
config = Configuration.from_env()

# Initialize the pipeline
pipeline = KnowledgeGraphPipeline(config)

# Process PDF
success, result, error = pipeline.process_pdf("path/to/document.pdf")

# Display results
if success:
    pipeline.display_results(result)
    pipeline.display_summary(result)
else:
    print(f"Error processing PDF: {error}")
```

### Evaluation System

The `diff_example.py` demonstrates the evaluation system that compares different LLM providers and configurations. It uses direct prompt imports from settings to enable flexible configuration testing and comparison.

### Triple Extraction Example

The `triple_extraction_example.py` is a versatile example that can process both text files and PDFs:

```bash
# Process a specific text file
python examples/triple_extraction_example.py examples/Marie_Curie.txt

# Process a specific PDF file
python examples/triple_extraction_example.py examples/Marie_Curie.pdf

# Process both default files (Marie_Curie.txt and Marie_Curie.pdf)
python examples/triple_extraction_example.py --default

# Show help
python examples/triple_extraction_example.py --help
```

The example automatically detects file types based on extensions (.txt, .md, .pdf) and uses the appropriate processing method.

## Environment Setup

1. Create a `.env` file with your API keys and configuration:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
LLM_PROVIDER=openai  # or anthropic
LLM_MODEL_NAME=gpt-4-turbo  # or claude-3-7-sonnet-20250219
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096

# Text Processing Configuration
CHUNK_SIZE=2000
CHUNK_OVERLAP=100

# Extraction Configuration
EXTRACTION_MODE=triples  # or jsonld
ONTOLOGY_PATH=path/to/ontology.owl  # required for jsonld mode
ENABLE_VALIDATION=true
ENABLE_NORMALIZATION=true

# Logging Configuration
ENABLE_LOGGING=true
LOG_LEVEL=INFO

# Optional: GraphDB Configuration
GRAPHDB_REPO_ID=your_repo_id
GRAPHDB_BASE_URL=http://localhost:7200
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Examples

The project includes several examples demonstrating different use cases:

- `examples/triple_extraction_example.py`: Combined text and PDF processing with triple extraction
- `examples/jsonld_extraction_example.py`: JSON-LD extraction with ontology from Marie Curie biography
- `examples/upload_jsonld_example.py`: Uploading results to GraphDB
- `examples/diff_example.py`: LLM provider comparison and evaluation

## Extending the Pipeline

### Adding a New LLM Provider

1. Create a new client class inheriting from `BaseLLMClient`
2. Implement the required abstract methods
3. Update the pipeline initialization to support the new provider

### Adding a New Extraction Mode

1. Create a new extractor class inheriting from `BaseExtractor`
2. Implement the required abstract methods
3. Update the `ExtractorFactory` to support the new mode
4. Add configuration options if needed

### Adding a New Storage Backend

1. Create a new storage class with upload/query methods
2. Implement the storage interface
3. Update the storage factory (if implemented)

## Configuration Options

### LLM Configuration
- `provider`: LLM provider ("openai" or "anthropic")
- `model_name`: Model to use
- `temperature`: Sampling temperature (0.0-2.0)
- `max_tokens`: Maximum tokens for responses

### Text Processing Configuration
- `chunk_size`: Maximum words per chunk
- `chunk_overlap`: Words to overlap between chunks

### Extraction Configuration
- `extraction_mode`: "triples" or "jsonld"
- `ontology_path`: Path to OWL ontology file (required for JSON-LD)
- `enable_validation`: Enable data validation
- `enable_normalization`: Enable data normalization

## Contributing

The modular architecture makes it easy to contribute new features:

1. Follow the existing patterns for new components
2. Use the centralized logging system
3. Add appropriate configuration options
4. Include tests for new functionality
5. Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
