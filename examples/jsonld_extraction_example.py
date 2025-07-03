import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.pipeline import KnowledgeGraphPipeline
from src.config.configuration import Configuration

def main():
    # Print current working directory and .env file existence
    print("\n--- Debug Information ---")
    print(f"Current working directory: {os.getcwd()}")
    print(f".env file exists: {os.path.exists('.env')}")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Display loaded environment variables with more detail
    print("\n--- Environment Variables Loaded ---")
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    llm_provider = os.getenv('LLM_PROVIDER', 'openai')
    print(f"LLM Provider: {llm_provider}")
    print(f"OPENAI_API_KEY: {openai_key if openai_key else 'Not Set'}")
    print(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE', 'Not Set')}")
    print(f"ANTHROPIC_API_KEY: {anthropic_key if anthropic_key else 'Not Set'}")
    print("-" * 40)
    
    # Path to the text file
    text_path = "examples/Marie_Curie.txt"

    # Path to the ontology file
    ontology_path = "examples/scientist_ontology.owl"

    # Set up configuration for JSON-LD extraction
    config = Configuration.from_env()
    config.extraction.extraction_mode = "jsonld"
    config.extraction.ontology_path = ontology_path

    pipeline = KnowledgeGraphPipeline(config)
    pipeline.display_configuration()
    
    # Read the text file
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"\n--- Processing Text File: {text_path} ---")
        print(f"Text length: {len(text)} characters")
        
        success, result, error = pipeline.process_text(text)

        if success:
            pipeline.display_results(result)
            pipeline.display_summary(result)
        else:
            print(f"Error processing text: {error}")
            
    except FileNotFoundError:
        print(f"Error: Text file {text_path} not found")
    except Exception as e:
        print(f"Error reading text file: {e}")

if __name__ == "__main__":
    main() 