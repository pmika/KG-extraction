import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.pipeline import KnowledgeGraphPipeline
from src.config.configuration import Configuration

def process_file(file_path: str, pipeline: KnowledgeGraphPipeline):
    """
    Process a file (text or PDF) using the knowledge graph pipeline.
    
    Args:
        file_path: Path to the file to process
        pipeline: Initialized pipeline instance
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return False
    
    # Determine file type based on extension
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pdf':
        print(f"\n--- Processing PDF File: {file_path} ---")
        success, result, error = pipeline.process_pdf(file_path)
        
        if success:
            pipeline.display_results(result)
            pipeline.display_summary(result)
            return True
        else:
            print(f"Error processing PDF: {error}")
            return False
            
    elif file_extension in ['.txt', '.md']:
        print(f"\n--- Processing Text File: {file_path} ---")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            print(f"Text length: {len(text_content)} characters")
            
            success, result, error = pipeline.process_text(text_content)
            
            if success:
                pipeline.display_results(result)
                pipeline.display_summary(result)
                return True
            else:
                print(f"Error processing text: {error}")
                return False
                
        except FileNotFoundError:
            print(f"Error: Text file {file_path} not found")
            return False
        except Exception as e:
            print(f"Error reading text file: {e}")
            return False
    else:
        print(f"Error: Unsupported file type '{file_extension}'. Supported types: .pdf, .txt, .md")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract knowledge graph triples from text files or PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python triple_extraction_example.py examples/Marie_Curie.txt
  python triple_extraction_example.py examples/Marie_Curie.pdf
        """
    )
    parser.add_argument(
        'file_path', 
        nargs='?', 
        help='Path to the file to process (.txt, .md, or .pdf)'
    )
    
    args = parser.parse_args()
    
    # Check if any arguments were provided
    if not args.file_path:
        # No arguments provided, show error and help
        print("\n" + "="*60)
        print("❌ ERROR: No arguments provided")
        print("="*60)
        print("You must specify a file path.")
        print("\nUsage options:")
        print("  • Provide a file path: python triple_extraction_example.py <file_path>")
        print("  • Show help: python triple_extraction_example.py --help")
        print("\n" + "="*60)
        print("HELP")
        print("="*60)
        parser.print_help()
        sys.exit(1)
    
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
    
    # Initialize the pipeline
    config = Configuration.from_env()
    pipeline = KnowledgeGraphPipeline(config)
    pipeline.display_configuration()
    
    if args.file_path:
        # Process the specified file
        success = process_file(args.file_path, pipeline)
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main() 