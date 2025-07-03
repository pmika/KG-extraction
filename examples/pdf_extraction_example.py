import os
from pathlib import Path
from dotenv import load_dotenv
from src.pipeline import KnowledgeGraphPipeline
from src.processors.text_processor import TextProcessor

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
    
    # Example PDF path - replace with your PDF file
    pdf_path = Path("Marie Curie.pdf")
    
    if not pdf_path.exists():
        print(f"\nError: PDF file not found at {pdf_path}")
        print("Please place your PDF file in the root directory and update the pdf_path variable.")
        return
    
    print(f"\nProcessing PDF: {pdf_path}")
    
    try:
        # Initialize the text processor and pipeline
        text_processor = TextProcessor()
        pipeline = KnowledgeGraphPipeline()
        
        # Extract and process text from PDF
        print("\nExtracting text from PDF...")
        chunks = text_processor.process_pdf(pdf_path, pages=[0])
        
        # Combine chunks into a single text for processing
        combined_text = " ".join(chunk["text"] for chunk in chunks)
        
        print(f"Successfully extracted {len(chunks)} chunks from the PDF")
        print("\nProcessing text through knowledge graph pipeline...")
        
        # Process the extracted text through the pipeline
        success, result, error = pipeline.process_text(combined_text)
        
        if success:
            pipeline.display_results(result)
        else:
            print(f"Error processing text: {error}")
            
    except Exception as e:
        print(f"\nError processing PDF: {str(e)}")
        print("Make sure the PDF file is not corrupted and is readable.")

if __name__ == "__main__":
    main() 