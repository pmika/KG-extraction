import pandas as pd
from src.models.openai_client import OpenAIClient
from src.models.anthropic_client import AnthropicClient
from src.processors.text_processor import TextProcessor
from src.config.settings import LLM_PROVIDER

class KnowledgeGraphPipeline:
    def __init__(self):
        """
        Initialize the knowledge graph extraction pipeline.
        """
        # Initialize the appropriate LLM client based on the provider
        if LLM_PROVIDER == "openai":
            self.llm_client = OpenAIClient()
        elif LLM_PROVIDER == "anthropic":
            self.llm_client = AnthropicClient()
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
            
        self.text_processor = TextProcessor()
        
    def process_text(self, text):
        """
        Process text through the entire pipeline.
        
        Args:
            text (str): The input text to process
            
        Returns:
            tuple: (success, result, error_message)
            - success (bool): Whether the processing was successful
            - result (dict): Dictionary containing processed data and statistics
            - error_message (str): Error message if unsuccessful
        """
        try:
            # 1. Split text into chunks
            chunks = self.text_processor.split_into_chunks(text)
            if not chunks:
                return False, None, "No chunks were created from the input text"
                
            # 2. Process each chunk
            all_extracted_triples = []
            failed_chunks = []
            
            for chunk in chunks:
                success, triples, error = self.llm_client.extract_triples(
                    chunk['text'],
                    chunk['chunk_number']
                )
                
                if success:
                    all_extracted_triples.extend(triples)
                else:
                    failed_chunks.append({
                        'chunk_number': chunk['chunk_number'],
                        'error': error
                    })
                    
            # 3. Normalize and deduplicate triples
            normalized_triples = self.text_processor.deduplicate_triples(all_extracted_triples)
            
            # 4. Prepare results
            result = {
                'triples': normalized_triples,
                'statistics': {
                    'total_chunks': len(chunks),
                    'processed_chunks': len(chunks) - len(failed_chunks),
                    'failed_chunks': len(failed_chunks),
                    'total_triples': len(all_extracted_triples),
                    'unique_triples': len(normalized_triples)
                },
                'failed_chunks': failed_chunks
            }
            
            return True, result, None
            
        except Exception as e:
            return False, None, f"Pipeline error: {str(e)}"
            
    def display_results(self, result):
        """
        Display the processing results in a readable format.
        
        Args:
            result (dict): The result dictionary from process_text
        """
        if not result:
            print("No results to display")
            return
            
        # Display statistics
        stats = result['statistics']
        print("\n--- Processing Statistics ---")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Successfully processed chunks: {stats['processed_chunks']}")
        print(f"Failed chunks: {stats['failed_chunks']}")
        print(f"Total triples extracted: {stats['total_triples']}")
        print(f"Unique triples after normalization: {stats['unique_triples']}")
        
        # Display failed chunks if any
        if result['failed_chunks']:
            print("\n--- Failed Chunks ---")
            for failure in result['failed_chunks']:
                print(f"Chunk {failure['chunk_number']}: {failure['error']}")
                
        # Display triples
        if result['triples']:
            print("\n--- Extracted Triples ---")
            df = pd.DataFrame(result['triples'])
            print(df.to_string(index=False)) 