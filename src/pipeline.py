import pandas as pd
from src.models.openai_client import OpenAIClient
from src.models.anthropic_client import AnthropicClient
from src.processors.text_processor import TextProcessor
from src.config.settings import (
    LLM_PROVIDER, 
    LLM_MODEL_NAME, 
    LLM_TEMPERATURE, 
    LLM_MAX_TOKENS,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE
)

class KnowledgeGraphPipeline:
    def __init__(
        self, 
        llm_provider: str = None, 
        model_name: str = None, 
        temperature: float = None, 
        max_tokens: int = None,
        system_prompt: str = None,
        user_prompt_template: str = None
    ):
        """
        Initialize the knowledge graph extraction pipeline.
        
        Args:
            llm_provider: Optional LLM provider to use. If not provided, uses the global setting.
            model_name: Optional model name to use. If not provided, uses the global setting.
            temperature: Optional temperature to use. If not provided, uses the global setting.
            max_tokens: Optional maximum tokens to use. If not provided, uses the global setting.
            system_prompt: Optional system prompt to use. If not provided, uses the global setting.
            user_prompt_template: Optional user prompt template to use. If not provided, uses the global setting.
        """
        # Initialize the appropriate LLM client based on the provider
        provider = llm_provider if llm_provider is not None else LLM_PROVIDER
        model = model_name if model_name is not None else LLM_MODEL_NAME
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        sys_prompt = system_prompt if system_prompt is not None else EXTRACTION_SYSTEM_PROMPT
        usr_prompt = user_prompt_template if user_prompt_template is not None else EXTRACTION_USER_PROMPT_TEMPLATE
        
        print(f"\nCreating pipeline with provider: {provider}")
        print(f"Model name: {model}")
        print(f"Temperature: {temp}")
        print(f"Max tokens: {tokens}")
        print(f"System prompt length: {len(sys_prompt)}")
        print(f"User prompt template length: {len(usr_prompt)}")
        
        if provider == "openai":
            print("Initializing OpenAI client...")
            self.llm_client = OpenAIClient(
                model_name=model, 
                temperature=temp, 
                max_tokens=tokens,
                system_prompt=sys_prompt,
                user_prompt_template=usr_prompt
            )
        elif provider == "anthropic":
            print("Initializing Anthropic client...")
            self.llm_client = AnthropicClient(
                model_name=model, 
                temperature=temp, 
                max_tokens=tokens,
                system_prompt=sys_prompt,
                user_prompt_template=usr_prompt
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
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
                
            print(f"\nProcessing {len(chunks)} chunks...")
                
            # 2. Process each chunk
            all_extracted_triples = []
            failed_chunks = []
            
            for chunk in chunks:
                print(f"\nProcessing chunk {chunk['chunk_number']}...")
                success, triples, error = self.llm_client.extract_triples(
                    chunk['text'],
                    chunk['chunk_number']
                )
                
                if success:
                    print(f"Successfully extracted {len(triples)} triples from chunk {chunk['chunk_number']}")
                    all_extracted_triples.extend(triples)
                else:
                    print(f"Failed to process chunk {chunk['chunk_number']}: {error}")
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