from typing import Dict, List, Optional, Tuple, Union
from src.extractors.base_extractor import BaseExtractor
from src.models.base_llm_client import BaseLLMClient
from src.config.configuration import Configuration
from src.processors.text_processor import TextProcessor
from src.utils.logger import Logger


class TripleExtractor(BaseExtractor):
    """Extractor for Subject-Predicate-Object triples."""
    
    def __init__(self, llm_client: BaseLLMClient, config: Configuration):
        """
        Initialize the triple extractor.
        
        Args:
            llm_client: LLM client for making API calls
            config: Configuration settings
        """
        super().__init__(llm_client, config)
        self.text_processor = TextProcessor(
            chunk_size=config.text_processing.chunk_size,
            overlap=config.text_processing.chunk_overlap
        )
    
    def extract_from_chunk(self, chunk: Dict[str, Union[str, int]]) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Extract triples from a text chunk.
        
        Args:
            chunk: Dictionary containing chunk text and number
            
        Returns:
            tuple: (success, triples, error_message)
        """
        try:
            Logger.info(f"Processing chunk {chunk['chunk_number']} for triple extraction")
            
            # Format user prompt
            user_prompt = self.llm_client.user_prompt_template.format(text_chunk=chunk['text'])
            
            # Extract triples using LLM client
            success, data, error = self.llm_client.extract_triples(user_prompt, chunk['chunk_number'])
            
            if success:
                if self.validate_data(data):
                    Logger.info(f"Successfully extracted {len(data)} triples from chunk {chunk['chunk_number']}")
                    return True, data, None
                else:
                    error_msg = f"Invalid triple data from chunk {chunk['chunk_number']}"
                    Logger.warning(error_msg)
                    return False, [], error_msg
            else:
                Logger.error(f"Failed to extract triples from chunk {chunk['chunk_number']}: {error}")
                return False, [], error
                
        except Exception as e:
            error_msg = f"Error processing chunk {chunk['chunk_number']}: {str(e)}"
            Logger.error(error_msg)
            return False, [], error_msg
    
    def process_results(self, all_extracted_data: List[List[Dict]], failed_chunks: List[Dict]) -> Dict:
        """
        Process and combine all extracted triples.
        
        Args:
            all_extracted_data: List of triple lists from all chunks
            failed_chunks: List of failed chunks with error information
            
        Returns:
            Dictionary containing processed results and statistics
        """
        try:
            # Flatten all triples
            all_triples = []
            for triples in all_extracted_data:
                all_triples.extend(triples)
            
            # Normalize and deduplicate triples
            normalized_triples = self.text_processor.deduplicate_triples(all_triples)
            
            # Prepare statistics
            stats = {
                'total_chunks': len(all_extracted_data) + len(failed_chunks),
                'processed_chunks': len(all_extracted_data),
                'failed_chunks': len(failed_chunks),
                'total_triples': len(all_triples),
                'unique_triples': len(normalized_triples),
                'duplicates_removed': len(all_triples) - len(normalized_triples)
            }
            
            Logger.info(f"Processed {stats['total_triples']} triples, {stats['unique_triples']} unique after deduplication")
            
            return {
                'triples': normalized_triples,
                'statistics': stats,
                'failed_chunks': failed_chunks
            }
            
        except Exception as e:
            Logger.error(f"Error processing triple results: {str(e)}")
            return {
                'triples': [],
                'statistics': {
                    'total_chunks': len(all_extracted_data) + len(failed_chunks),
                    'processed_chunks': 0,
                    'failed_chunks': len(all_extracted_data) + len(failed_chunks),
                    'total_triples': 0,
                    'unique_triples': 0,
                    'duplicates_removed': 0
                },
                'failed_chunks': failed_chunks + [{'error': f'Processing error: {str(e)}'}]
            }
    
    def validate_data(self, data: List[Dict]) -> bool:
        """
        Validate extracted triple data.
        
        Args:
            data: List of triple dictionaries
            
        Returns:
            True if data is valid, False otherwise
        """
        if not isinstance(data, list):
            return False
        
        for triple in data:
            if not isinstance(triple, dict):
                return False
            
            # Check required fields
            required_fields = ['subject', 'predicate', 'object']
            if not all(field in triple for field in required_fields):
                return False
            
            # Check field types
            if not all(isinstance(triple[field], str) for field in required_fields):
                return False
            
            # Check for empty values
            if not all(triple[field].strip() for field in required_fields):
                return False
        
        return True 