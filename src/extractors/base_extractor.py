from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from src.config.configuration import Configuration
from src.models.base_llm_client import BaseLLMClient


class BaseExtractor(ABC):
    """Abstract base class for information extractors."""
    
    def __init__(self, llm_client: BaseLLMClient, config: Configuration):
        """
        Initialize the extractor.
        
        Args:
            llm_client: LLM client for making API calls
            config: Configuration settings
        """
        self.llm_client = llm_client
        self.config = config
    
    @abstractmethod
    def extract_from_chunk(self, chunk: Dict[str, Union[str, int]]) -> Tuple[bool, Union[List[Dict], Dict], Optional[str]]:
        """
        Extract information from a text chunk.
        
        Args:
            chunk: Dictionary containing chunk text and number
            
        Returns:
            tuple: (success, extracted_data, error_message)
        """
        pass
    
    @abstractmethod
    def process_results(self, all_extracted_data: List[Union[List[Dict], Dict]], failed_chunks: List[Dict]) -> Dict:
        """
        Process and combine all extracted data.
        
        Args:
            all_extracted_data: List of extracted data from all chunks
            failed_chunks: List of failed chunks with error information
            
        Returns:
            Dictionary containing processed results and statistics
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Union[List[Dict], Dict]) -> bool:
        """
        Validate extracted data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass 