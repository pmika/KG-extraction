from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def __init__(self):
        """Initialize the LLM client."""
        pass
        
    @abstractmethod
    def extract_triples(self, text_chunk, chunk_number):
        """
        Extract triples from a text chunk using the LLM.
        
        Args:
            text_chunk (str): The text to process
            chunk_number (int): The chunk number for tracking
            
        Returns:
            tuple: (success, result, error_message)
            - success (bool): Whether the extraction was successful
            - result (list): List of extracted triples if successful
            - error_message (str): Error message if unsuccessful
        """
        pass 