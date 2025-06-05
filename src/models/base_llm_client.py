from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def __init__(
        self, 
        model_name: str = None, 
        temperature: float = None, 
        max_tokens: int = None,
        system_prompt: str = None,
        user_prompt_template: str = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Optional model name to use
            temperature: Optional temperature to use
            max_tokens: Optional maximum tokens to use
            system_prompt: Optional system prompt to use
            user_prompt_template: Optional user prompt template to use
        """
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