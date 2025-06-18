from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

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
    def extract_triples(self, user_prompt: str, chunk_number: int) -> Tuple[bool, Union[List[Dict], Dict, str], Optional[str]]:
        """
        Extract information from a text chunk using the LLM.
        
        Args:
            user_prompt (str): The fully formatted user prompt
            chunk_number (int): The chunk number for tracking
            
        Returns:
            tuple: (success, result, error_message)
            - success (bool): Whether the extraction was successful
            - result: For triple extraction: List of dicts with subject/predicate/object
                     For JSON-LD: Dict containing JSON-LD data or string containing JSON-LD
            - error_message (str): Error message if unsuccessful
        """
        pass 