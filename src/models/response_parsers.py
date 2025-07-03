import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
from src.utils.logger import Logger


class BaseResponseParser(ABC):
    """Abstract base class for response parsers."""
    
    @abstractmethod
    def parse(self, response: str, chunk_number: int) -> Tuple[bool, Union[List[Dict], Dict], Optional[str]]:
        """
        Parse the LLM response.
        
        Args:
            response: Raw response from LLM
            chunk_number: Chunk number for tracking
            
        Returns:
            tuple: (success, parsed_data, error_message)
        """
        pass


class TripleResponseParser(BaseResponseParser):
    """Parser for triple extraction responses."""
    
    def parse(self, response: str, chunk_number: int) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Parse triple extraction response.
        
        Args:
            response: Raw response from LLM
            chunk_number: Chunk number for tracking
            
        Returns:
            tuple: (success, triples, error_message)
        """
        try:
            if not response.strip():
                return False, [], "Empty response from LLM"
            
            # Parse the JSON response
            parsed_data = json.loads(response)
            
            # Handle different response formats
            if isinstance(parsed_data, dict):
                # Check if this is a single triple object
                if all(k in parsed_data for k in ['subject', 'predicate', 'object']):
                    parsed_json = [parsed_data]
                else:
                    # Look for a list of triples in the dictionary
                    list_values = [v for v in parsed_data.values() if isinstance(v, list)]
                    if len(list_values) == 1:
                        parsed_json = list_values[0]
                    else:
                        return False, [], "JSON object received, but doesn't contain a single list of triples"
            elif isinstance(parsed_data, list):
                parsed_json = parsed_data
            else:
                return False, [], "Parsed JSON is not a list or expected dictionary wrapper"
            
            # Validate and extract triples
            valid_triples = []
            for item in parsed_json:
                if isinstance(item, dict) and all(k in item for k in ['subject', 'predicate', 'object']):
                    if all(isinstance(item[k], str) for k in ['subject', 'predicate', 'object']):
                        item['chunk'] = chunk_number
                        valid_triples.append(item)
            
            Logger.info(f"Successfully parsed {len(valid_triples)} triples from chunk {chunk_number}")
            return True, valid_triples, None
            
        except json.JSONDecodeError as json_err:
            error_msg = f"JSON parsing error: {str(json_err)}"
            Logger.error(error_msg)
            return False, [], error_msg
        except Exception as e:
            error_msg = f"Unexpected parsing error: {str(e)}"
            Logger.error(error_msg)
            return False, [], error_msg


class JSONLDResponseParser(BaseResponseParser):
    """Parser for JSON-LD extraction responses."""
    
    def parse(self, response: str, chunk_number: int) -> Tuple[bool, Dict, Optional[str]]:
        """
        Parse JSON-LD extraction response.
        
        Args:
            response: Raw response from LLM
            chunk_number: Chunk number for tracking
            
        Returns:
            tuple: (success, jsonld_data, error_message)
        """
        try:
            if not response.strip():
                return False, {}, "Empty response from LLM"
            
            # Parse the JSON response
            parsed_data = json.loads(response)
            
            # Validate JSON-LD structure
            if isinstance(parsed_data, dict):
                if "@graph" in parsed_data:
                    Logger.info(f"Successfully parsed JSON-LD from chunk {chunk_number}")
                    return True, parsed_data, None
                else:
                    return False, {}, "JSON-LD response missing @graph key"
            else:
                return False, {}, "Parsed JSON is not a dictionary"
                
        except json.JSONDecodeError as json_err:
            error_msg = f"JSON parsing error: {str(json_err)}"
            Logger.error(error_msg)
            return False, {}, error_msg
        except Exception as e:
            error_msg = f"Unexpected parsing error: {str(e)}"
            Logger.error(error_msg)
            return False, {}, error_msg


class ResponseParserFactory:
    """Factory for creating appropriate response parsers."""
    
    @staticmethod
    def create_parser(extraction_mode: str) -> BaseResponseParser:
        """
        Create a response parser based on extraction mode.
        
        Args:
            extraction_mode: "triples" or "jsonld"
            
        Returns:
            Appropriate response parser instance
        """
        if extraction_mode == "triples":
            return TripleResponseParser()
        elif extraction_mode == "jsonld":
            return JSONLDResponseParser()
        else:
            raise ValueError(f"Unsupported extraction mode: {extraction_mode}") 