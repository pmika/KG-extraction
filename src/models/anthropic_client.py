import anthropic
import time
import json
from src.models.base_llm_client import BaseLLMClient
from src.config.settings import (
    ANTHROPIC_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE
)
import requests
from requests.exceptions import Timeout, RequestException
import signal
from contextlib import contextmanager
import sys
import os
from typing import List, Dict

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class AnthropicClient(BaseLLMClient):
    def __init__(self, model_name: str = None, temperature: float = None, max_tokens: int = None):
        """
        Initialize the Anthropic client.
        
        Args:
            model_name: Optional model name to use. If not provided, uses the global setting.
            temperature: Optional temperature to use. If not provided, uses the global setting.
            max_tokens: Optional maximum tokens to use. If not provided, uses the global setting.
        """
        self.api_key = ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        # Check if we're in test mode
        self.is_test_mode = ANTHROPIC_API_KEY == "test-key"
        
        if not self.is_test_mode:
            self.client = anthropic.Anthropic(
                api_key=ANTHROPIC_API_KEY
            )
            
        self.model_name = model_name or LLM_MODEL_NAME
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        self.timeout = 30  # 30 seconds timeout
        
        print(f"\nAnthropic client initialized with:")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")

    def extract_triples(self, text_chunk, chunk_number):
        """
        Extract triples from a text chunk using the Anthropic API.
        
        Args:
            text_chunk (str): The text to process
            chunk_number (int): The chunk number for tracking
            
        Returns:
            tuple: (success, result, error_message)
            - success (bool): Whether the extraction was successful
            - result (list): List of extracted triples if successful
            - error_message (str): Error message if unsuccessful
        """
        if self.is_test_mode:
            # Return mock data for testing
            return True, [
                {
                    "subject": "marie curie",
                    "predicate": "discovered",
                    "object": "radium",
                    "chunk": chunk_number
                },
                {
                    "subject": "marie curie",
                    "predicate": "won",
                    "object": "nobel prize in physics",
                    "chunk": chunk_number
                }
            ], None
            
        try:
            # Format the user prompt
            user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(text_chunk=text_chunk)
            
            print(f"Making API call to Anthropic for chunk {chunk_number}...")
            print(f"Using model: {self.model_name}")
            print(f"System prompt length: {len(EXTRACTION_SYSTEM_PROMPT)}")
            print(f"User prompt length: {len(user_prompt)}")
            
            try:
                with time_limit(self.timeout):
                    # Make the API call with correct message format for Anthropic
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        system=EXTRACTION_SYSTEM_PROMPT,  # System prompt as top-level parameter
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature
                    )
            except TimeoutException:
                print(f"Request timed out after {self.timeout} seconds")
                return False, None, f"Request timed out after {self.timeout} seconds for chunk {chunk_number}"
            
            print(f"Received response from Anthropic for chunk {chunk_number}")
            
            # Extract and parse the response
            llm_output = response.content[0].text.strip()
            if not llm_output:
                return False, None, "Empty response from LLM"
                
            # Parse the JSON response
            try:
                parsed_data = json.loads(llm_output)
                
                # Handle if response is a dict containing the list
                if isinstance(parsed_data, dict):
                    # Check if this is a single triple object
                    if all(k in parsed_data for k in ['subject', 'predicate', 'object']):
                        parsed_json = [parsed_data]  # Wrap single triple in array
                    else:
                        list_values = [v for v in parsed_data.values() if isinstance(v, list)]
                        if len(list_values) == 1:
                            parsed_json = list_values[0]
                        else:
                            return False, None, "JSON object received, but doesn't contain a single list of triples"
                elif isinstance(parsed_data, list):
                    parsed_json = parsed_data
                else:
                    return False, None, "Parsed JSON is not a list or expected dictionary wrapper"
                
                # Validate and extract triples
                valid_triples = []
                for item in parsed_json:
                    if isinstance(item, dict) and all(k in item for k in ['subject', 'predicate', 'object']):
                        if all(isinstance(item[k], str) for k in ['subject', 'predicate', 'object']):
                            item['chunk'] = chunk_number
                            valid_triples.append(item)
                
                return True, valid_triples, None
                
            except json.JSONDecodeError as json_err:
                return False, None, f"JSON parsing error: {str(json_err)}"
                
        except Timeout:
            return False, None, f"Request timed out after {self.timeout} seconds for chunk {chunk_number}"
        except RequestException as e:
            return False, None, f"Network error: {str(e)}"
        except anthropic.APIError as e:
            return False, None, f"Anthropic API Error: {str(e)}"
        except anthropic.RateLimitError as e:
            time.sleep(60)  # Wait before retrying
            return False, None, f"Rate limit exceeded: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}" 