import anthropic
import time
import json
import os
from src.models.base_llm_client import BaseLLMClient
from src.config.settings import (
    ANTHROPIC_API_KEY
)
import requests
from requests.exceptions import Timeout, RequestException
import signal
from contextlib import contextmanager
import sys
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
    def __init__(
        self, 
        model_name: str = None, 
        temperature: float = None, 
        max_tokens: int = None,
        system_prompt: str = None,
        user_prompt_template: str = None
    ):
        """
        Initialize the Anthropic client.
        
        Args:
            model_name: Model name to use
            temperature: Temperature to use
            max_tokens: Maximum tokens to use
            system_prompt: System prompt to use
            user_prompt_template: User prompt template to use
        """
        # Use provided values or fall back to environment variables
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        # Check if we're in test mode
        self.is_test_mode = self.api_key == "test-key"
        
        if not self.is_test_mode:
            self.client = anthropic.Anthropic(
                api_key=self.api_key
            )
            
        # Use provided values (no fallbacks to settings)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.timeout = 30  # 30 seconds timeout
        
        print(f"\nAnthropic client initialized with:")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")
        print(f"System prompt length: {len(self.system_prompt)}")
        print(f"User prompt template length: {len(self.user_prompt_template)}")

    def extract_triples(self, user_prompt, chunk_number):
        """
        Extract information from a text chunk using the Anthropic API.
        
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
        if self.is_test_mode:
            # Return mock data for testing
            if "JSON-LD" in self.system_prompt:
                # Return mock JSON-LD data
                return True, {
                    "@graph": [{
                        "@id": "person:marie_curie",
                        "@type": "Person",
                        "name": "Marie Curie",
                        "discovered": [{
                            "@id": "element:radium",
                            "@type": "Discovery",
                            "name": "Radium"
                        }]
                    }]
                }, None
            else:
                # Return mock triple data
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
            print(f"Making API call to Anthropic for chunk {chunk_number}...")
            print(f"Using model: {self.model_name}")
            print(f"System prompt length: {len(self.system_prompt)}")
            print(f"User prompt length: {len(user_prompt)}")
            
            try:
                with time_limit(self.timeout):
                    # Make the API call with correct message format for Anthropic
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        system=[
                            {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
                        ],
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature
                    )
            except TimeoutException:
                print(f"Request timed out after {self.timeout} seconds")
                return False, None, f"Request timed out after {self.timeout} seconds for chunk {chunk_number}"
            
            print(f"Received response from Anthropic for chunk {chunk_number}")
            
            # Calculate and print cost
            try:
                usage = getattr(response, 'usage', None)
                if usage:
                    input_tokens = getattr(usage, 'input_tokens', 0)
                    output_tokens = getattr(usage, 'output_tokens', 0)
                else:
                    # fallback for dict-like response
                    input_tokens = response.get('usage', {}).get('input_tokens', 0)
                    output_tokens = response.get('usage', {}).get('output_tokens', 0)

                # Pricing per 1k tokens (as of June 2024)
                model_prices = {
                    # Claude 3
                    'claude-3-opus-20240229': (0.015, 0.075),
                    'claude-3-sonnet-20240229': (0.003, 0.015),
                    'claude-3-haiku-20240307': (0.00025, 0.00125),
                    # Claude 3.5
                    'claude-3-5-sonnet-20240620': (0.003, 0.015),
                    'claude-3-5-sonnet-20241022': (0.003, 0.015),
                    'claude-3-5-haiku-20241022': (0.0008, 0.004),
                    # Claude 3.7
                    'claude-3-7-sonnet-20250219': (0.003, 0.015),
                    # Claude 4
                    'claude-opus-4-20250514': (0.015, 0.075),
                    'claude-sonnet-4-20250514': (0.003, 0.015),
                }
                # Default to Sonnet pricing if model not found
                input_price, output_price = model_prices.get(self.model_name, (0.003, 0.015))
                cost = (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
                print(f"Token usage: input={input_tokens}, output={output_tokens}")
                print(f"Estimated cost for this call: ${cost:.6f} (model: {self.model_name})")
            except Exception as e:
                print(f"[Cost Calculation Error] {e}")
            
            # Extract and parse the response
            llm_output = response.content[0].text.strip()
            if not llm_output:
                return False, None, "Empty response from LLM"
                
            # Parse the JSON response
            try:
                parsed_data = json.loads(llm_output)
                
                # Check if we're in JSON-LD mode
                if "JSON-LD" in self.system_prompt:
                    # Return the JSON-LD data directly
                    return True, parsed_data, None
                else:
                    # Handle triple extraction format
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