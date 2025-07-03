import openai
import time
import json
import os
from src.models.base_llm_client import BaseLLMClient
from src.config.settings import (
    OPENAI_API_KEY,
    OPENAI_API_BASE
)
from typing import List, Dict

class OpenAIClient(BaseLLMClient):
    def __init__(
        self, 
        model_name: str = None, 
        temperature: float = None, 
        max_tokens: int = None,
        system_prompt: str = None,
        user_prompt_template: str = None
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name: Model name to use
            temperature: Temperature to use
            max_tokens: Maximum tokens to use
            system_prompt: System prompt to use
            user_prompt_template: User prompt template to use
        """
        # Use provided values or fall back to environment variables
        self.api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Check if we're in test mode
        self.is_test_mode = self.api_key == "test-key"
        
        if not self.is_test_mode:
            self.client = openai.OpenAI(
                base_url=os.getenv("OPENAI_API_BASE") or OPENAI_API_BASE,
                api_key=self.api_key
            )
            
        # Use provided values (no fallbacks to settings)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        
        print(f"\nOpenAI client initialized with:")
        print(f"Model: {self.model_name}")
        print(f"Temperature: {self.temperature}")
        print(f"System prompt length: {len(self.system_prompt)}")
        print(f"User prompt template length: {len(self.user_prompt_template)}")

    def extract_triples(self, user_prompt, chunk_number):
        """
        Extract information from a text chunk using the OpenAI API.
        
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
            print(f"\nMaking API call to OpenAI for chunk {chunk_number}...")
            print(f"Using model: {self.model_name}")
            print(f"System prompt length: {len(self.system_prompt)}")
            print(f"User prompt length: {len(user_prompt)}")
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            print(f"Received response from OpenAI for chunk {chunk_number}")
            
            # Extract and parse the response
            llm_output = response.choices[0].message.content.strip()
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
                    
                    print(f"Successfully parsed {len(valid_triples)} triples from response")
                    return True, valid_triples, None
                
            except json.JSONDecodeError as json_err:
                print(f"JSON parsing error: {str(json_err)}")
                return False, None, f"JSON parsing error: {str(json_err)}"
                
        except openai.APIError as e:
            print(f"OpenAI API Error: {str(e)}")
            return False, None, f"OpenAI API Error: {str(e)}"
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded: {str(e)}")
            time.sleep(60)  # Wait before retrying
            return False, None, f"Rate limit exceeded: {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return False, None, f"Unexpected error: {str(e)}" 