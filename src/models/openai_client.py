import openai
import time
import json
from src.models.base_llm_client import BaseLLMClient
from src.config.settings import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE
)

class OpenAIClient(BaseLLMClient):
    def __init__(self):
        """Initialize the OpenAI client."""
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found. Please ensure you have created a .env file with OPENAI_API_KEY=your-api-key")
            
        # Check if we're in test mode
        self.is_test_mode = self.api_key == "test-key"
        
        if not self.is_test_mode:
            self.client = openai.OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=self.api_key
            )
            
        self.model_name = LLM_MODEL_NAME
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS

    def extract_triples(self, text_chunk, chunk_number):
        """
        Extract triples from a text chunk using the OpenAI API.
        
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
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and parse the response
            llm_output = response.choices[0].message.content.strip()
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
                
        except openai.APIError as e:
            return False, None, f"OpenAI API Error: {str(e)}"
        except openai.RateLimitError as e:
            time.sleep(60)  # Wait before retrying
            return False, None, f"Rate limit exceeded: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}" 