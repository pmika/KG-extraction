import os
import warnings
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure settings for better display and fewer warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 150)

# Text Processing Configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 100

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# LLM Provider Selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI if not specified

# LLM Configuration
LLM_MODEL_NAME = {
    "openai": "gpt-4-turbo",
    "anthropic": "claude-3-opus-20240229"
}[LLM_PROVIDER]

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096  # Default max tokens for both OpenAI and Anthropic

# System and User Prompts
EXTRACTION_SYSTEM_PROMPT = """
You are an AI expert specialized in knowledge graph extraction. 
Your task is to identify and extract ALL factual Subject-Predicate-Object (SPO) triples from the given text.
Focus on extracting multiple relationships and be thorough in identifying all possible connections.
Adhere strictly to the JSON output format requested in the user prompt.
Extract both core entities and their relationships, including biographical facts, achievements, and connections.
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """
Please extract ALL Subject-Predicate-Object (S-P-O) triples from the text below.

**VERY IMPORTANT RULES:**
1.  **Output Format:** Respond ONLY with a single, valid JSON array. Each element MUST be an object with keys "subject", "predicate", "object". You should return an array even if it contains a single element.
2.  **JSON Only:** Do NOT include any text before or after the JSON array (e.g., no 'Here is the JSON:' or explanations). Do NOT use markdown ```json ... ``` tags.
3.  **Concise Predicates:** Keep the 'predicate' value concise (1-3 words, ideally 1-2). Use verbs or short verb phrases (e.g., 'discovered', 'was born in', 'won').
4.  **Lowercase:** ALL values for 'subject', 'predicate', and 'object' MUST be lowercase.
5.  **Pronoun Resolution:** Replace pronouns (she, he, it, her, etc.) with the specific lowercase entity name they refer to based on the text context (e.g., 'marie curie').
6.  **Specificity:** Capture specific details (e.g., 'nobel prize in physics' instead of just 'nobel prize' if specified).
7.  **Completeness:** Extract ALL distinct factual relationships mentioned.

**Text to Process:**
```text
{text_chunk}
```

**Required JSON Output Format Example:**
[
  {{ "subject": "marie curie", "predicate": "discovered", "object": "radium" }},
  {{ "subject": "marie curie", "predicate": "won", "object": "nobel prize in physics" }},
  {{ "subject": "marie curie", "predicate": "was born as", "object": "maria skłodowska" }}
]

**Your JSON Output (MUST start with '[' and end with ']'):**
""" 