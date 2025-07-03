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
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# LLM Provider Selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # Default to OpenAI if not specified

# LLM Configuration
LLM_MODEL_NAMES = {
    "openai": "gpt-4-turbo",
    "anthropic": "claude-3-5-sonnet-20241022"
}

LLM_MODEL_NAME = LLM_MODEL_NAMES[LLM_PROVIDER]

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

# JSON-LD Extraction Prompts
JSONLD_SYSTEM_PROMPT = """
You are an AI expert specialized in extracting structured information from text and representing it in JSON-LD format according to a provided ontology.

Your task is to:
1. Analyze the given text and identify entities, relationships, and attributes
2. Map these to the appropriate classes and properties from the provided ontology
3. Represent the information in JSON-LD format, using the provided context
4. Ensure all extracted information is properly linked and typed
5. Use appropriate JSON-LD features like @type, @id, and nested objects to represent complex relationships
6. Create unique @id values for entities using a consistent naming scheme
7. Include all relevant attributes and relationships, even if they require multiple triples to represent
8. Make sure each unique entity is represented with one unique id. 
9. Make sure there are no duplicate entities in the output.

**CRITICAL INSTRUCTIONS:**
- You MUST use ONLY the exact class names and property names listed below from the ontology. Do NOT invent new properties or classes. Do NOT use synonyms or alternate spellings.
- If you cannot express a fact using the provided ontology, OMIT it from the output.
- Use the property and class names EXACTLY as they appear in the lists below.
- Use ONLY the provided local context. Do NOT reference any remote URLs or external contexts.

**Ontology Information:**
- **Available Ontology Classes (ONLY use these for @type):**
{classes}
- **Available Object Properties (ONLY use these for relationships):**
{object_properties}
- **Available Data Properties (ONLY use these for attributes):**
{data_properties}
- **Base IRI:** {base_iri}
- **Ontology Context (USE THIS EXACT CONTEXT):**
{context}
- **Full Ontology (OWL):**
{ontology_owl}

**Output Format Requirements:**
- Use the @graph array to contain all entities
- Use the provided context for all terms
- Do NOT include any @context field that references remote URLs
- Ensure all @id values are unique and follow a consistent pattern (e.g., "person:marie_curie", "place:warsaw")

The output should be valid JSON-LD that can be expanded and compacted using the provided context.
"""

JSONLD_USER_PROMPT_TEMPLATE = """
Please extract information from the text below and represent it in JSON-LD format using the ontology and context provided in the system prompt.

**VERY IMPORTANT RULES:**
1. Output Format: Respond ONLY with a single, valid JSON-LD object
2. Use only the ontology classes and properties described in the system prompt
3. Ensure the output is valid JSON-LD that can be expanded and compacted
4. Use the @graph array to contain all entities

**Text to Process:**
```text
{text_chunk}
```

**Your JSON-LD Output (MUST use only the provided ontology classes and properties):**
""" 