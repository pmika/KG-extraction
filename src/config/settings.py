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
LLM_MODEL_NAMES = {
    "openai": "gpt-4-turbo",
    "anthropic": "claude-3-7-sonnet-20250219"
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

The output should be valid JSON-LD that can be expanded and compacted using the provided context.
"""

JSONLD_USER_PROMPT_TEMPLATE = """
Please extract information from the text below and represent it in JSON-LD format using the provided ontology context and the full OWL ontology definitions.

**VERY IMPORTANT RULES:**
1. Output Format: Respond ONLY with a single, valid JSON-LD object
2. **ONTOLOGY COMPLIANCE:** You MUST ONLY use classes and properties that are explicitly listed in the provided ontology. Do NOT use any classes or properties that are not in the provided lists.
3. Use @type to specify the class of entities - ONLY use classes from the provided Classes list
4. Use @id to create unique identifiers for entities (e.g., "entity:unique_id")
5. Represent complex relationships using nested objects
6. Include all relevant attributes and relationships that can be mapped to the provided ontology
7. Ensure the output is valid JSON-LD that can be expanded and compacted
8. Use the @graph array to contain all entities
9. Create new entities for any referenced objects that have their own properties
10. **STRICT ADHERENCE:** If information in the text cannot be mapped to the provided ontology classes and properties, either omit it or map it to the most appropriate available class/property
11. **USE DEFINITIONS:** Use the definitions and descriptions from the full OWL ontology below to guide your mapping and ensure correct usage of terms.

**Available Ontology Classes (ONLY use these for @type):**
{classes}

**Available Object Properties (ONLY use these for relationships):**
{object_properties}

**Available Data Properties (ONLY use these for attributes):**
{data_properties}

**Base IRI:** {base_iri}

**Ontology Context:**
{context}

**Full Ontology (OWL):**
```owl
{ontology_owl}
```

**Text to Process:**
```text
{text_chunk}
```

**Example JSON-LD Output (using only provided ontology classes):**
{{
  "@context": {context},
  "@graph": [
    {{
      "@id": "entity:example_entity",
      "@type": "[USE_ONLY_CLASSES_FROM_ABOVE]",
      "[USE_ONLY_DATA_PROPERTIES_FROM_ABOVE]": "value",
      "[USE_ONLY_OBJECT_PROPERTIES_FROM_ABOVE]": [
        {{
          "@id": "entity:related_entity",
          "@type": "[USE_ONLY_CLASSES_FROM_ABOVE]",
          "[USE_ONLY_DATA_PROPERTIES_FROM_ABOVE]": "related_value"
        }}
      ]
    }}
  ]
}}

**Your JSON-LD Output (MUST use only the provided ontology classes and properties):**
""" 