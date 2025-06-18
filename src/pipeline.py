import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.plugins.serializers.jsonld import from_rdf
from rdflib.plugins.parsers.jsonld import to_rdf
from src.models.openai_client import OpenAIClient
from src.models.anthropic_client import AnthropicClient
from src.processors.text_processor import TextProcessor
from src.processors.ontology_processor import OntologyProcessor
from src.config.settings import (
    LLM_PROVIDER, 
    LLM_MODEL_NAME, 
    LLM_TEMPERATURE, 
    LLM_MAX_TOKENS,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
    JSONLD_SYSTEM_PROMPT,
    JSONLD_USER_PROMPT_TEMPLATE
)

class KnowledgeGraphPipeline:
    def __init__(
        self, 
        llm_provider: str = None, 
        model_name: str = None, 
        temperature: float = None, 
        max_tokens: int = None,
        ontology_path: Union[str, Path] = None
    ):
        """
        Initialize the knowledge graph extraction pipeline.
        
        Args:
            llm_provider: Optional LLM provider to use. If not provided, uses the global setting.
            model_name: Optional model name to use. If not provided, uses the global setting.
            temperature: Optional temperature to use. If not provided, uses the global setting.
            max_tokens: Optional maximum tokens to use. If not provided, uses the global setting.
            ontology_path: Optional path to an OWL ontology file. If provided, enables JSON-LD extraction.
        """
        # Initialize ontology processor if path is provided
        self.ontology_processor = None
        self.ontology_info = None
        self.ontology_context = None
        if ontology_path:
            self.ontology_processor = OntologyProcessor(ontology_path)
            self.ontology_info = self.ontology_processor.get_ontology_info()
            self.ontology_context = self.ontology_processor.get_context()
            
        # Initialize the appropriate LLM client based on the provider
        provider = llm_provider if llm_provider is not None else LLM_PROVIDER
        model = model_name if model_name is not None else LLM_MODEL_NAME
        temp = temperature if temperature is not None else LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        
        # Use JSON-LD prompts if ontology is provided
        if self.ontology_processor:
            system_prompt = JSONLD_SYSTEM_PROMPT
            user_prompt = JSONLD_USER_PROMPT_TEMPLATE  # Do not format here
        else:
            # Use original triple extraction prompts
            system_prompt = EXTRACTION_SYSTEM_PROMPT
            user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE
        
        print(f"\nCreating pipeline with provider: {provider}")
        print(f"Model name: {model}")
        print(f"Temperature: {temp}")
        print(f"Max tokens: {tokens}")
        print(f"Using {'JSON-LD' if self.ontology_processor else 'triple'} extraction")
        
        if provider == "openai":
            print("Initializing OpenAI client...")
            self.llm_client = OpenAIClient(
                model_name=model, 
                temperature=temp, 
                max_tokens=tokens,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt
            )
        elif provider == "anthropic":
            print("Initializing Anthropic client...")
            self.llm_client = AnthropicClient(
                model_name=model, 
                temperature=temp, 
                max_tokens=tokens,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        self.text_processor = TextProcessor()
        
    def _normalize_jsonld_through_rdf(self, jsonld_data: Dict) -> Dict:
        """
        Normalize JSON-LD data by converting it to an RDF graph and back.
        This ensures proper deduplication and RDF semantics.
        
        Args:
            jsonld_data (Dict): The JSON-LD data to normalize
            
        Returns:
            Dict: Normalized JSON-LD data
        """
        # Create a new RDF graph
        g = Graph()
        
        # Add the JSON-LD data to the graph
        g.parse(data=json.dumps(jsonld_data), format='json-ld')
        
        # Convert back to JSON-LD using the ontology's context
        context = self.ontology_processor.get_context()["@context"]
        normalized = from_rdf(g, context)
        
        return normalized

    def process_text(self, text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Process text through the pipeline.
        
        Args:
            text (str): The input text to process
            
        Returns:
            tuple: (success, result, error_message)
            - success (bool): Whether the processing was successful
            - result (dict): Dictionary containing processed data and statistics
            - error_message (str): Error message if unsuccessful
        """
        try:
            # 1. Split text into chunks
            chunks = self.text_processor.split_into_chunks(text)
            if not chunks:
                return False, None, "No chunks were created from the input text"
                
            print(f"\nProcessing {len(chunks)} chunks...")
                
            # 2. Process each chunk
            all_extracted_data = []
            failed_chunks = []
            
            for chunk in chunks:
                print(f"\nProcessing chunk {chunk['chunk_number']}...")
                if self.ontology_processor:
                    prompt_vars = {
                        'classes': ", ".join(self.ontology_info.get("classes", [])),
                        'object_properties': ", ".join(self.ontology_info.get("object_properties", [])),
                        'data_properties': ", ".join(self.ontology_info.get("data_properties", [])),
                        'base_iri': self.ontology_info.get("base_iri", ""),
                        'context': json.dumps(self.ontology_context, indent=2),
                        'ontology_owl': self.ontology_processor.get_owl_content(),
                        'text_chunk': chunk['text']
                    }
                    user_prompt = self.llm_client.user_prompt_template.format(**prompt_vars)
                else:
                    user_prompt = self.llm_client.user_prompt_template.format(text_chunk=chunk['text'])

                success, data, error = self.llm_client.extract_triples(
                    user_prompt,
                    chunk['chunk_number']
                )
                
                if success:
                    if self.ontology_processor:
                        # For JSON-LD mode, data should already be a JSON-LD object
                        try:
                            if isinstance(data, dict) and "@graph" in data:
                                # Direct JSON-LD response
                                if self.ontology_processor.validate_jsonld(data):
                                    normalized = self.ontology_processor.normalize_jsonld(data)
                                    if normalized:
                                        print(f"Successfully extracted and validated JSON-LD from chunk {chunk['chunk_number']}")
                                        all_extracted_data.append(normalized)
                                    else:
                                        print(f"Failed to normalize JSON-LD from chunk {chunk['chunk_number']}")
                                        failed_chunks.append({
                                            'chunk_number': chunk['chunk_number'],
                                            'error': "JSON-LD normalization failed"
                                        })
                                else:
                                    print(f"Invalid JSON-LD in chunk {chunk['chunk_number']}")
                                    failed_chunks.append({
                                        'chunk_number': chunk['chunk_number'],
                                        'error': "Invalid JSON-LD"
                                    })
                            else:
                                # Try to parse as JSON-LD if it's a string
                                if isinstance(data, str):
                                    try:
                                        json_data = json.loads(data)
                                        if self.ontology_processor.validate_jsonld(json_data):
                                            normalized = self.ontology_processor.normalize_jsonld(json_data)
                                            if normalized:
                                                print(f"Successfully extracted and validated JSON-LD from chunk {chunk['chunk_number']}")
                                                all_extracted_data.append(normalized)
                                            else:
                                                print(f"Failed to normalize JSON-LD from chunk {chunk['chunk_number']}")
                                                failed_chunks.append({
                                                    'chunk_number': chunk['chunk_number'],
                                                    'error': "JSON-LD normalization failed"
                                                })
                                        else:
                                            print(f"Invalid JSON-LD in chunk {chunk['chunk_number']}")
                                            failed_chunks.append({
                                                'chunk_number': chunk['chunk_number'],
                                                'error': "Invalid JSON-LD"
                                            })
                                    except json.JSONDecodeError:
                                        print(f"Invalid JSON in chunk {chunk['chunk_number']}")
                                        failed_chunks.append({
                                            'chunk_number': chunk['chunk_number'],
                                            'error': "Invalid JSON"
                                        })
                                else:
                                    print(f"Unexpected data format in chunk {chunk['chunk_number']}")
                                    failed_chunks.append({
                                        'chunk_number': chunk['chunk_number'],
                                        'error': "Unexpected data format"
                                    })
                        except Exception as e:
                            print(f"Error processing JSON-LD in chunk {chunk['chunk_number']}: {str(e)}")
                            failed_chunks.append({
                                'chunk_number': chunk['chunk_number'],
                                'error': f"JSON-LD processing error: {str(e)}"
                            })
                    else:
                        # Original triple extraction
                        print(f"Successfully extracted {len(data)} triples from chunk {chunk['chunk_number']}")
                        all_extracted_data.extend(data)
                else:
                    print(f"Failed to process chunk {chunk['chunk_number']}: {error}")
                    failed_chunks.append({
                        'chunk_number': chunk['chunk_number'],
                        'error': error
                    })
                    
            # 3. Prepare results
            if self.ontology_processor:
                # Merge all JSON-LD graphs into a single graph
                merged_data = {
                    "@context": self.ontology_processor.get_context()["@context"],
                    "@graph": []
                }
                for data in all_extracted_data:
                    if "@graph" in data:
                        merged_data["@graph"].extend(data["@graph"])
                
                # Normalize through RDF graph to ensure proper deduplication
                normalized_data = self._normalize_jsonld_through_rdf(merged_data)
                
                # Get statistics about the normalization
                original_count = len(merged_data["@graph"])
                final_count = len(normalized_data["@graph"])
                
                result = {
                    'jsonld': normalized_data,
                    'statistics': {
                        'total_chunks': len(chunks),
                        'processed_chunks': len(chunks) - len(failed_chunks),
                        'failed_chunks': len(failed_chunks),
                        'total_entities': final_count,
                        'original_entities': original_count,
                        'duplicates_removed': original_count - final_count
                    },
                    'failed_chunks': failed_chunks
                }
            else:
                # Original triple extraction results
                normalized_triples = self.text_processor.deduplicate_triples(all_extracted_data)
                result = {
                    'triples': normalized_triples,
                    'statistics': {
                        'total_chunks': len(chunks),
                        'processed_chunks': len(chunks) - len(failed_chunks),
                        'failed_chunks': len(failed_chunks),
                        'total_triples': len(all_extracted_data),
                        'unique_triples': len(normalized_triples)
                    },
                    'failed_chunks': failed_chunks
                }
            
            return True, result, None
            
        except Exception as e:
            return False, None, f"Pipeline error: {str(e)}"
            
    def display_results(self, result: Dict):
        """
        Display the processing results in a readable format.
        
        Args:
            result (dict): The result dictionary from process_text
        """
        if not result:
            print("No results to display")
            return
            
        # Display statistics
        stats = result['statistics']
        print("\n--- Processing Statistics ---")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Successfully processed chunks: {stats['processed_chunks']}")
        print(f"Failed chunks: {stats['failed_chunks']}")
        
        if self.ontology_processor:
            print(f"Total entities extracted: {stats['total_entities']}")
            print(f"Original entities: {stats['original_entities']}")
            print(f"Duplicates removed: {stats['duplicates_removed']}")
            print("\n--- Extracted JSON-LD ---")
            print(json.dumps(result['jsonld'], indent=2))
        else:
            print(f"Total triples extracted: {stats['total_triples']}")
            print(f"Unique triples after normalization: {stats['unique_triples']}")
            print("\n--- Extracted Triples ---")
            df = pd.DataFrame(result['triples'])
            print(df.to_string(index=False))
        
        # Display failed chunks if any
        if result['failed_chunks']:
            print("\n--- Failed Chunks ---")
            for failure in result['failed_chunks']:
                print(f"Chunk {failure['chunk_number']}: {failure['error']}") 