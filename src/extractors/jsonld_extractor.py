import json
from typing import Dict, List, Optional, Tuple, Union
from src.extractors.base_extractor import BaseExtractor
from src.models.base_llm_client import BaseLLMClient
from src.config.configuration import Configuration
from src.processors.ontology_processor import OntologyProcessor
from src.utils.logger import Logger
from rdflib import Graph
from rdflib.plugins.serializers.jsonld import from_rdf
from rdflib.plugins.parsers.jsonld import to_rdf


class JSONLDExtractor(BaseExtractor):
    """Extractor for JSON-LD structured data."""
    
    def __init__(self, llm_client: BaseLLMClient, config: Configuration):
        """
        Initialize the JSON-LD extractor.
        
        Args:
            llm_client: LLM client for making API calls
            config: Configuration settings
        """
        super().__init__(llm_client, config)
        
        # Initialize ontology processor
        if not config.extraction.ontology_path:
            raise ValueError("Ontology path is required for JSON-LD extraction")
        
        self.ontology_processor = OntologyProcessor(config.extraction.ontology_path)
        self.ontology_info = self.ontology_processor.get_ontology_info()
        self.ontology_context = self.ontology_processor.get_context()
    
    def extract_from_chunk(self, chunk: Dict[str, Union[str, int]]) -> Tuple[bool, Dict, Optional[str]]:
        """
        Extract JSON-LD from a text chunk.
        
        Args:
            chunk: Dictionary containing chunk text and number
            
        Returns:
            tuple: (success, jsonld_data, error_message)
        """
        try:
            Logger.info(f"Processing chunk {chunk['chunk_number']} for JSON-LD extraction")
            
            # Format user and system prompts with ontology information
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
            system_prompt = self.llm_client.system_prompt.format(**prompt_vars)
            
            # Debug: Print the exact prompts sent to the LLM
            print(f"\n{'='*80}")
            print(f"EXACT PROMPTS SENT TO LLM FOR CHUNK {chunk['chunk_number']}")
            print(f"{'='*80}")
            print(f"\nSYSTEM PROMPT:")
            print(f"{'='*40}")
            print(system_prompt)
            print(f"\n{'='*40}")
            print(f"USER PROMPT:")
            print(f"{'='*40}")
            print(user_prompt)
            print(f"\n{'='*80}")
            
            # Set the formatted system prompt on the client (if needed)
            self.llm_client.system_prompt = system_prompt
            
            # Extract JSON-LD using LLM client
            success, data, error = self.llm_client.extract_triples(user_prompt, chunk['chunk_number'])
            
            if success:
                # Process the extracted data
                processed_data = self._process_extracted_data(data, chunk['chunk_number'])
                if processed_data:
                    return True, processed_data, None
                else:
                    error_msg = f"Failed to process JSON-LD data from chunk {chunk['chunk_number']}"
                    Logger.warning(error_msg)
                    return False, {}, error_msg
            else:
                Logger.error(f"Failed to extract JSON-LD from chunk {chunk['chunk_number']}: {error}")
                return False, {}, error
                
        except Exception as e:
            error_msg = f"Error processing chunk {chunk['chunk_number']}: {str(e)}"
            Logger.error(error_msg)
            return False, {}, error_msg
    
    def _fix_llm_context(self, jsonld_data: Dict) -> Dict:
        """
        Fix the LLM's context by replacing it with the correct ontology context.
        The LLM often generates its own context that doesn't match our ontology.
        
        Args:
            jsonld_data: JSON-LD data with potentially incorrect context
            
        Returns:
            JSON-LD data with corrected context
        """
        # Get our correct context
        correct_context = self.ontology_processor.get_context()
        
        # Replace the context
        fixed_data = jsonld_data.copy()
        fixed_data["@context"] = correct_context["@context"]
        
        print(f"Fixed LLM context - replaced with correct ontology context")
        return fixed_data
    
    def _process_extracted_data(self, data: Union[Dict, str], chunk_number: int) -> Optional[Dict]:
        """
        Process extracted JSON-LD data.
        
        Args:
            data: Raw extracted data
            chunk_number: Chunk number for tracking
            
        Returns:
            Processed JSON-LD data or None if processing failed
        """
        try:
            # Debug: Print the raw data
            print(f"\nRaw LLM response for chunk {chunk_number}:")
            print(f"Type: {type(data)}")
            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
                if "@context" in data:
                    print(f"Context: {data['@context']}")
            elif isinstance(data, str):
                print(f"String length: {len(data)}")
                print(f"First 200 chars: {data[:200]}")
            
            # Handle different data formats
            if isinstance(data, dict) and "@graph" in data:
                # Fix the context first
                fixed_data = self._fix_llm_context(data)
                
                # Direct JSON-LD response
                if self._validate_jsonld(fixed_data):
                    normalized = self._normalize_jsonld(fixed_data)
                    if normalized:
                        Logger.info(f"Successfully extracted and validated JSON-LD from chunk {chunk_number}")
                        return normalized
            elif isinstance(data, str):
                # Try to parse as JSON-LD if it's a string
                try:
                    json_data = json.loads(data)
                    if "@graph" in json_data:
                        # Fix the context first
                        fixed_data = self._fix_llm_context(json_data)
                        
                        if self._validate_jsonld(fixed_data):
                            normalized = self._normalize_jsonld(fixed_data)
                            if normalized:
                                Logger.info(f"Successfully extracted and validated JSON-LD from chunk {chunk_number}")
                                return normalized
                except json.JSONDecodeError:
                    Logger.warning(f"Invalid JSON in chunk {chunk_number}")
            
            Logger.warning(f"Unexpected data format in chunk {chunk_number}")
            return None
            
        except Exception as e:
            Logger.error(f"Error processing JSON-LD in chunk {chunk_number}: {str(e)}")
            return None
    
    def _validate_jsonld(self, jsonld_data: Dict) -> bool:
        """
        Validate JSON-LD data against the ontology.
        
        Args:
            jsonld_data: JSON-LD data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.config.extraction.enable_validation:
            return True
        
        return self.ontology_processor.validate_jsonld(jsonld_data)
    
    def _normalize_jsonld(self, jsonld_data: Dict) -> Optional[Dict]:
        """
        Normalize JSON-LD data.
        
        Args:
            jsonld_data: JSON-LD data to normalize
            
        Returns:
            Normalized JSON-LD data or None if normalization failed
        """
        if not self.config.extraction.enable_normalization:
            return jsonld_data
        
        return self.ontology_processor.normalize_jsonld(jsonld_data)
    
    def _normalize_jsonld_through_rdf(self, jsonld_data: Dict) -> Dict:
        """
        Normalize JSON-LD data by converting it to an RDF graph and back.
        This ensures proper deduplication and RDF semantics.
        
        Args:
            jsonld_data: The JSON-LD data to normalize
            
        Returns:
            Normalized JSON-LD data
        """
        # Create a new RDF graph
        g = Graph()
        
        # Add the JSON-LD data to the graph
        g.parse(data=json.dumps(jsonld_data), format='json-ld')
        
        # Convert back to JSON-LD using the ontology's context
        context = self.ontology_processor.get_context()["@context"]
        normalized = from_rdf(g, context)
        
        return normalized
    
    def process_results(self, all_extracted_data: List[Dict], failed_chunks: List[Dict]) -> Dict:
        """
        Process and combine all extracted JSON-LD data.
        
        Args:
            all_extracted_data: List of JSON-LD data from all chunks
            failed_chunks: List of failed chunks with error information
            
        Returns:
            Dictionary containing processed results and statistics
        """
        try:
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
            
            stats = {
                'total_chunks': len(all_extracted_data) + len(failed_chunks),
                'processed_chunks': len(all_extracted_data),
                'failed_chunks': len(failed_chunks),
                'total_entities': final_count,
                'original_entities': original_count,
                'duplicates_removed': original_count - final_count
            }
            
            Logger.info(f"Processed {stats['total_entities']} entities, {stats['duplicates_removed']} duplicates removed")
            
            return {
                'jsonld': normalized_data,
                'statistics': stats,
                'failed_chunks': failed_chunks
            }
            
        except Exception as e:
            Logger.error(f"Error processing JSON-LD results: {str(e)}")
            return {
                'jsonld': {"@context": self.ontology_processor.get_context()["@context"], "@graph": []},
                'statistics': {
                    'total_chunks': len(all_extracted_data) + len(failed_chunks),
                    'processed_chunks': 0,
                    'failed_chunks': len(all_extracted_data) + len(failed_chunks),
                    'total_entities': 0,
                    'original_entities': 0,
                    'duplicates_removed': 0
                },
                'failed_chunks': failed_chunks + [{'error': f'Processing error: {str(e)}'}]
            }
    
    def validate_data(self, data: Dict) -> bool:
        """
        Validate extracted JSON-LD data.
        
        Args:
            data: JSON-LD data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not isinstance(data, dict):
            return False
        
        # Check for required JSON-LD structure
        if "@graph" not in data:
            return False
        
        if not isinstance(data["@graph"], list):
            return False
        
        return True 