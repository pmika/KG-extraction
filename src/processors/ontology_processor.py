from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from owlready2 import *
from pyld import jsonld

class OntologyProcessor:
    def __init__(self, ontology_path: Union[str, Path]):
        """
        Initialize the ontology processor with a local OWL file.
        
        Args:
            ontology_path (Union[str, Path]): Path to the local OWL file
        """
        if not isinstance(ontology_path, Path):
            ontology_path = Path(ontology_path)
            
        if not ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
            
        if ontology_path.suffix.lower() != '.owl':
            raise ValueError(f"Expected .owl file, got {ontology_path.suffix}")
            
        # Load the ontology
        self.ontology = get_ontology(str(ontology_path)).load()
        
        # Store the raw OWL content
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.owl_content = f.read()
        
        # Build JSON-LD context
        self.context = self._build_jsonld_context()
        
    def _build_jsonld_context(self) -> Dict:
        """
        Build a JSON-LD context from the ontology.
        Maps ontology terms to their URIs and provides type information.
        """
        context = {
            "@context": {
                "@vocab": str(self.ontology.base_iri),
                # Add all classes
                **{cls.name: {
                    "@id": str(cls.iri),
                    "@type": "@id"
                } for cls in self.ontology.classes()},
                # Add all object properties
                **{prop.name: {
                    "@id": str(prop.iri),
                    "@type": "@id"
                } for prop in self.ontology.object_properties()},
                # Add all data properties
                **{prop.name: {
                    "@id": str(prop.iri)
                } for prop in self.ontology.data_properties()}
            }
        }
        return context
        
    def get_context(self) -> Dict:
        """Get the JSON-LD context for the ontology."""
        return self.context
        
    def get_ontology_info(self) -> Dict:
        """
        Get basic information about the ontology structure.
        Useful for LLM prompts and validation.
        """
        return {
            "classes": [cls.name for cls in self.ontology.classes()],
            "object_properties": [prop.name for prop in self.ontology.object_properties()],
            "data_properties": [prop.name for prop in self.ontology.data_properties()],
            "base_iri": str(self.ontology.base_iri)
        }
        
    def _get_term_iri(self, term: str) -> str:
        """
        Get the full IRI for a term, whether it's already an IRI or a compacted term.
        
        Args:
            term (str): The term to expand (either a compacted term or an IRI)
            
        Returns:
            str: The full IRI for the term
        """
        if term.startswith('@'):
            return term
            
        # If it's already an IRI, return it
        if term.startswith('http://') or term.startswith('https://'):
            return term
            
        # If it's a compacted term, get its IRI from context
        if term in self.context["@context"]:
            return self.context["@context"][term]["@id"]
            
        return term

    def validate_jsonld(self, jsonld_data: Union[str, Dict]) -> bool:
        """
        Validate JSON-LD data against the ontology.
        Checks if all terms are defined in the context and follows JSON-LD syntax.
        """
        try:
            if isinstance(jsonld_data, str):
                jsonld_data = json.loads(jsonld_data)
                
            print("\nValidating JSON-LD data:")
            print(json.dumps(jsonld_data, indent=2))
                
            # Expand the JSON-LD to check for valid terms
            try:
                expanded = jsonld.expand(jsonld_data, {"expandContext": self.context})
             
            except Exception as e:
                print(f"\nJSON-LD expansion error: {str(e)}")
                return False
            
            # Get all valid IRIs from the context
            valid_iris = set()
            for term, info in self.context["@context"].items():
                if isinstance(info, dict) and "@id" in info:
                    valid_iris.add(info["@id"])
            
            # Basic validation - check if all terms are in the context
            def check_terms(obj, path=""):
                if isinstance(obj, dict):
                    for key in obj:
                        if key.startswith('@'):
                            continue
                            
                        # Get the full IRI for this term
                        term_iri = self._get_term_iri(key)
                        
                        # Check if either the compacted term or its IRI is valid
                        if (key not in self.context["@context"] and 
                            term_iri not in valid_iris):
                            print(f"\nInvalid term '{key}' (IRI: {term_iri}) at path '{path}'")
                            print(f"Available terms: {list(self.context['@context'].keys())}")
                            print(f"Available IRIs: {sorted(list(valid_iris))}")
                            return False
                            
                        if isinstance(obj[key], (dict, list)):
                            if not check_terms(obj[key], f"{path}.{key}"):
                                return False
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if not check_terms(item, f"{path}[{i}]"):
                            return False
                return True
                
            is_valid = check_terms(expanded)
            if not is_valid:
                print("\nJSON-LD validation failed")
            return is_valid
            
        except Exception as e:
            print(f"\nJSON-LD validation error: {str(e)}")
            return False
            
    def normalize_jsonld(self, jsonld_data: Union[str, Dict]) -> Optional[Dict]:
        """
        Normalize JSON-LD data by expanding and compacting it.
        Ensures consistent representation and removes redundant information.
        """
        try:
            if isinstance(jsonld_data, str):
                jsonld_data = json.loads(jsonld_data)
                
            # Expand the JSON-LD
            expanded = jsonld.expand(jsonld_data, {"expandContext": self.context})
            
            # Compact it back using our context
            compacted = jsonld.compact(expanded, self.context)
            
            return compacted
            
        except Exception as e:
            print(f"JSON-LD normalization error: {str(e)}")
            return None

    def get_owl_content(self) -> str:
        """Return the raw OWL ontology as a string."""
        return self.owl_content 