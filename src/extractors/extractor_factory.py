from typing import Union
from src.extractors.base_extractor import BaseExtractor
from src.extractors.triple_extractor import TripleExtractor
from src.extractors.jsonld_extractor import JSONLDExtractor
from src.models.base_llm_client import BaseLLMClient
from src.config.configuration import Configuration


class ExtractorFactory:
    """Factory for creating appropriate extractors."""
    
    @staticmethod
    def create_extractor(
        extraction_mode: str,
        llm_client: BaseLLMClient,
        config: Configuration
    ) -> BaseExtractor:
        """
        Create an extractor based on the extraction mode.
        
        Args:
            extraction_mode: "triples" or "jsonld"
            llm_client: LLM client for making API calls
            config: Configuration settings
            
        Returns:
            Appropriate extractor instance
            
        Raises:
            ValueError: If extraction mode is not supported
        """
        if extraction_mode == "triples":
            return TripleExtractor(llm_client, config)
        elif extraction_mode == "jsonld":
            return JSONLDExtractor(llm_client, config)
        else:
            raise ValueError(f"Unsupported extraction mode: {extraction_mode}")
    
    @staticmethod
    def create_extractor_from_config(
        llm_client: BaseLLMClient,
        config: Configuration
    ) -> BaseExtractor:
        """
        Create an extractor using the extraction mode from configuration.
        
        Args:
            llm_client: LLM client for making API calls
            config: Configuration settings
            
        Returns:
            Appropriate extractor instance
        """
        return ExtractorFactory.create_extractor(
            config.extraction.extraction_mode,
            llm_client,
            config
        ) 