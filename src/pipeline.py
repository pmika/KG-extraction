from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from src.config.configuration import Configuration
from src.models.openai_client import OpenAIClient
from src.models.anthropic_client import AnthropicClient
from src.processors.text_processor import TextProcessor
from src.extractors.extractor_factory import ExtractorFactory
from src.utils.logger import Logger
from src.utils.display_manager import DisplayManager
from src.config.settings import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
    JSONLD_SYSTEM_PROMPT,
    JSONLD_USER_PROMPT_TEMPLATE
)


class KnowledgeGraphPipeline:
    """Simplified knowledge graph extraction pipeline using modular components."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """
        Initialize the knowledge graph extraction pipeline.
        
        Args:
            config: Configuration settings. If None, loads from environment.
        """
        # Load configuration
        if config is None:
            config = Configuration.from_env()
        
        self.config = config
        
        # Configure logging
        if self.config.enable_logging:
            Logger.configure(level=self.config.log_level)
        
        # Initialize components
        self._initialize_llm_client()
        self._initialize_text_processor()
        self._initialize_extractor()
        
        Logger.info("Knowledge Graph Pipeline initialized successfully")
        Logger.info(f"Using {self.config.extraction.extraction_mode} extraction mode")
    
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client."""
        llm_config = self.config.llm
        
        # Determine system and user prompts based on extraction mode
        if self.config.extraction.extraction_mode == "jsonld":
            system_prompt = JSONLD_SYSTEM_PROMPT
            user_prompt_template = JSONLD_USER_PROMPT_TEMPLATE
        else:
            system_prompt = EXTRACTION_SYSTEM_PROMPT
            user_prompt_template = EXTRACTION_USER_PROMPT_TEMPLATE
        
        # Create LLM client
        if llm_config.provider == "openai":
            self.llm_client = OpenAIClient(
                model_name=llm_config.model_name,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template
            )
        elif llm_config.provider == "anthropic":
            self.llm_client = AnthropicClient(
                model_name=llm_config.model_name,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
        
        Logger.info(f"Initialized {llm_config.provider} client with model {llm_config.model_name}")
    
    def _initialize_text_processor(self):
        """Initialize the text processor."""
        self.text_processor = TextProcessor(
            chunk_size=self.config.text_processing.chunk_size,
            overlap=self.config.text_processing.chunk_overlap
        )
        Logger.info(f"Initialized text processor with chunk size {self.config.text_processing.chunk_size}")
    
    def _initialize_extractor(self):
        """Initialize the appropriate extractor."""
        self.extractor = ExtractorFactory.create_extractor_from_config(
            self.llm_client,
            self.config
        )
        Logger.info(f"Initialized {self.config.extraction.extraction_mode} extractor")
    
    def process_text(self, text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Process text through the pipeline.
        
        Args:
            text: The input text to process
            
        Returns:
            tuple: (success, result, error_message)
        """
        try:
            Logger.info("Starting text processing")
            
            # 1. Split text into chunks
            chunks = self.text_processor.split_into_chunks(text)
            if not chunks:
                error_msg = "No chunks were created from the input text"
                Logger.error(error_msg)
                return False, None, error_msg
            
            Logger.info(f"Created {len(chunks)} chunks for processing")
            
            # 2. Process each chunk
            all_extracted_data = []
            failed_chunks = []
            
            for i, chunk in enumerate(chunks):
                Logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                success, data, error = self.extractor.extract_from_chunk(chunk)
                
                if success:
                    all_extracted_data.append(data)
                else:
                    failed_chunks.append({
                        'chunk_number': chunk['chunk_number'],
                        'error': error
                    })
            
            # 3. Process results
            result = self.extractor.process_results(all_extracted_data, failed_chunks)
            
            Logger.info("Text processing completed successfully")
            return True, result, None
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            Logger.error(error_msg)
            return False, None, error_msg
    
    def process_pdf(self, pdf_path: Union[str, Path], pages: Optional[List[int]] = None) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Process a PDF file through the pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            pages: Optional list of page numbers to process
            
        Returns:
            tuple: (success, result, error_message)
        """
        try:
            Logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text from PDF
            text = self.text_processor.extract_text_from_pdf(pdf_path, pages)
            
            # Process the extracted text
            return self.process_text(text)
            
        except Exception as e:
            error_msg = f"PDF processing error: {str(e)}"
            Logger.error(error_msg)
            return False, None, error_msg
    
    def display_results(self, result: Dict) -> None:
        """
        Display the processing results.
        
        Args:
            result: The result dictionary from process_text
        """
        DisplayManager.display_results(result, self.config.extraction.extraction_mode)
    
    def display_summary(self, result: Dict) -> None:
        """
        Display a summary of the processing results.
        
        Args:
            result: The result dictionary from process_text
        """
        DisplayManager.display_summary(result, self.config.extraction.extraction_mode)
    
    def get_configuration(self) -> Dict:
        """
        Get the current configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.to_dict()
    
    def display_configuration(self) -> None:
        """Display the current configuration."""
        DisplayManager.display_configuration(self.get_configuration()) 