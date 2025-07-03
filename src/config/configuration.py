from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import os
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    provider: str = "openai"
    model_name: str = "gpt-4-turbo"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    def __post_init__(self):
        """Validate LLM configuration."""
        if self.provider not in ["openai", "anthropic"]:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")
        if self.max_tokens <= 0:
            raise ValueError(f"Max tokens must be positive, got {self.max_tokens}")


@dataclass
class TextProcessingConfig:
    """Configuration for text processing."""
    chunk_size: int = 2000
    chunk_overlap: int = 100
    
    def __post_init__(self):
        """Validate text processing configuration."""
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"Chunk overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be smaller than chunk size ({self.chunk_size})")


@dataclass
class ExtractionConfig:
    """Configuration for extraction settings."""
    extraction_mode: str = "triples"  # "triples" or "jsonld"
    ontology_path: Optional[Union[str, Path]] = None
    enable_validation: bool = True
    enable_normalization: bool = True
    
    def __post_init__(self):
        """Validate extraction configuration."""
        if self.extraction_mode not in ["triples", "jsonld"]:
            raise ValueError(f"Unsupported extraction mode: {self.extraction_mode}")
        if self.extraction_mode == "jsonld" and not self.ontology_path:
            raise ValueError("Ontology path is required for JSON-LD extraction mode")


@dataclass
class Configuration:
    """Main configuration class for the knowledge graph pipeline."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output_dir: Optional[Union[str, Path]] = None
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {self.log_level}")
    
    @classmethod
    def from_env(cls, env_file: Optional[Union[str, Path]] = None) -> 'Configuration':
        """Create configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Determine provider and set appropriate default model
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # Set default model based on provider
        default_models = {
            "openai": "gpt-4-turbo",
            "anthropic": "claude-3-5-sonnet-20241022"  # Updated to a more current model
        }
        default_model = default_models.get(provider, "gpt-4-turbo")
        
        # Load LLM configuration from environment
        llm_config = LLMConfig(
            provider=provider,
            model_name=os.getenv("LLM_MODEL_NAME", default_model),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            api_key=os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("ANTHROPIC_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE")
        )
        
        # Load text processing configuration
        text_config = TextProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "2000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100"))
        )
        
        # Load extraction configuration
        extraction_config = ExtractionConfig(
            extraction_mode=os.getenv("EXTRACTION_MODE", "triples"),
            ontology_path=os.getenv("ONTOLOGY_PATH"),
            enable_validation=os.getenv("ENABLE_VALIDATION", "true").lower() == "true",
            enable_normalization=os.getenv("ENABLE_NORMALIZATION", "true").lower() == "true"
        )
        
        return cls(
            llm=llm_config,
            text_processing=text_config,
            extraction=extraction_config,
            output_dir=os.getenv("OUTPUT_DIR"),
            enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model_name": self.llm.model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "api_key": "***" if self.llm.api_key else None,
                "api_base": self.llm.api_base
            },
            "text_processing": {
                "chunk_size": self.text_processing.chunk_size,
                "chunk_overlap": self.text_processing.chunk_overlap
            },
            "extraction": {
                "extraction_mode": self.extraction.extraction_mode,
                "ontology_path": str(self.extraction.ontology_path) if self.extraction.ontology_path else None,
                "enable_validation": self.extraction.enable_validation,
                "enable_normalization": self.extraction.enable_normalization
            },
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        return f"Configuration({config_dict})" 