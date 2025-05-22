from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import unified_diff

from src.pipeline import KnowledgeGraphPipeline
from src.config.settings import LLM_PROVIDER

@dataclass
class EvaluationConfig:
    """Configuration for pipeline evaluation."""
    # LLM Configuration
    llm_provider: str  # "openai" or "anthropic"
    system_prompt: str
    user_prompt: str
    
    # Text Processing Configuration
    chunk_size: int
    chunk_overlap: int
    
    # Input Text
    input_text: str
    
    # Additional Parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    model_name: Optional[str] = None  # e.g., "gpt-4" or "claude-3-opus"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for storage."""
        return {
            "llm_provider": self.llm_provider,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_name": self.model_name
        }

class PipelineEvaluator:
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the pipeline evaluator.
        
        Args:
            output_dir: Directory to store evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def evaluate_config(self, config: EvaluationConfig) -> Dict:
        """
        Evaluate a single configuration.
        
        Args:
            config: Evaluation configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\nInitializing pipeline with provider: {config.llm_provider}")
        print(f"Model name: {config.model_name}")
        print(f"Temperature: {config.temperature}")
        
        # Initialize pipeline with configuration
        pipeline = KnowledgeGraphPipeline(
            llm_provider=config.llm_provider,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        # Process text and collect metrics
        start_time = datetime.now()
        success, result, error = pipeline.process_text(config.input_text)
        end_time = datetime.now()
        
        if not success:
            print(f"Error processing text: {error}")
        
        # Prepare evaluation results
        eval_result = {
            "config": config.to_dict(),
            "success": success,
            "error": error,
            "processing_time": (end_time - start_time).total_seconds(),
            "results": result if success else None
        }
        
        self.results.append(eval_result)
        return eval_result
    
    def compare_configurations(self, configs: List[EvaluationConfig]) -> pd.DataFrame:
        """
        Compare multiple configurations.
        
        Args:
            configs: List of configurations to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for config in configs:
            result = self.evaluate_config(config)
            if result["success"]:
                stats = result["results"]["statistics"]
                comparison_data.append({
                    "llm_provider": config.llm_provider,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "processing_time": result["processing_time"],
                    "total_triples": stats["total_triples"],
                    "unique_triples": stats["unique_triples"],
                    "success_rate": stats["processed_chunks"] / stats["total_chunks"]
                })
        
        return pd.DataFrame(comparison_data)
    
    def save_results(self, filename: Optional[str] = None) -> None:
        """
        Save evaluation results to file.
        
        Args:
            filename: Optional custom filename
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {
                "config": result["config"],
                "success": result["success"],
                "error": result["error"],
                "processing_time": result["processing_time"]
            }
            if result["results"]:
                serializable_result["results"] = {
                    "statistics": result["results"]["statistics"],
                    "failed_chunks": result["results"]["failed_chunks"]
                }
            serializable_results.append(serializable_result)
            
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
            
    def plot_comparison(self, comparison_df: pd.DataFrame, metric: str) -> None:
        """
        Create visualization of comparison results.
        
        Args:
            comparison_df: DataFrame with comparison results
            metric: Metric to visualize (e.g., "processing_time", "total_triples")
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=comparison_df, x="llm_provider", y=metric)
        plt.title(f"Comparison of {metric} across LLM providers")
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f"comparison_{metric}.png"
        plt.savefig(output_path)
        plt.close()

    def compare_triples(self, config1: EvaluationConfig, config2: EvaluationConfig) -> str:
        """
        Compare the output triples between two configurations and display differences in a git-diff-like format.
        
        Args:
            config1: First configuration to compare
            config2: Second configuration to compare
            
        Returns:
            String containing the diff output in a git-diff-like format
        """
        # Run both configurations
        result1 = self.evaluate_config(config1)
        result2 = self.evaluate_config(config2)
        
        if not result1["success"] or not result2["success"]:
            return "Error: One or both configurations failed to process"
        
        # Extract triples from results
        print("\nExtracting triples from results...")
        print(f"Config 1 raw triples: {result1['results']['triples']}")
        print(f"Config 2 raw triples: {result2['results']['triples']}")
        
        # Convert triples to comparable format (excluding source_chunk)
        def make_comparable(triple):
            return (triple['subject'].lower(), triple['predicate'].lower(), triple['object'].lower())
            
        triples1 = set(make_comparable(triple) for triple in result1["results"]["triples"])
        triples2 = set(make_comparable(triple) for triple in result2["results"]["triples"])
        
        print(f"\nConverted to sets:")
        print(f"Config 1 set size: {len(triples1)}")
        print(f"Config 2 set size: {len(triples2)}")
        
        # Convert triples to sorted lists for diff
        triples1_list = sorted(str(triple) for triple in triples1)
        triples2_list = sorted(str(triple) for triple in triples2)
        
        print(f"\nConverted to sorted lists:")
        print(f"Config 1 list size: {len(triples1_list)}")
        print(f"Config 2 list size: {len(triples2_list)}")
        
        # Generate diff
        diff = list(unified_diff(
            triples1_list,
            triples2_list,
            fromfile=f"Config 1 ({config1.llm_provider})",
            tofile=f"Config 2 ({config2.llm_provider})",
            lineterm=""
        ))
        
        # Add summary statistics
        summary = [
            f"\nSummary:",
            f"Config 1 ({config1.llm_provider}): {len(triples1)} triples",
            f"Config 2 ({config2.llm_provider}): {len(triples2)} triples",
            f"Unique to Config 1: {len(triples1 - triples2)} triples",
            f"Unique to Config 2: {len(triples2 - triples1)} triples",
            f"Common triples: {len(triples1 & triples2)} triples"
        ]
        
        return "\n".join(diff + summary) 