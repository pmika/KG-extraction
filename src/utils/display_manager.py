import json
import pandas as pd
from typing import Dict, Optional
from src.utils.logger import Logger


class DisplayManager:
    """Manager for displaying pipeline results."""
    
    @staticmethod
    def display_results(result: Dict, extraction_mode: str = "triples") -> None:
        """
        Display the processing results in a readable format.
        
        Args:
            result: The result dictionary from pipeline processing
            extraction_mode: The extraction mode used ("triples" or "jsonld")
        """
        if not result:
            Logger.warning("No results to display")
            return
        
        try:
            # Display statistics
            stats = result.get('statistics', {})
            DisplayManager._display_statistics(stats, extraction_mode)
            
            # Display extracted data
            if extraction_mode == "jsonld":
                DisplayManager._display_jsonld_results(result)
            else:
                DisplayManager._display_triple_results(result)
            
            # Display failed chunks if any
            DisplayManager._display_failed_chunks(result)
            
        except Exception as e:
            Logger.error(f"Error displaying results: {str(e)}")
    
    @staticmethod
    def _display_statistics(stats: Dict, extraction_mode: str) -> None:
        """Display processing statistics."""
        print("\n--- Processing Statistics ---")
        print(f"Total chunks: {stats.get('total_chunks', 0)}")
        print(f"Successfully processed chunks: {stats.get('processed_chunks', 0)}")
        print(f"Failed chunks: {stats.get('failed_chunks', 0)}")
        
        if extraction_mode == "jsonld":
            print(f"Total entities extracted: {stats.get('total_entities', 0)}")
            print(f"Original entities: {stats.get('original_entities', 0)}")
            print(f"Duplicates removed: {stats.get('duplicates_removed', 0)}")
        else:
            print(f"Total triples extracted: {stats.get('total_triples', 0)}")
            print(f"Unique triples after normalization: {stats.get('unique_triples', 0)}")
            print(f"Duplicates removed: {stats.get('duplicates_removed', 0)}")
    
    @staticmethod
    def _display_jsonld_results(result: Dict) -> None:
        """Display JSON-LD results."""
        print("\n--- Extracted JSON-LD ---")
        jsonld_data = result.get('jsonld', {})
        print(json.dumps(jsonld_data, indent=2))
    
    @staticmethod
    def _display_triple_results(result: Dict) -> None:
        """Display triple results."""
        print("\n--- Extracted Triples ---")
        triples = result.get('triples', [])
        if triples:
            df = pd.DataFrame(triples)
            print(df.to_string(index=False))
        else:
            print("No triples extracted.")
    
    @staticmethod
    def _display_failed_chunks(result: Dict) -> None:
        """Display information about failed chunks."""
        failed_chunks = result.get('failed_chunks', [])
        if failed_chunks:
            print("\n--- Failed Chunks ---")
            for failure in failed_chunks:
                chunk_num = failure.get('chunk_number', 'Unknown')
                error = failure.get('error', 'Unknown error')
                print(f"Chunk {chunk_num}: {error}")
    
    @staticmethod
    def display_configuration(config: Dict) -> None:
        """
        Display configuration information.
        
        Args:
            config: Configuration dictionary
        """
        print("\n--- Configuration ---")
        print(json.dumps(config, indent=2))
    
    @staticmethod
    def display_progress(current: int, total: int, message: str = "Processing") -> None:
        """
        Display progress information.
        
        Args:
            current: Current progress
            total: Total items
            message: Progress message
        """
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"\r{message}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)
        if current == total:
            print()  # New line when complete
    
    @staticmethod
    def display_error(error: str, details: Optional[str] = None) -> None:
        """
        Display error information.
        
        Args:
            error: Error message
            details: Optional error details
        """
        print(f"\n--- Error ---")
        print(f"Error: {error}")
        if details:
            print(f"Details: {details}")
    
    @staticmethod
    def display_success(message: str) -> None:
        """
        Display success message.
        
        Args:
            message: Success message
        """
        print(f"\n--- Success ---")
        print(f"{message}")
    
    @staticmethod
    def display_summary(result: Dict, extraction_mode: str) -> None:
        """
        Display a summary of the processing results.
        
        Args:
            result: Processing results
            extraction_mode: Extraction mode used
        """
        if not result:
            return
        
        stats = result.get('statistics', {})
        print("\n--- Summary ---")
        
        if extraction_mode == "jsonld":
            print(f"✅ Successfully extracted {stats.get('total_entities', 0)} entities")
            print(f"📊 Processed {stats.get('processed_chunks', 0)}/{stats.get('total_chunks', 0)} chunks")
            print(f"🗑️  Removed {stats.get('duplicates_removed', 0)} duplicates")
        else:
            print(f"✅ Successfully extracted {stats.get('unique_triples', 0)} unique triples")
            print(f"📊 Processed {stats.get('processed_chunks', 0)}/{stats.get('total_chunks', 0)} chunks")
            print(f"🗑️  Removed {stats.get('duplicates_removed', 0)} duplicates")
        
        if stats.get('failed_chunks', 0) > 0:
            print(f"❌ {stats.get('failed_chunks', 0)} chunks failed") 