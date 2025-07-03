import pymupdf4llm
from pathlib import Path
from typing import Union, Optional, List, Dict

class TextProcessor:
    def __init__(self, chunk_size=2000, overlap=100):
        """
        Initialize the text processor with chunking parameters.
        
        Args:
            chunk_size (int): Maximum number of words per chunk
            overlap (int): Number of words to overlap between chunks
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if overlap < 0:
            raise ValueError("Overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be smaller than chunk size ({chunk_size})")
            
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: Union[str, Path], pages: Optional[list] = None) -> str:
        """
        Extract text from a PDF file using PyMuPDF4LLM.
        
        Args:
            pdf_path (Union[str, Path]): Path to the PDF file
            pages (Optional[list]): List of page numbers to extract (0-based). If None, extracts all pages.
            
        Returns:
            str: Extracted text in markdown format
        """
        try:
            # Convert to markdown format which preserves document structure
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path), pages=pages)
            return markdown_text
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    def process_text(self, text: str) -> List[Dict[str, Union[str, int]]]:
        """
        Process text by splitting it into chunks.
        
        Args:
            text (str): The text to process
            
        Returns:
            List[Dict[str, Union[str, int]]]: List of dictionaries containing chunk text and chunk number
        """
        return self.split_into_chunks(text)

    def process_pdf(self, pdf_path: Union[str, Path], pages: Optional[list] = None) -> List[Dict[str, Union[str, int]]]:
        """
        Process a PDF file by extracting its text and splitting it into chunks.
        
        Args:
            pdf_path (Union[str, Path]): Path to the PDF file
            pages (Optional[list]): List of page numbers to extract (0-based). If None, extracts all pages.
            
        Returns:
            List[Dict[str, Union[str, int]]]: List of dictionaries containing chunk text and chunk number
        """
        text = self.extract_text_from_pdf(pdf_path, pages=pages)
        return self.process_text(text)

    def split_into_chunks(self, text):
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): The text to split
            
        Returns:
            list: List of dictionaries containing chunk text and chunk number
        """
        words = text.split()
        total_words = len(words)
        chunks = []
        start_index = 0
        chunk_number = 1
        
        while start_index < total_words:
            end_index = min(start_index + self.chunk_size, total_words)
            chunk_text = " ".join(words[start_index:end_index])
            chunks.append({
                "text": chunk_text,
                "chunk_number": chunk_number
            })
            
            # Calculate the start of the next chunk
            next_start_index = start_index + self.chunk_size - self.overlap
            
            # Ensure progress is made
            if next_start_index <= start_index:
                if end_index == total_words:
                    break
                next_start_index = start_index + 1
                
            start_index = next_start_index
            chunk_number += 1
            
            # Safety break
            if chunk_number > total_words:
                break
                
        return chunks

    def normalize_triple(self, triple):
        """
        Normalize a triple by cleaning and standardizing its components.
        
        Args:
            triple (dict): Dictionary containing subject, predicate, and object
            
        Returns:
            dict: Normalized triple or None if invalid
        """
        if not all(k in triple for k in ['subject', 'predicate', 'object']):
            return None
            
        subject = triple.get('subject', '').strip().lower()
        predicate = triple.get('predicate', '').strip().lower()
        object_ = triple.get('object', '').strip().lower()
        
        # Remove extra whitespace in predicate
        predicate = ' '.join(predicate.split())
        
        if not all([subject, predicate, object_]):
            return None
            
        return {
            'subject': subject,
            'predicate': predicate,
            'object': object_,
            'source_chunk': triple.get('chunk', 'unknown')
        }

    def deduplicate_triples(self, triples):
        """
        Remove duplicate triples while preserving source chunk information.
        
        Args:
            triples (list): List of triple dictionaries
            
        Returns:
            list: List of unique triples
        """
        seen_triples = set()
        unique_triples = []
        
        for triple in triples:
            normalized = self.normalize_triple(triple)
            if normalized:
                triple_key = (normalized['subject'], normalized['predicate'], normalized['object'])
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    unique_triples.append(normalized)
                    
        return unique_triples 