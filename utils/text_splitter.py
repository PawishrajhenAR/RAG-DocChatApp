"""
Enhanced Text Splitter

This module provides intelligent text splitting capabilities for RAG applications.
It handles various document types and ensures optimal chunk sizes for embedding.
"""

from typing import List, Dict, Any, Optional
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EnhancedTextSplitter:
    """
    Enhanced text splitter with intelligent chunking for RAG applications.
    
    Features:
    - Recursive character-based splitting
    - Intelligent chunk size optimization
    - Metadata preservation
    - Overlap management for context continuity
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Target size for each text chunk
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for intelligent splitting
        if separators is None:
            self.separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                "! ",    # Exclamations
                "? ",    # Questions
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Words
                ""       # Characters
            ]
        else:
            self.separators = separators
        
        # Initialize the underlying splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The text to split
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        if not cleaned_text:
            return []
        
        try:
            # Split the text
            chunks = self.splitter.split_text(cleaned_text)
            
            # Create result with metadata
            result = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                }
                
                # Add original metadata if provided
                if metadata:
                    chunk_metadata.update(metadata)
                
                # Always include file_name and source for filtering
                chunk_metadata['file_name'] = metadata.get('file_name', 'unknown') if metadata else 'unknown'
                chunk_metadata['source'] = metadata.get('file_name', 'unknown') if metadata else 'unknown'
                
                result.append({
                    'page_content': chunk.strip(),
                    'metadata': chunk_metadata
                })
            
            return result
            
        except Exception as e:
            print(f"Error splitting text: {e}")
            # Fallback: return the original text as a single chunk
            return [{
                'page_content': cleaned_text,
                'metadata': metadata or {}
            }]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better splitting.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove excessive periods
        text = re.sub(r'\.{3,}', '...', text)
        
        # Remove excessive spaces around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of documents with 'page_content' and optional 'metadata'
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            if not doc or 'page_content' not in doc:
                continue
                
            text = doc['page_content']
            metadata = doc.get('metadata', {})
            
            chunks = self.split_text(text, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_optimal_chunk_size(self, text: str) -> int:
        """
        Calculate optimal chunk size based on text characteristics.
        
        Args:
            text: The text to analyze
            
        Returns:
            Recommended chunk size
        """
        if not text:
            return self.chunk_size
        
        # Analyze text characteristics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Adjust chunk size based on text characteristics
        if avg_sentence_length > 50:
            # Long sentences - use larger chunks
            return min(1500, self.chunk_size + 200)
        elif avg_sentence_length < 10:
            # Short sentences - use smaller chunks
            return max(500, self.chunk_size - 200)
        else:
            return self.chunk_size

