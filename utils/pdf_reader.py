"""
Enhanced PDF Reader

This module provides robust PDF text extraction capabilities for RAG applications.
It handles various PDF formats and ensures high-quality text extraction.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

class EnhancedPDFReader:
    """
    High-level, robust PDF and TXT reader for RAG applications.
    - Preserves page order
    - Handles multi-page, multi-column, and multi-format
    - Returns per-page and full-document text
    - Extracts and returns metadata
    - Robust error handling and cleaning
    """
    def __init__(self):
        self.available_libraries = []
        if PYMUPDF_AVAILABLE:
            self.available_libraries.append('pymupdf')
        if PYPDF2_AVAILABLE:
            self.available_libraries.append('pypdf2')
        if not self.available_libraries:
            raise ImportError("No PDF processing libraries available. Install PyMuPDF or PyPDF2.")

    def read(self, file_path: str) -> Dict[str, Any]:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return self.read_pdf(file_path)
        elif ext == '.txt':
            return self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def read_pdf(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        # Try PyMuPDF first
        if 'pymupdf' in self.available_libraries:
            try:
                return self._extract_with_pymupdf(file_path)
            except Exception as e:
                print(f"PyMuPDF extraction failed: {e}")
        # Fallback to PyPDF2
        if 'pypdf2' in self.available_libraries:
            try:
                return self._extract_with_pypdf2(file_path)
            except Exception as e:
                print(f"PyPDF2 extraction failed: {e}")
        raise Exception("All PDF extraction methods failed")

    def read_txt(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TXT file not found: {file_path}")
        # Try utf-8, fallback to latin-1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
        cleaned_text = self._clean_text(text)
        # Split into pseudo-pages if form feed or large gaps
        pages = self._split_txt_to_pages(cleaned_text)
        page_objs = [
            {'page_number': i+1, 'content': p, 'word_count': len(p.split())}
            for i, p in enumerate(pages) if p.strip()
        ]
        return {
            'text': '\n\n'.join(pages),
            'metadata': {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'total_pages': len(page_objs),
                'extraction_method': 'txt'
            },
            'pages': page_objs,
            'word_count': len(cleaned_text.split()),
            'page_count': len(page_objs)
        }

    def _split_txt_to_pages(self, text: str) -> List[str]:
        # Split on form feed or 3+ newlines as pseudo-pages
        if '\f' in text:
            return [p.strip() for p in text.split('\f') if p.strip()]
        else:
            return [p.strip() for p in re.split(r'\n{3,}', text) if p.strip()]

    def _extract_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        doc = fitz.open(file_path)
        text_content = []
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'total_pages': len(doc),
            'extraction_method': 'pymupdf'
        }
        # Try to extract document metadata
        try:
            meta = doc.metadata
            for k, v in meta.items():
                if v and k not in metadata:
                    metadata[k] = v
        except Exception:
            pass
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Try both text and blocks for multi-column
            text = page.get_text()
            if not text.strip():
                # Try blocks
                blocks = page.get_text("blocks")
                text = '\n'.join([b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)])
            cleaned_text = self._clean_text(text)
            if cleaned_text:
                text_content.append({
                    'page_number': page_num + 1,
                    'content': cleaned_text,
                    'word_count': len(cleaned_text.split())
                })
        doc.close()
        full_text = '\n\n'.join([p['content'] for p in text_content])
        return {
            'text': full_text,
            'metadata': metadata,
            'pages': text_content,
            'word_count': len(full_text.split()),
            'page_count': len(text_content)
        }

    def _extract_with_pypdf2(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_content = []
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'total_pages': len(reader.pages),
                'extraction_method': 'pypdf2'
            }
            # Try to extract document metadata
            try:
                meta = reader.metadata
                if meta:
                    for k, v in meta.items():
                        if v and k not in metadata:
                            metadata[k] = v
            except Exception:
                pass
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                cleaned_text = self._clean_text(text)
                if cleaned_text:
                    text_content.append({
                        'page_number': page_num + 1,
                        'content': cleaned_text,
                        'word_count': len(cleaned_text.split())
                    })
            full_text = '\n\n'.join([p['content'] for p in text_content])
            return {
                'text': full_text,
                'metadata': metadata,
                'pages': text_content,
                'word_count': len(full_text.split()),
                'page_count': len(text_content)
            }

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path)
        }
        
        # Try to extract PDF-specific metadata
        if 'pymupdf' in self.available_libraries:
            try:
                doc = fitz.open(file_path)
                pdf_metadata = doc.metadata
                doc.close()
                
                # Add PDF metadata
                for key, value in pdf_metadata.items():
                    if value:
                        metadata[f'pdf_{key}'] = value
                        
            except Exception as e:
                print(f"Error extracting PDF metadata: {e}")
        
        return metadata