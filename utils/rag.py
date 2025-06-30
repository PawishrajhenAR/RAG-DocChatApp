"""
Enhanced RAG (Retrieval-Augmented Generation) System

This module provides a complete RAG pipeline for document processing and question answering.
It combines document retrieval with LLM generation for accurate, context-aware responses.
"""

from typing import List, Dict, Any, Optional
import os
import tempfile
from pathlib import Path

from .pdf_reader import EnhancedPDFReader
from .docx_reader import EnhancedDOCXReader
from .text_splitter import EnhancedTextSplitter
from .embedding import EnhancedEmbeddingIndex
from .ollama_llm import OllamaLLM

class RAGSystem:
    """
    Enhanced RAG system with complete document processing pipeline.
    
    Features:
    - Multi-format document support (PDF, DOCX, TXT)
    - Intelligent text splitting and embedding
    - Hybrid search capabilities
    - Context-aware response generation
    - Robust error handling and fallbacks
    """
    
    def __init__(self, model_name: str = "phi3"):
        """
        Initialize the RAG system.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.pdf_reader = EnhancedPDFReader()
        self.docx_reader = EnhancedDOCXReader()
        self.text_splitter = EnhancedTextSplitter()
        self.embedding_index = EnhancedEmbeddingIndex()
        self.llm = OllamaLLM(model_name=model_name)
        
        # Document storage
        self.documents = []
        self.is_indexed = False
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document and add it to the system.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processing results and metadata
        """
        print(f"DEBUG: RAGSystem.process_document called with {file_path}")
        
        if not os.path.exists(file_path):
            error_msg = f'File not found: {file_path}'
            print(f"DEBUG: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'file_name': os.path.basename(file_path) if 'file_path' in locals() else 'Unknown',
                'chunks_created': 0,
                'word_count': 0,
                'metadata': {}
            }
        
        file_extension = Path(file_path).suffix.lower()
        print(f"DEBUG: File extension: {file_extension}")
        
        try:
            # Extract text based on file type
            if file_extension == '.pdf':
                print("DEBUG: Processing as PDF")
                result = self.pdf_reader.read_pdf(file_path)
                text = result['text']
                metadata = result['metadata']
            elif file_extension == '.txt':
                print("DEBUG: Processing as TXT")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                metadata = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_type': 'txt'
                }
            elif file_extension == '.docx':
                print("DEBUG: Processing as DOCX")
                # Use EnhancedDOCXReader to properly extract text from DOCX files
                try:
                    docx_result = self.docx_reader.read_docx(file_path)
                    text = docx_result['text']
                    metadata = docx_result['metadata']
                    print(f"DEBUG: DOCX extracted {len(text)} characters, {docx_result['word_count']} words")
                    
                    # Validate that we got meaningful text
                    if not text or len(text.strip()) < 10:
                        error_msg = 'DOCX file appears to be empty or contains no readable text'
                        print(f"DEBUG: {error_msg}")
                        return {
                            'success': False,
                            'error': error_msg,
                            'file_name': os.path.basename(file_path),
                            'chunks_created': 0,
                            'word_count': 0,
                            'metadata': {}
                        }
                        
                except Exception as e:
                    error_msg = f'Error reading DOCX file: {str(e)}'
                    print(f"DEBUG: {error_msg}")
                    return {
                        'success': False,
                        'error': error_msg,
                        'file_name': os.path.basename(file_path),
                        'chunks_created': 0,
                        'word_count': 0,
                        'metadata': {}
                    }
            else:
                error_msg = f'Unsupported file type: {file_extension}'
                print(f"DEBUG: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'file_name': os.path.basename(file_path) if 'file_path' in locals() else 'Unknown',
                    'chunks_created': 0,
                    'word_count': 0,
                    'metadata': {}
                }
            
            print(f"DEBUG: Extracted text length: {len(text)} characters")
            
            if not text.strip():
                error_msg = 'No text content found in document'
                print(f"DEBUG: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'file_name': os.path.basename(file_path) if 'file_path' in locals() else 'Unknown',
                    'chunks_created': 0,
                    'word_count': 0,
                    'metadata': {}
                }
            
            # Split text into chunks
            print("DEBUG: Splitting text into chunks")
            chunks = self.text_splitter.split_text(text, metadata)
            print(f"DEBUG: Created {len(chunks)} chunks")
            
            if not chunks:
                error_msg = 'Failed to split document into chunks'
                print(f"DEBUG: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'file_name': os.path.basename(file_path) if 'file_path' in locals() else 'Unknown',
                    'chunks_created': 0,
                    'word_count': 0,
                    'metadata': {}
                }
            
            # Add to documents
            self.documents.extend(chunks)
            print(f"DEBUG: Added {len(chunks)} chunks to documents. Total documents: {len(self.documents)}")
            
            return {
                'success': True,
                'file_name': metadata.get('file_name', os.path.basename(file_path)),
                'chunks_created': len(chunks),
                'word_count': len(text.split()),
                'metadata': metadata
            }
            
        except Exception as e:
            error_msg = f'Error processing document: {str(e)}'
            print(f"DEBUG: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'file_name': os.path.basename(file_path) if 'file_path' in locals() else 'Unknown',
                'chunks_created': 0,
                'word_count': 0,
                'metadata': {}
            }
    
    def build_index(self) -> Dict[str, Any]:
        """
        Build the search index from processed documents.
        
        Returns:
            Dictionary containing indexing results
        """
        if not self.documents:
            return {
                'success': False,
                'error': 'No documents to index',
                'documents_indexed': 0,
                'total_chunks': 0
            }
        
        try:
            # Extract text content for indexing
            texts = [doc['page_content'] for doc in self.documents if doc['page_content'].strip()]
            
            if not texts:
                return {
                    'success': False,
                    'error': 'No valid text content found in documents',
                    'documents_indexed': 0,
                    'total_chunks': len(self.documents)
                }
            
            print(f"Building index with {len(texts)} text chunks...")
            
            # Build the embedding index
            self.embedding_index.build_index(texts, use_hybrid=True)
            
            self.is_indexed = True
            
            print(f"Index built successfully! {len(texts)} documents indexed.")
            
            return {
                'success': True,
                'documents_indexed': len(texts),
                'total_chunks': len(self.documents)
            }
            
        except Exception as e:
            print(f"Error building index: {str(e)}")
            return {
                'success': False,
                'error': f'Error building index: {str(e)}',
                'documents_indexed': 0,
                'total_chunks': len(self.documents) if hasattr(self, 'documents') else 0
            }
    
    def ask_question(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get an answer using the RAG system.
        
        Args:
            question: The question to ask
            max_results: Maximum number of relevant documents to retrieve
            
        Returns:
            Dictionary containing the answer and metadata
        """
        if not question.strip():
            return {
                'success': False,
                'error': 'Question cannot be empty',
                'answer': '',
                'context_docs': 0,
                'llm_success': False,
                'relevant_documents': []
            }
        
        if not self.is_indexed:
            return {
                'success': False,
                'error': 'Search index not built. Please process documents first.',
                'answer': '',
                'context_docs': 0,
                'llm_success': False,
                'relevant_documents': []
            }
        
        # Special case: if user asks about uploaded document, always summarize
        if any(kw in question.lower() for kw in ["uploaded document", "document i have uploaded", "see the document", "can you see the document"]):
            relevant_docs = self.embedding_index.documents[:max_results]
            context = self._build_context(relevant_docs)
            summary_prompt = "Summarize the uploaded document."
            print(f"DEBUG: Special case: summarizing uploaded document.")
            llm_response = self.llm.generate_response(summary_prompt, context=context)
            print(f"DEBUG: LLM response (summary): {llm_response}")
            return {
                'success': True,
                'answer': llm_response.get('response', 'No answer generated'),
                'context_docs': len(relevant_docs),
                'llm_success': llm_response.get('success', False),
                'llm_error': llm_response.get('error'),
                'relevant_documents': [
                    {
                        'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                        'score': doc.metadata.get('similarity_score', 0)
                    }
                    for doc in relevant_docs
                ]
            }
        
        # Detect if user refers to a specific document by name/type
        doc_keywords = []
        if "python" in question.lower():
            doc_keywords.append("python")
        if "resume" in question.lower():
            doc_keywords.append("resume")
        
        try:
            # Search for relevant documents
            relevant_docs = self.embedding_index.search(question, k=max_results*2)
            print(f"DEBUG: Found {len(relevant_docs)} relevant documents before filtering")
            
            # Filter by file_name/source if a specific doc is referenced
            if doc_keywords:
                filtered = []
                for doc in relevant_docs:
                    fname = doc.metadata.get('file_name', '').lower()
                    src = doc.metadata.get('source', '').lower()
                    if any(kw in fname or kw in src for kw in doc_keywords):
                        filtered.append(doc)
                if filtered:
                    relevant_docs = filtered[:max_results]
                    print(f"DEBUG: Filtered to {len(relevant_docs)} docs for keywords {doc_keywords}")
                else:
                    relevant_docs = relevant_docs[:max_results]
                    print(f"DEBUG: No docs matched keywords {doc_keywords}, using top results.")
            else:
                relevant_docs = relevant_docs[:max_results]
            
            if not relevant_docs:
                return {
                    'success': False,
                    'error': 'No relevant documents found for the question',
                    'answer': '',
                    'context_docs': 0,
                    'llm_success': False,
                    'relevant_documents': []
                }
            
            # Build context from relevant documents
            context = self._build_context(relevant_docs)
            print(f"DEBUG: Built context with {len(context)} characters")
            
            # Generate answer using LLM
            print(f"DEBUG: Calling LLM with question: '{question[:100]}...'")
            llm_response = self.llm.generate_response(question, context=context)
            print(f"DEBUG: LLM response: {llm_response}")
            
            return {
                'success': True,
                'answer': llm_response.get('response', 'No answer generated'),
                'context_docs': len(relevant_docs),
                'llm_success': llm_response.get('success', False),
                'llm_error': llm_response.get('error'),
                'relevant_documents': [
                    {
                        'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                        'score': doc.metadata.get('similarity_score', 0)
                    }
                    for doc in relevant_docs
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error generating answer: {str(e)}',
                'answer': '',
                'context_docs': 0,
                'llm_success': False,
                'relevant_documents': []
            }
    
    def _build_context(self, documents: List) -> str:
        """
        Build context string from relevant documents.
        
        Args:
            documents: List of relevant documents
            
        Returns:
            Formatted context string
        """
        print(f"DEBUG: Building context from {len(documents)} documents")
        
        if not documents:
            print("DEBUG: No documents provided for context building")
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            if content:
                context_parts.append(f"Document {i}:\n{content}")
                print(f"DEBUG: Added document {i} to context (length: {len(content)})")
            else:
                print(f"DEBUG: Document {i} has no content")
        
        context = "\n\n".join(context_parts)
        print(f"DEBUG: Built context with {len(context_parts)} parts, total length: {len(context)}")
        
        return context
    
    def clear_documents(self):
        """Clear all documents and reset the system."""
        self.documents = []
        self.embedding_index.clear_index()
        self.is_indexed = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'documents_loaded': len(self.documents),
            'is_indexed': self.is_indexed,
            'llm_connected': self.llm.get_model_info().get('available', False),
            'model_name': self.llm.model_name
        }