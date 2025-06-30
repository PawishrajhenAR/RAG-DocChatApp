"""
Enhanced Embedding with Hybrid Search

This module implements a hybrid search system that combines:
- Dense vector embeddings (using Sentence Transformers and FAISS)
- Sparse TF-IDF representations (using scikit-learn)

Key features:
- Efficient similarity search with hybrid scoring
- Automatic handling of document preprocessing
- Fallback mechanisms for edge cases
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import re
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Document:
    """
    Simple document class to standardize document representation.
    
    Attributes:
        page_content (str): The main text content of the document
        metadata (dict): Optional metadata associated with the document
    """
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new document.
        
        Args:
            page_content: The main text content of the document
            metadata: Optional dictionary of metadata (default: empty dict)
        """
        self.page_content = page_content
        self.metadata = metadata or {}

class EnhancedEmbeddingIndex:
    """
    Enhanced embedding index supporting hybrid dense and sparse search.
    
    Combines the strengths of:
    - Dense embeddings (semantic understanding)
    - Sparse TF-IDF (lexical matching)
    
    Attributes:
        dense_model: Sentence Transformer model for dense embeddings
        faiss_index: FAISS index for efficient similarity search
        tfidf_vectorizer: TF-IDF vectorizer for sparse representations
        tfidf_matrix: Sparse matrix of TF-IDF vectors
        documents: List of processed documents
        use_hybrid: Whether to use hybrid search (True) or dense-only (False)
    """
    
    def __init__(self, dense_model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the enhanced embedding index.
        
        Args:
            dense_model_name: Name of the Sentence Transformer model to use
        """
        try:
            self.dense_model = SentenceTransformer(dense_model_name)
            # Test the model
            test_embedding = self.dense_model.encode(["test"], show_progress_bar=False, convert_to_numpy=True)
            print(f"✓ Model loaded successfully, embedding dimension: {test_embedding.shape[1]}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
            
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None
        self.documents: List[Document] = []
        self.use_hybrid: bool = False
        
    def build_index(self, documents: List[Union[Document, str]], use_hybrid: bool = True) -> None:
        """
        Build both dense and sparse indices for hybrid search on document chunks.
        
        Args:
            documents: List of documents (Document objects or strings) to index
            use_hybrid: Whether to enable hybrid search (dense + sparse)
        """
        if not documents:
            print("⚠️ No documents provided for indexing")
            return
        
        # Convert strings to Document objects if needed
        self.documents = []
        for doc in documents:
            if isinstance(doc, str):
                if doc.strip():
                    self.documents.append(Document(doc.strip()))
            elif hasattr(doc, 'page_content'):
                if doc.page_content and doc.page_content.strip():
                    self.documents.append(doc)
        
        if not self.documents:
            print("⚠️ No valid documents found after processing")
            return
        
        print(f"✓ Processing {len(self.documents)} documents")
        self.use_hybrid = use_hybrid
        
        # Extract text content
        texts = [doc.page_content.strip() for doc in self.documents if doc.page_content.strip()]
        
        if not texts:
            print("⚠️ No text content found in documents")
            return
        
        # Build dense index
        try:
            self._build_dense_index(texts)
            print(f"✓ Dense index built with {len(texts)} documents")
        except Exception as e:
            print(f"✗ Error building dense index: {e}")
            return
        
        # Build sparse index if hybrid search is enabled
        if self.use_hybrid and len(texts) >= 2:
            try:
                self._build_sparse_index(texts)
                print(f"✓ Sparse index built")
            except Exception as e:
                print(f"⚠️ Sparse index failed, using dense-only: {e}")
                self.use_hybrid = False
    
    def _build_dense_index(self, texts: List[str]) -> None:
        """Build a dense vector index using FAISS."""
        # Encode documents to dense vectors
        embeddings = self.dense_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        # Ensure 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        dimension = embeddings.shape[1]
        
        # Initialize FAISS index
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.faiss_index.add(embeddings.astype('float32'))
    
    def _build_sparse_index(self, texts: List[str]) -> None:
        """Build a sparse TF-IDF index for hybrid search."""
        if len(texts) < 2:
            return
            
        # Configure TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            sublinear_tf=True
        )
        
        # Fit and transform documents
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def search(self, query: str, k: int = 5, use_hybrid: Optional[bool] = None) -> List[Document]:
        """
        Enhanced search with hybrid approach (dense + sparse).
        
        Args:
            query: The search query string
            k: Number of results to return
            use_hybrid: Whether to use hybrid search (None = auto-detect)
            
        Returns:
            List of documents sorted by relevance (most relevant first)
        """
        print(f"DEBUG - Search called with query: '{query}', k: {k}")
        print(f"DEBUG - Total documents in embedder: {len(self.documents) if hasattr(self, 'documents') else 'No documents'}")
        
        if use_hybrid is None:
            use_hybrid = self.use_hybrid
            
        if not self.faiss_index:
            print("Warning: Index not built or empty")
            # Return empty list instead of failing
            return []
        
        try:
            if use_hybrid and self.tfidf_vectorizer and self.tfidf_matrix is not None:
                results = self._hybrid_search(query, k)
            else:
                results = self._dense_search(query, k)
            
            print(f"DEBUG - Search returned {len(results)} results")
            return results
        except Exception as e:
            print(f"ERROR in search: {e}")
            return []
    
    def _dense_search(self, query: str, k: int) -> List[Document]:
        """
        Perform dense vector search using FAISS.
        
        Args:
            query: The search query string
            k: Number of results to return
            
        Returns:
            List of documents with similarity scores in metadata
        """
        try:
            # Encode query
            query_embedding = self.dense_model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
                
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(k, len(self.documents))
            )
            
            # Return documents with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    doc = self.documents[idx]
                    doc.metadata['similarity_score'] = float(score)
                    results.append(doc)
            
            print(f"DEBUG - Dense search found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error in dense search: {str(e)}")
            # Return first document as fallback if available
            if self.documents:
                fallback_doc = self.documents[0]
                fallback_doc.metadata['similarity_score'] = 0.0
                print(f"DEBUG - Returning fallback document")
                return [fallback_doc]
            return []
    
    def _sparse_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Perform sparse search using TF-IDF.
        
        Args:
            query: The search query string
            k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
            
        try:
            query_vec = self.tfidf_vectorizer.transform([query])
            if query_vec.nnz == 0:  # No matching terms
                return []
                
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Ensure k doesn't exceed the number of documents
            k = min(k, len(similarities))
            if k <= 0:
                return []
            
            # Get top-k non-zero similarity scores with their indices
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[similarities[top_indices] > 0]  # Filter zero scores
            
            # Sort by score in descending order
            if len(top_indices) > 0:
                top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
            
            return list(zip(top_indices, similarities[top_indices]))
            
        except Exception as e:
            print(f"Sparse search failed: {e}")
            return []
    
    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """
        Hybrid search combining dense and sparse retrieval.
        
        Args:
            query: The search query string
            k: Number of results to return
            
        Returns:
            List of documents sorted by hybrid score
        """
        # Ensure k doesn't exceed available documents
        max_available = len(self.documents)
        k = min(k, max_available)
        
        if k <= 0:
            return []
        
        # Get dense and sparse results in parallel
        dense_results = self._dense_search(query, k * 2)
        sparse_results = self._sparse_search(query, k * 2)
        
        # If sparse search failed, fall back to dense only
        if not sparse_results:
            return dense_results[:k]
        
        # Create a mapping of document IDs to their scores and references
        doc_scores: Dict[Any, Dict[str, Any]] = {}
        
        # Add dense scores
        for doc in dense_results:
            doc_id = id(doc)  # Fallback to object ID if no chunk_id
            if hasattr(doc, 'metadata') and 'chunk_id' in doc.metadata:
                doc_id = doc.metadata['chunk_id']
                
            doc_scores[doc_id] = {
                'doc': doc,
                'dense': doc.metadata.get('similarity_score', 0),
                'sparse': 0
            }
        
        # Add sparse scores
        for idx, sparse_score in sparse_results:
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx]
                doc_id = id(doc)
                if hasattr(doc, 'metadata') and 'chunk_id' in doc.metadata:
                    doc_id = doc.metadata['chunk_id']
                
                if doc_id in doc_scores:
                    doc_scores[doc_id]['sparse'] = sparse_score
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'dense': 0,
                        'sparse': sparse_score
                    }
        
        # Calculate hybrid scores (weighted combination)
        dense_weight = 0.7  # Higher weight for dense retrieval
        sparse_weight = 0.3
        
        for doc_info in doc_scores.values():
            hybrid_score = (
                dense_weight * doc_info['dense'] + 
                sparse_weight * doc_info['sparse']
            )
            doc_info['hybrid'] = hybrid_score
            doc_info['doc'].metadata['hybrid_score'] = hybrid_score
        
        # Sort by hybrid score and return top-k
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['hybrid'],
            reverse=True
        )
        
        return [item['doc'] for item in sorted_docs[:k]]
    
    def clear_index(self):
        """Clear all indices and documents from the embedder."""
        self.faiss_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.use_hybrid = False