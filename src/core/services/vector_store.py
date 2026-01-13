"""
FAISS Vector Store implementation for document embeddings and retrieval.

This module provides:
- Document chunking and embedding
- Semantic search functionality
- Document management (add, delete, search)
"""

import os
import pickle
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.core.constant import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    OPENAI_API_KEY,
)
from src.core.logger import app_logger as logger


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    text: str
    metadata: Dict[str, str]
    chunk_id: str


class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(
        self,
        index_name: str = "bot_gpt_documents",
        persist_directory: str = VECTOR_STORE_DIR,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        """
        Initialize the vector store.
        
        Args:
            index_name: Name of the FAISS index
            persist_directory: Directory to persist the index
            embedding_model: Embedding model to use
        """
        self.index_name = index_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding function
        self.embedding_function = self._get_embedding_function()
        
        # Get embedding dimension
        test_embedding = self.embedding_function.embed_query("test")
        self.dimension = len(test_embedding)
        
        # Initialize or load FAISS index
        self.index_path = os.path.join(persist_directory, f"{index_name}.index")
        self.metadata_path = os.path.join(persist_directory, f"{index_name}_metadata.pkl")
        
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self._create_index()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        logger.info(f"FAISS vector store initialized: {index_name}, dimension: {self.dimension}")
    
    def _create_index(self):
        """Create a new FAISS index"""
        # Use IndexFlatL2 for exact search (can be changed to IndexIVFFlat for faster approximate search)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = []  # List to store metadata for each vector
        self.id_to_index = {}  # Map chunk_id to index position
        logger.info("Created new FAISS index")
    
    def _load_index(self):
        """Load existing FAISS index"""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata']
                self.id_to_index = data['id_to_index']
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading index, creating new one: {e}")
            self._create_index()
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata_store,
                    'id_to_index': self.id_to_index
                }, f)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {e}", exc_info=True)
    
    def _get_embedding_function(self):
        """Get the embedding function based on the model name"""
        if "openai" in self.embedding_model.lower() or "text-embedding" in self.embedding_model.lower():
            # Use OpenAI embeddings
            if not OPENAI_API_KEY:
                logger.warning("OpenAI API key not found, falling back to HuggingFace embeddings")
                return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            return OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=OPENAI_API_KEY,
            )
        else:
            # Use HuggingFace embeddings
            return HuggingFaceEmbeddings(model_name=self.embedding_model)
    
    def chunk_text(self, text: str, document_id: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            document_id: ID of the source document
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create DocumentChunk objects with metadata
        document_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "document_id": document_id,
                "chunk_index": str(i),
                "total_chunks": str(len(chunks)),
                **metadata,
            }
            
            chunk_id = f"{document_id}_chunk_{i}"
            document_chunks.append(
                DocumentChunk(
                    text=chunk,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                )
            )
        
        logger.info(f"Created {len(document_chunks)} chunks for document {document_id}")
        return document_chunks
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts for each chunk
            ids: List of unique IDs for each chunk
        """
        try:
            # Generate embeddings
            embeddings = self.embedding_function.embed_documents(texts)
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings_array)
            
            # Store metadata
            for i, (text, metadata, chunk_id) in enumerate(zip(texts, metadatas, ids)):
                idx = start_idx + i
                self.metadata_store.append({
                    'text': text,
                    'metadata': metadata,
                    'chunk_id': chunk_id,
                })
                self.id_to_index[chunk_id] = idx
            
            # Save index
            self._save_index()
            
            logger.info(f"Added {len(texts)} chunks to FAISS index")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
            raise
    
    def add_document_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        self.add_documents(texts, metadatas, ids)
    
    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filter (e.g., {"document_id": "doc123"} or {"document_id": ["doc1", "doc2"]})
            
        Returns:
            List of tuples (text, metadata, distance)
        """
        try:
            if self.index.ntotal == 0:
                logger.warning("Index is empty, no results to return")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_function.embed_query(query)
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS (get more results for filtering)
            search_k = min(top_k * 10, self.index.ntotal) if filter_dict else top_k
            distances, indices = self.index.search(query_array, search_k)
            
            # Format results
            formatted_results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                if idx >= len(self.metadata_store):
                    continue
                
                metadata_entry = self.metadata_store[idx]
                
                # Apply metadata filter if provided
                if filter_dict:
                    match = True
                    for k, v in filter_dict.items():
                        metadata_value = metadata_entry['metadata'].get(k)
                        
                        # Support both single value and list of values
                        if isinstance(v, list):
                            # Check if metadata value is in the list
                            if metadata_value not in v:
                                match = False
                                break
                        else:
                            # Check for exact match
                            if metadata_value != v:
                                match = False
                                break
                    
                    if not match:
                        continue
                
                formatted_results.append((
                    metadata_entry['text'],
                    metadata_entry['metadata'],
                    float(distances[0][i])
                ))
                
                if len(formatted_results) >= top_k:
                    break
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {e}", exc_info=True)
            return []
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: ID of the document to delete
        """
        try:
            # Find all chunk IDs for this document
            chunk_ids_to_delete = [
                entry['chunk_id']
                for entry in self.metadata_store
                if entry['metadata'].get('document_id') == document_id
            ]
            
            if chunk_ids_to_delete:
                self.delete_chunks(chunk_ids_to_delete)
                logger.info(f"Deleted {len(chunk_ids_to_delete)} chunks for document {document_id}")
            else:
                logger.warning(f"No chunks found for document {document_id}")
        
        except Exception as e:
            logger.error(f"Error deleting document from vector store: {e}", exc_info=True)
            raise
    
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete specific chunks by ID.
        Note: FAISS doesn't support direct deletion, so we rebuild the index without deleted items.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        try:
            # Get indices to keep
            indices_to_delete = set(self.id_to_index.get(cid) for cid in chunk_ids if cid in self.id_to_index)
            
            if not indices_to_delete:
                logger.warning("No chunks found to delete")
                return
            
            # Rebuild index without deleted items
            new_metadata_store = []
            new_id_to_index = {}
            vectors_to_keep = []
            
            for idx in range(self.index.ntotal):
                if idx not in indices_to_delete:
                    # Get vector from old index
                    vector = self.index.reconstruct(int(idx))
                    vectors_to_keep.append(vector)
                    
                    # Update metadata
                    new_idx = len(new_metadata_store)
                    metadata_entry = self.metadata_store[idx]
                    new_metadata_store.append(metadata_entry)
                    new_id_to_index[metadata_entry['chunk_id']] = new_idx
            
            # Create new index
            self._create_index()
            
            if vectors_to_keep:
                vectors_array = np.array(vectors_to_keep).astype('float32')
                self.index.add(vectors_array)
            
            self.metadata_store = new_metadata_store
            self.id_to_index = new_id_to_index
            
            # Save updated index
            self._save_index()
            
            logger.info(f"Deleted {len(chunk_ids)} chunks, {self.index.ntotal} remaining")
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}", exc_info=True)
            raise
    
    def get_document_count(self, document_id: Optional[str] = None) -> int:
        """
        Get count of chunks in the store.
        
        Args:
            document_id: Optional document ID to filter by
            
        Returns:
            Number of chunks
        """
        try:
            if document_id:
                count = sum(
                    1 for entry in self.metadata_store
                    if entry['metadata'].get('document_id') == document_id
                )
                return count
            else:
                return self.index.ntotal
        except Exception as e:
            logger.error(f"Error getting document count: {e}", exc_info=True)
            return 0
    
    def reset(self) -> None:
        """Reset the entire index (use with caution!)"""
        try:
            self._create_index()
            self._save_index()
            logger.warning(f"Reset FAISS index")
        except Exception as e:
            logger.error(f"Error resetting index: {e}", exc_info=True)
            raise


# Global vector store instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
