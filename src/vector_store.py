from typing import List, Optional
import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
try:
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    LLMChainExtractor = None
    print("⚠️ Warning: LLMChainExtractor could not be imported. Contextual Compression will be disabled.")
from src.embeddings import EmbeddingProvider

class VectorStoreManager:
    """
    Manages FAISS vector store operations: creation, persistence, and search.
    """
    def __init__(self, embedding_provider: EmbeddingProvider, persist_directory: str = "vector_db"):
        self.embeddings = embedding_provider.get_embeddings()
        self.persist_directory = persist_directory
        self.vector_store: Optional[FAISS] = None

    def create_index(self, documents: List[Document]):
        """Creates a new FAISS index from a list of documents."""
        print(f"Creating FAISS index with {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save_index()
        return self.vector_store

    def save_index(self):
        """Persists the FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(self.persist_directory)
            print(f"Index saved to {self.persist_directory}")

    def load_index(self):
        """Loads a FAISS index from disk."""
        if os.path.exists(self.persist_directory):
            print(f"Loading index from {self.persist_directory}...")
            self.vector_store = FAISS.load_local(
                self.persist_directory, 
                self.embeddings,
                allow_dangerous_deserialization=True # Required for local loading of pickle files
            )
            return True
        print("No existing index found.")
        return False

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Performs a similarity search."""
        if not self.vector_store:
            if not self.load_index():
                raise ValueError("Vector store not initialized or loaded.")
        
        return self.vector_store.similarity_search(query, k=k)

    def add_documents(self, documents: List[Document]):
        """Adds documents to an existing index."""
        if not self.vector_store:
            if not self.load_index():
                return self.create_index(documents)
        
        self.vector_store.add_documents(documents)
        self.save_index()

    def get_hybrid_retriever(self, documents: List[Document], vector_weight: float = 0.5, b_weight: float = 0.5):
        """
        Creates an EnsembleRetriever combining FAISS and BM25.
        Requires the original documents to initialize BM25.
        """
        if not self.vector_store:
            raise ValueError("Vector store must be initialized before creating hybrid retriever.")
            
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, b_weight]
        )
        return ensemble_retriever

    def get_compressed_retriever(self, llm):
        """
        Creates a ContextualCompressionRetriever that uses an LLM to
        extract only the relevant parts of documents.
        """
        if not self.vector_store:
            raise ValueError("Vector store must be initialized before creating compressed retriever.")
            
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        if LLMChainExtractor is None:
             raise ImportError("LLMChainExtractor is not available in this environment.")
             
        compressor = LLMChainExtractor.from_llm(llm)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        return compression_retriever
