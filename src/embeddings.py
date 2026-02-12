from typing import List, Optional
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class EmbeddingProvider:
    """
    Manages embedding models, supporting both local HuggingFace and OpenAI models.
    """
    def __init__(self, provider: str = "huggingface", model_name: Optional[str] = None):
        self.provider = provider.lower()
        
        if self.provider == "huggingface":
            # Default to a lightweight, high-performance model
            self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Initializing HuggingFace Embeddings with model: {self.model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        elif self.provider == "openai":
            print("Initializing OpenAI Embeddings")
            self.embeddings = OpenAIEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def get_embeddings(self):
        """Returns the LangChain embeddings object."""
        return self.embeddings

    def embed_query(self, text: str) -> List[float]:
        """Converts a single query string into a vector."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Converts a list of documents into vectors."""
        return self.embeddings.embed_documents(texts)
