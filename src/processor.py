from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from typing import List, Optional
from langchain_core.documents import Document

class DocumentProcessor:
    """
    A class to handle document processing tasks like chunking.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, embeddings=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = embeddings
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.semantic_splitter = None
        if self.embeddings:
            self.semantic_splitter = SemanticChunker(self.embeddings)

    def split_documents(self, documents: List[Document], mode: str = "recursive"):
        """
        Splits a list of documents into smaller chunks.
        mode: "recursive" or "semantic"
        Returns a tuple: (List of Chunks, List of educational traces)
        """
        if not documents:
            return [], [{"message": "⚠️ No documents found to process.", "code": ""}]
            
        traces = []
        
        if mode == "semantic":
            if not self.semantic_splitter:
                return [], []
            
            traces.append({
                "step": "Semantic Topic Analysis",
                "module": "langchain_experimental.text_splitter.SemanticChunker",
                "command": "splitter.split_documents(documents)",
                "variables": {
                    "embeddings": self.embeddings.__class__.__name__ if self.embeddings else "None",
                    "breakpoint_threshold_type": "percentile"
                },
                "input": f"{len(documents)} Document(s)",
                "output": "Variable Chunk List",
                "explanation": "Calculates cosine similarity between sentence embeddings to find optimal break points where themes change."
            })
            chunks = self.semantic_splitter.split_documents(documents)
        else:
            traces.append({
                "step": "Recursive Character Splitting",
                "module": "langchain_text_splitters.RecursiveCharacterTextSplitter",
                "command": f"splitter = RecursiveCharacterTextSplitter(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})",
                "variables": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "separators": ["\\n\\n", "\\n", " ", ""]
                },
                "input": f"{len(documents)} Document(s)",
                "output": "List[Document] (Chunks)",
                "explanation": "Recursively splits text by paragraphs, then sentences, then words until blocks fit within the chunk_size limit."
            })
            chunks = self.recursive_splitter.split_documents(documents)
        
        traces.append({
            "step": "Chunk Analysis & Statistics",
            "module": "src.processor.DocumentProcessor",
            "command": "N/A (Summarization)",
            "variables": {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c.page_content) for c in chunks)//len(chunks) if chunks else 0
            },
            "input": f"{len(documents)} Docs",
            "output": f"{len(chunks)} Chunks",
            "explanation": "Summarizing the results of the splitting process for pipeline monitoring."
        })
            
        return chunks, traces

    def update_settings(self, chunk_size: int, chunk_overlap: int):
        """
        Updates the splitter settings.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
