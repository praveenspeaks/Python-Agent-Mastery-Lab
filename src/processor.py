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
                return [], [{"message": "❌ Semantic splitter requires embeddings provider.", "code": ""}]
            
            traces.append({
                "message": "🧠 **Advanced Step: Semantic Chunking**",
                "code": """
# 🎯 GOAL: Group sentences by meaning, not length.

# 1. COMMAND:
# Uses embeddings to find 'breakpoints' where the topic changes.
splitter = SemanticChunker(embeddings_provider)
chunks = splitter.split_documents(documents)

# 2. LOGIC:
# It calculates the cosine similarity between every sentence.
# If the similarity drops below a threshold, it creates a new chunk.
"""
            })
            chunks = self.semantic_splitter.split_documents(documents)
        else:
            traces.append({
                "message": "✂️ **Step 2: Intelligent Chunking Strategy**",
                "code": (
f"""
# 🎯 GOAL: Break large text into manageable pieces for the AI.

# 1. COMMAND:
splitter = RecursiveCharacterTextSplitter(
    chunk_size={self.chunk_size},
    chunk_overlap={self.chunk_overlap}
)
chunks = splitter.split_documents(documents)

# 2. PACKAGE:
# langchain_text_splitters

# 3. PARAMETERS:
# chunk_size={self.chunk_size}: Max characters per text block. 
# chunk_overlap={self.chunk_overlap}: Characters repeated between chunks. 

# 4. STRATEGY EXPLAINED:
# Why 'Recursive'? It respects language structure.
# Priorities for splitting:
#   1. Paragraphs (\\n\\n) -> Best (Keeps independent thoughts whole)
#   2. Sentences (\\n)     -> Good (Keeps logic whole)
#   3. Words (space)       -> Okay
#   4. Characters          -> Last resort
#
# Why 'Overlap'? 
#   If a sentence is cut in half at the end of Chunk 1, 
#   the overlap repeats it at the start of Chunk 2.
#   This ensures the AI doesn't lose context.
"""
                )
            })
            chunks = self.recursive_splitter.split_documents(documents)
        
        traces.append({
            "message": f"📈 **Output Analysis:** Generated {len(chunks)} Chunks",
            "code": (
f"""
# 📊 RESULTS:

# Input Documents: {len(documents)}
# Output Chunks:   {len(chunks)}

# VISUALIZING THE TRANSFORMATION:
# [Document 1 (3000 chars)] 
#       ⬇️ Splitter (Size=1000, Overlap=100)
# [Chunk 1 (0-1000)]
# [Chunk 2 (900-1900)]  <-- Note the overlap starting at 900
# [Chunk 3 (1800-2800)]
# [Chunk 4 (2700-3000)]
"""
            )
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
