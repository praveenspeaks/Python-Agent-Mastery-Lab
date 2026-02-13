import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    TextLoader, 
    UnstructuredFileLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    JSONLoader,
    BSHTMLLoader
)

class DocumentParser:
    """
    A class to handle parsing of various document formats using LangChain loaders.
    """
    
    def __init__(self):
        # Map extensions to specific LangChain loaders
        self.loaders = {
            ".pdf": PyMuPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
            ".csv": CSVLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
            ".html": BSHTMLLoader,
            ".htm": BSHTMLLoader,
            ".xml": UnstructuredFileLoader, # Unstructured handles XML well
        }

    def load_document(self, file_path: str):
        """
        Loads a document based on its file extension.
        Returns a tuple: (List of Documents, List of educational traces)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        traces = []
        traces.append({
            "step": "File Type Detection",
            "module": "src.parser.DocumentParser",
            "command": "ext = os.path.splitext(file_path)[1].lower()",
            "variables": {
                "file_path": file_path,
                "extension": ext,
                "basename": os.path.basename(file_path)
            },
            "input": file_path,
            "output": ext,
            "explanation": "Identify the file format to select the correct LangChain loader strategy."
        })
        
        # Special case for JSON
        if ext == ".json":
            traces.append({
                "step": "JSON Loading Strategy",
                "module": "langchain_community.document_loaders.JSONLoader",
                "command": "loader = JSONLoader(file_path=file_path, jq_schema='.[]', text_content=False)",
                "variables": {"jq_schema": ".[]", "text_content": False},
                "input": "Raw JSON File Context",
                "output": "List[Document]",
                "explanation": "Flattening nested JSON structures so each entry becomes a distinct Document object."
            })
            loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
            docs = loader.load()
            return docs, traces

        if ext == ".pdf":
            traces.append({
                "message": "📑 **PDF Strategy: Multi-Engine Orchestration**",
                "code": (
"""
# 🎯 GOAL: Extract text from PDF (which is a binary format, not text).

# STRATEGY OVERVIEW:
# PDFs are complex. They can be:
# 1. Digital (Text Layer exists) -> Fast extraction.
# 2. Scanned (Images only) -> Needs OCR (Optical Character Recognition).

# SYSTEM LOGIC:
try:
    # Attempt 1: Fast & Clean (PyMuPDF)
    loader = PyMuPDFLoader(file_path)
except:
    # Attempt 2: Slow & Heavy (Unstructured OCR)
    loader = UnstructuredPDFLoader(file_path)
"""
                )
            })
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                text_content = "".join(doc.page_content for doc in docs).strip()
                if text_content:
                    traces.append({
                        "message": "🚀 **PyMuPDF Engine Success** (Text Layer Found)", 
                        "code": (
"""
# 🎯 GOAL: Fast text extraction from a Digital PDF.

# 1. COMMAND:
loader = PyMuPDFLoader(file_path)
docs = loader.load()

# 2. PACKAGE:
# langchain_community.document_loaders (wraps 'fitz' / PyMuPDF)

# 3. PARAMETERS:
# file_path: Path to the .pdf file

# 4. WHY WE USE THIS:
# It directly accesses the internal text stream of the PDF.
# Benefit: Extremely fast (milliseconds) and 100% accurate for digital PDFs.
"""
                        )
                    })
                    return docs, traces
            except:
                pass

            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(file_path, strategy="auto")
                docs = loader.load()
                traces.append({
                    "message": "🛡️ **Unstructured Engine Success** (OCR/Partitioning)", 
                    "code": (
"""
# 🎯 GOAL: OCR extraction for Scanned PDFs.

# 1. COMMAND:
loader = UnstructuredPDFLoader(file_path, strategy='auto')
docs = loader.load()

# 2. PACKAGE:
# langchain_community.document_loaders (wraps 'unstructured')

# 3. PARAMETERS:
# strategy='auto': Automatically determines if OCR is needed.
# mode='elements': (Optional) Can keep tables/titles separate.

# 4. WHY WE USE THIS:
# PyMuPDF failed (likely a scanned image). 
# Unstructured uses computer vision (Tesseract) to 'read' the pixels.
"""
                    )
                })
                return docs, traces
            except Exception as e:
                raise ValueError(f"Full PDF extraction failed: {e}")

        loader_class = self.loaders.get(ext, UnstructuredFileLoader)
        
        traces.append({
            "step": f"Loader Selection: {loader_class.__name__}",
            "module": f"langchain_community.document_loaders.{loader_class.__name__}",
            "command": f"loader = {loader_class.__name__}(file_path); docs = loader.load()",
            "variables": {"loader_type": loader_class.__name__, "extension": ext},
            "input": file_path,
            "output": "List[Document]",
            "explanation": f"Using the specialized {loader_class.__name__} for optimized text extraction from {ext} files."
        })
            
        loader = loader_class(file_path)
        docs = loader.load()
        traces.append({
            "step": "Document Normalization",
            "module": "langchain_core.documents.Document",
            "command": "[Document(page_content=text, metadata={...}) for text in raw_splits]",
            "variables": {
                "total_documents": len(docs),
                "metadata_keys": list(docs[0].metadata.keys()) if docs else []
            },
            "input": "File-specific raw data",
            "output": f"List with {len(docs)} standardized Document objects",
            "explanation": "Standardizing all inputs into a common 'Document' class so the rest of the pipeline works regardless of source format."
        })
        return docs, traces

    def load_sql(self, connection_string: str, query: str) -> List[Document]:
        """
        Loads data from a SQL database.
        """
        from langchain_community.document_loaders import SQLAlchemyLoader
        print(f"Loading data from SQL using query: {query}")
        loader = SQLAlchemyLoader(query=query, url=connection_string)
        return loader.load()

    def load_multiple(self, file_paths: List[str]) -> List[Document]:
        """
        Loads multiple documents and returns a combined list.
        """
        all_docs = []
        for path in file_paths:
            all_docs.extend(self.load_document(path))
        return all_docs
