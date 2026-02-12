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
            "message": f"🔎 **Step 1: File Type Detection**",
            "code": f"""
# 🎯 GOAL: Identify the file format to select the correct parsing strategy.

# 1. COMMAND:
ext = os.path.splitext('{os.path.basename(file_path)}')[1].lower()

# 2. PACKAGE:
# 'os' (Python Standard Library) -> Operating System interfaces

# 3. PARAMETERS:
# path: '{os.path.basename(file_path)}' -> The filename to split

# 4. EXPLANATION:
# We need the extension (.csv, .pdf) to decide which LangChain loader to use.
# It splits "data.csv" into ("data", ".csv").
"""
        })
        
        # Special case for JSON
        if ext == ".json":
            code_snippet = (
"""
# 🎯 GOAL: Parse structured JSON data into linear text for the LLM.

# 1. COMMAND:
loader = JSONLoader(
    file_path='{file_path}', 
    jq_schema='.[]', 
    text_content=False
)
docs = loader.load()

# 2. PACKAGE:
# langchain_community.document_loaders

# 3. PARAMETERS:
# file_path: Path to the .json file
# jq_schema: '.[]' -> A filter to flatten a list of objects.
# text_content: False -> Keep the structure (keys/values) instead of just values.

# 4. WHY THIS STRATEGY?
# JSON is nested. LLMs read left-to-right text.
# We flatten the JSON list so each item becomes a separate 'Document'.
"""
            )
            traces.append({
                "message": "📦 **Module: JSONLoader** (Structured Data Flattening)",
                "code": code_snippet
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
        
        # Determine explanation code based on loader
        if ext in [".html", ".htm"]:
            logic_code = (
"""
# 🎯 GOAL: Clean HTML boilerplate to extract only human content.

# 1. COMMAND:
loader = BSHTMLLoader(file_path='{file_path}')
docs = loader.load()

# 2. PACKAGE:
# langchain_community.document_loaders (wraps 'BeautifulSoup4')

# 3. STRATEGY DETAILS:
# - It builds a DOM tree from the HTML.
# - It IDENTIFIES and REMOVES non-content tags: <script>, <style>, <meta>.
# - It EXTRACTS text from <p>, <div>, <span>, <h1>, etc.

# 4. WHY WE USE THIS:
# Raw HTML has too much noise (CSS/JS) that confuses the LLM.
"""
            )
            msg = "🛠️ **Module: BSHTMLLoader** (DOM Parsing & Cleaning)"
        elif ext == ".docx":
            logic_code = (
"""
# 🎯 GOAL: Extract text from MS Word Documents.

# 1. COMMAND:
loader = Docx2txtLoader(file_path='{file_path}')
docs = loader.load()

# 2. PACKAGE:
# langchain_community.document_loaders

# 3. TECHNICAL CONTEXT:
# A .docx file is actually a zipped folder of XML files!
# This loader:
#   a) Unzips the .docx archive.
#   b) Finds 'word/document.xml'.
#   c) Parses the XML tags to find the actual text content.

# 4. WHY WE USE THIS:
# It's lighter and faster than loading the full Word application object model.
"""
            )
            msg = "📎 **Module: Docx2txtLoader** (XML Extraction)"
        elif ext in [".csv", ".xlsx"]:
            logic_code = (
"""
# 🎯 GOAL: Load a Spreadsheet where every row is a distinct data point.

# 1. COMMAND:
loader = CSVLoader(file_path='{file_path}')
docs = loader.load()

# 2. PACKAGE:
# langchain_community.document_loaders

# 3. PARAMETERS:
# file_path: Path to the .csv file

# 4. WHY THIS STRATEGY?
# CSVs are structured. We don't want to lump all rows into one text blob.
# This loader treats EVERY ROW as a separate 'Document' object.
# This helps the AI understand that Row 1 is different from Row 2.
"""
            )
            msg = "📊 **Module: CSV/Excel Loader** (Row-based splitting)"
        else:
            logic_code = f"""
# 🎯 GOAL: Generic loading for {ext} files.

# 1. COMMAND:
loader = {loader_class.__name__}(file_path)
docs = loader.load()

# 2. PACKAGE:
# langchain_community.document_loaders

# 3. STRATEGY:
# Standard text loading strategy.
"""
            msg = f"🛠️ **Module Selected:** `{loader_class.__name__}`"

        traces.append({
            "message": msg,
            "code": logic_code
        })
            
        loader = loader_class(file_path)
        docs = loader.load()
        traces.append({
            "message": "✅ **Normalization Complete: 'Universal Document Format'**",
            "code": (
"""
# 🎯 GOAL: Standardize distinct file types into a common format.

# 1. CLASS:
class Document(BaseModel):
    page_content: str
    metadata: dict

# 2. PACKAGE:
# langchain_core.documents

# 3. ATTRIBUTES:
# page_content: The raw text extracted (e.g. "Meeting Minutes...")
# metadata: Contextual info (e.g. {'source': 'report.pdf', 'page': 1})

# 4. WHY WE USE THIS:
# The LLM doesn't care if the source was a PDF or a website.
# It only processes 'Documents'. This step unifies all inputs.
"""
            )
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
