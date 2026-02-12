# Project Architecture & Logic Flow

This document provides a detailed breakdown of the LangChain Document Parser project.

## 1. Project Structure
```text
PythonAgent/
├── src/                        # Core Logic Package
│   ├── __init__.py             # Makes 'src' a Python package
│   ├── parser.py               # Document Loading & Extraction (PDF, Word, Excel, CSV, SQL, etc.)
│   ├── processor.py            # Text Transformation (Chunking)
│   ├── ai_model.py             # OpenRouter AI Integration (LLM Logic)
│   └── app.py                  # Main Orchestrator (LangChainAgent)
├── testdata/                   # Input Folder
│   ├── sample_pdf.pdf          # Binary PDF test file
│   ├── sample_text.txt         # Plain text test file
│   ├── sample_markdown.md      # Markdown test file
│   ├── sample_word.docx        # Word document test file
│   ├── sample_excel.xlsx       # Excel spreadsheet test file
│   ├── sample_csv.csv          # CSV data test file
│   ├── sample_json.json        # JSON data test file
│   ├── sample_html.html        # HTML test file
│   └── sample_xml.xml          # XML test file
├── main.py                     # Project Entry Point (Batch File Processing)
├── ai_demo.py                  # AI Features Demo (Summarization & Q&A)
├── generate_test_data.py       # Utility to create all sample test files
├── requirements.txt            # Project Dependencies
└── .env                        # Configuration (API Keys & Model Settings)
```

---

## 2. File Responsibilities

### `src/parser.py` (The Input Layer)
- **Class**: `DocumentParser`
- **Role**: Factory for LangChain **Document Loaders**.
- **Supported Formats**:
  - **PDF**: `PyMuPDFLoader` (primary) + `UnstructuredPDFLoader` (OCR fallback)
  - **Text/Markdown**: `TextLoader`
  - **Word**: `Docx2txtLoader`
  - **CSV**: `CSVLoader`
  - **Excel**: `UnstructuredExcelLoader`
  - **JSON**: `JSONLoader` (configured with `.[]` jq schema)
  - **SQL**: `SQLAlchemyLoader` (supports any database via connection string)
  - **HTML**: `BSHTMLLoader` (for clean text extraction from tags)
  - **XML**: `UnstructuredFileLoader` (robust XML handling)

### `src/processor.py` (The Logic Layer)
- **Class**: `DocumentProcessor`
- **Role**: Handles **Text Splitting** (Chunking).
- **Logic**: Uses `RecursiveCharacterTextSplitter` to maintain semantic coherence by splitting on hierarchy (paragraphs -> sentences -> words).

### `src/ai_model.py` (The Intelligence Layer)
- **Class**: `AIModelHandler`
- **Role**: Manages communication with **OpenRouter**.
- **Logic**: Connects to any model via OpenRouter's OpenAI-compatible API. Provides methods for summarization and context-aware Q&A.

### `src/app.py` (The Orchestration Layer)
- **Class**: `LangChainAgent`
- **Role**: The "Brain" that connects components.
- **Methods**:
  - `process_file()`: Orchestrates loading a file, chunking it, and optionally analyzing it.
  - `process_sql()`: Similar to file processing but pulls data from a SQL database.

---

## 3. Process Flow (Data Lifecycle)

1.  **Ingestion**: `main.py` (files) or a direct call (SQL) identifies the source.
2.  **Loading**: `DocumentParser` converts raw data into LangChain `Document` objects.
3.  **Refining**: `DocumentProcessor` splits large documents into manageable chunks.
4.  **Intelligence**: If enabled, `AIModelHandler` sends these chunks to the LLM for analysis.
5.  **Output**: Results (chunks or AI responses) are returned to the final application layer.

---

## 4. Dependency Analysis (`requirements.txt`)

| Dependency | Purpose |
| :--- | :--- |
| **`langchain`** | Core orchestrator and interfaces. |
| **`langchain-openai`** | Driver for OpenAI/OpenRouter API. |
| **`pypdf`** | Engine for PDF text extraction. |
| **`docx2txt`** | Engine for Word document extraction. |
| **`pandas` / `openpyxl`** | Engines for CSV and Excel parsing. |
| **`sqlalchemy`** | Engine for connecting to various SQL databases. |
| **`python-dotenv`** | Secure management of API keys via `.env` files. |
| **`fpdf2`** | Utility to generate test PDF files. |
| **`unstructured`** | Advanced parser used for XML and fallback for complex layouts. |
| **`beautifulsoup4`** | Used by `BSHTMLLoader` for high-quality HTML parsing. |
| **`lxml`** | High-performance XML/HTML processing engine. |

---

## 5. Architectural Benefits
- **Modularity**: Every format has its own loader logic; every transformation has its own class.
- **Scalability**: New file types can be added to `DocumentParser` in seconds.
- **AI-Agnostic**: By using OpenRouter, you can swap models (GPT-4o mini, Claude 3.5 Sonnet, Gemini 2.0, etc.) just by changing a string in `.env`.
