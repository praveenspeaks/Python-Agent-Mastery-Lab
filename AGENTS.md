# Agent Guide: LangChain Multi-Format Document Parser

This file contains essential information for AI coding agents working on this project.

---

## Project Overview

This is **Chapter 1-2** of the **LangChain & LangGraph Mastery** learning journey. The project implements a robust, class-based document parsing and semantic search engine that:

1. **Parses** documents from multiple formats (PDF, Word, Excel, CSV, JSON, HTML, XML, Markdown, TXT)
2. **Chunks** text intelligently using recursive character splitting
3. **Embeds** text into vectors using local HuggingFace models
4. **Stores** vectors in FAISS for fast similarity search
5. **Answers** questions using OpenRouter LLM integration

The codebase is designed as an **educational platform** with extensive inline documentation, trace logging, and interactive Streamlit interfaces.

---

## Technology Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.x |
| **Core Framework** | LangChain, LangChain-Community |
| **LLM Provider** | OpenRouter (OpenAI-compatible API) |
| **Embeddings** | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Web UI** | Streamlit |
| **Config Management** | python-dotenv |

### Key Dependencies (from `requirements.txt`)
- `langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-openai`, `langchain-huggingface`
- `pypdf`, `pymupdf` - PDF extraction
- `docx2txt`, `python-docx` - Word documents
- `pandas`, `openpyxl` - CSV/Excel processing
- `unstructured` - Universal fallback parser
- `beautifulsoup4`, `lxml` - HTML/XML parsing
- `sqlalchemy` - Database connectivity
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Local embedding models
- `streamlit` - Web interface
- `fpdf2` - Test data generation

---

## Project Structure

```
PythonAgent/
├── src/                          # Core source package
│   ├── __init__.py               # Package init
│   ├── app.py                    # Main orchestrator: LangChainAgent class
│   ├── parser.py                 # DocumentParser: multi-format file loading
│   ├── processor.py              # DocumentProcessor: chunking logic
│   ├── ai_model.py               # AIModelHandler: OpenRouter LLM integration
│   ├── embeddings.py             # EmbeddingProvider: HuggingFace/OpenAI embeddings
│   └── vector_store.py           # VectorStoreManager: FAISS operations
│
├── testdata/                     # Sample test files (auto-generated)
│   ├── sample_pdf.pdf
│   ├── sample_text.txt
│   ├── sample_markdown.md
│   ├── sample_word.docx
│   ├── sample_excel.xlsx
│   ├── sample_csv.csv
│   ├── sample_json.json
│   ├── sample_html.html
│   └── sample_xml.xml
│
├── roadmap/                      # Chapter documentation
│   ├── 01_parsing_documents.md
│   └── 02_embeddings_and_vector_stores.md
│
├── temp_uploads/                 # Runtime file upload directory
├── vector_db/                    # FAISS index persistence directory
├── main.py                       # CLI entry point (batch processing)
├── web_app.py                    # Streamlit Chapter 1 UI
├── learning_lab.py               # Streamlit comprehensive learning UI
├── ai_demo.py                    # AI summarization demo
├── generate_test_data.py         # Test file generator utility
├── test_embeddings.py            # Embedding validation script
├── requirements.txt              # Python dependencies
├── .env                          # API keys and configuration
├── README.md                     # Project overview
└── ROADMAP.md                    # Full learning journey outline
```

---

## Architecture

### Core Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `LangChainAgent` | `app.py` | Main orchestrator. Combines all components and provides high-level API |
| `DocumentParser` | `parser.py` | Factory for LangChain Document Loaders. Handles format detection |
| `DocumentProcessor` | `processor.py` | Text chunking using `RecursiveCharacterTextSplitter` |
| `AIModelHandler` | `ai_model.py` | OpenRouter LLM integration with LCEL chains |
| `EmbeddingProvider` | `embeddings.py` | Manages embedding models (HuggingFace local or OpenAI) |
| `VectorStoreManager` | `vector_store.py` | FAISS index creation, persistence, and search |

### Data Flow

```
Raw File → DocumentParser → Documents → DocumentProcessor → Chunks 
                                                            ↓
User Query ← VectorStoreManager ← FAISS ← EmbeddingProvider
                ↓
         AIModelHandler (optional)
```

### Design Patterns

1. **Strategy Pattern**: Different loaders for different file types in `DocumentParser`
2. **Factory Pattern**: `EmbeddingProvider` creates appropriate embedding model
3. **Pipeline Pattern**: LCEL chains in `AIModelHandler` using pipe operator `|`
4. **Educational Tracing**: Every operation returns `(result, traces)` for UI display

---

## Configuration

### Environment Variables (`.env`)

```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=openai/gpt-4o-mini
```

**Important**: The `.env` file contains sensitive API keys. Never commit it to version control.

### Supported File Formats

| Extension | Loader | Notes |
|-----------|--------|-------|
| `.pdf` | PyMuPDFLoader → UnstructuredPDFLoader (fallback) | Multi-engine with OCR fallback |
| `.txt`, `.md` | TextLoader | Plain text |
| `.docx`, `.doc` | Docx2txtLoader | Word documents |
| `.csv` | CSVLoader | Row-based splitting |
| `.xlsx`, `.xls` | UnstructuredExcelLoader | Excel spreadsheets |
| `.json` | JSONLoader | Uses `.[]` jq schema |
| `.html`, `.htm` | BSHTMLLoader | DOM cleaning |
| `.xml` | UnstructuredFileLoader | Robust XML handling |
| SQL | SQLAlchemyLoader | Via connection string |

---

## Development Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate test data
python generate_test_data.py
```

### Running the Application

```bash
# CLI batch processing
python main.py

# AI demo (requires OPENROUTER_API_KEY)
python ai_demo.py

# Chapter 1 Streamlit UI
streamlit run web_app.py

# Comprehensive Learning Lab (Chapters 1-2)
streamlit run learning_lab.py

# Test embeddings
python test_embeddings.py
```

### Testing

There is no formal test suite (pytest). Testing is done via:
1. `generate_test_data.py` - Creates sample files for manual testing
2. `test_embeddings.py` - Validates embedding functionality
3. Interactive Streamlit UIs for end-to-end testing

---

## Code Style Guidelines

### Python Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Use docstrings for classes and public methods
- Use f-strings for string formatting

### Educational Code Comments
The codebase includes extensive educational comments with a consistent format:
```python
{
    "message": "📦 **Module: JSONLoader** (Structured Data Flattening)",
    "code": """
# 🎯 GOAL: Parse structured JSON data into linear text for the LLM.

# 1. COMMAND:
loader = JSONLoader(...)

# 2. PACKAGE:
# langchain_community.document_loaders

# 3. PARAMETERS:
# file_path: Path to the .json file
# jq_schema: '.[]' -> A filter to flatten a list of objects.

# 4. WHY THIS STRATEGY?
# JSON is nested. LLMs read left-to-right text.
# We flatten the JSON list so each item becomes a separate 'Document'.
"""
}
```

When modifying code with educational traces, maintain this format.

### Naming Conventions
- Classes: `PascalCase` (e.g., `LangChainAgent`, `DocumentParser`)
- Functions/Variables: `snake_case` (e.g., `process_file`, `chunk_size`)
- Constants: `UPPER_CASE` (environment variables)
- Private methods: `_leading_underscore`

---

## Key Implementation Details

### Chunking Strategy
- Uses `RecursiveCharacterTextSplitter` from `langchain_text_splitters`
- Default: `chunk_size=1000`, `chunk_overlap=200`
- Splits hierarchically: paragraphs → sentences → words → characters
- Overlap ensures context isn't lost at chunk boundaries

### PDF Processing
- Primary: `PyMuPDFLoader` (fast, digital PDFs)
- Fallback: `UnstructuredPDFLoader` with `strategy="auto"` (OCR for scanned)
- If PyMuPDF extracts empty text, automatically falls back

### Vector Store
- FAISS index persisted to `vector_db/` directory
- Uses `allow_dangerous_deserialization=True` for local loading
- Embedding dimension: 384 (all-MiniLM-L6-v2)

### LLM Integration
- OpenRouter provides OpenAI-compatible API
- Default model: `openai/gpt-4o-mini`
- Headers required by OpenRouter: `HTTP-Referer`, `X-Title`
- Uses LCEL syntax: `prompt | llm | output_parser`

---

## Security Considerations

1. **API Keys**: Stored in `.env`, never hardcode in source files
2. **File Uploads**: Saved to `temp_uploads/`, validate file types
3. **FAISS Deserialization**: Uses `allow_dangerous_deserialization=True` - only load indexes you created
4. **SQL Injection**: Use parameterized queries with `SQLAlchemyLoader`

---

## Learning Roadmap Context

This project is part of a larger curriculum:

- **Phase 1: Foundations**
  - ✅ 01. Parsing Documents (Current)
  - ✅ 02. Vector Embedding & Stores (Current)
  - ⬜ 03. Advanced Chunking
  - ⬜ 04. Hybrid Search

- **Phase 3: LangGraph & Agentic Workflows**
  - ⬜ 08. LangGraph Basics
  - ⬜ 09. Agentic Architecture
  - ⬜ **10. DeepAgents** (planned) - High-level agents with planning, memory, subagent spawning
  - ⬜ 11. Agentic RAG
  - ⬜ 12. Autonomous RAG

- **Phase 4-5**: Advanced RAG Patterns, Production & Evaluation

When adding features, consider how they fit into this learning progression.

---

## Common Tasks for Agents

### Adding a New Document Format
1. Add loader mapping in `src/parser.py:DocumentParser.__init__`
2. Add format-specific logic in `load_document()` method
3. Add educational trace explaining the loader
4. Update `generate_test_data.py` to create sample files
5. Update supported formats table in this file

### Adding a New Embedding Provider
1. Modify `src/embeddings.py:EmbeddingProvider.__init__`
2. Add provider configuration logic
3. Ensure `get_embeddings()` returns compatible object

### Modifying Chunking Behavior
1. Update `src/processor.py:DocumentProcessor`
2. Consider exposing new parameters in `LangChainAgent.__init__`
3. Update educational traces to explain new strategy

### Adding UI Components
1. For Chapter 1 features: modify `web_app.py`
2. For multi-chapter features: modify `learning_lab.py`
3. Maintain consistent styling with existing CSS
4. Use session state for persistence across reruns

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `OPENROUTER_API_KEY not found` | Check `.env` file exists and is loaded via `load_dotenv()` |
| PDF extraction fails | PyMuPDF may fail on scanned PDFs; Unstructured fallback should activate |
| FAISS index not found | Run `agent.index_file()` first to create the index |
| HuggingFace model download slow | First run downloads ~80MB model; subsequent runs use cache |
| Streamlit port conflict | Use `streamlit run app.py --server.port 8502` |

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [OpenRouter API](https://openrouter.ai/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- Chapter docs: `roadmap/01_parsing_documents.md`, `roadmap/02_embeddings_and_vector_stores.md`
