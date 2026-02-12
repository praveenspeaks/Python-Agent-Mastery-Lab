# 🎓 AI Mastery Learning Lab

Welcome to the **LangChain, LangGraph, and DeepAgents Mastery Lab**. 
This project is an interactive educational platform designed to take you from a Python beginner to an AI Engineer capable of building autonomous agents.

## 🚀 What is this?
This is not just a collection of scripts. It is a **Learning Lab application** (`learning_lab.py`) that provides a visual dashboard to interact with every concept we cover:
1.  **Foundations**: Parsing PDFs, standard RAG.
2.  **Advanced RAG**: HyDE, Contextual Compression, Semantic Routing.
3.  **Agentic Workflows**: Multi-Agent teams, Reflection, Persistence.
4.  **DeepAgents**: Autonomous Planning and Tool Synthesis.

## 🛠️ Prerequisites
- **Python 3.10+** installed on your system.
- Basic knowledge of Python (functions, classes).
- An API Key for an LLM (OpenAI or OpenRouter).

## � Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Python-Agent-Mastery-Lab
```

### 2. Create a Virtual Environment (Recommended)
Calculated isolation keeps your dependencies clean.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: If you encounter errors with `certifi` or `langchain` imports on Windows, try upgrading pip first: `python -m pip install --upgrade pip`.*

## 🔑 Configuration

1.  **Create env file**: Copy the example (or create new) `.env` file in the root directory.
    ```bash
    # Windows (PowerShell)
    cp .env.example .env
    ```
2.  **Add API Keys**: Open `.env` and add your keys.
    ```ini
    # RECOMMENDED: Use OpenRouter for access to all models (Claude, GPT-4, Llama)
    OPENROUTER_API_KEY=sk-or-v1-...
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
    DEFAULT_MODEL=openai/gpt-4o-mini
    
    # ADVANCED: Use OpenAI directly
    # OPENAI_API_KEY=sk-...
    ```
    *Note: You do NOT need a HuggingFace API key. The embedding models run locally for free.*

## ▶️ Running the Application

To start the interactive Learning Lab:
```bash
streamlit run learning_lab.py
```
This will open a tab in your browser (usually `http://localhost:8501`).

### How to Use the Lab
The sidebar allows you to navigate through the **3 Pillars** of the course:
1.  **LangChain (Phases 1-2)**: Select chapters like "Parsing" or "Vector Stores" to test data ingestion.
2.  **LangGraph (Phases 3-4)**: Select "Multi-Agent" or "DeepAgents" to see agents talking to each other.
3.  **DeepAgents (Phases 5-6)**: The "Graduation" content where agents write their own code.

## 📂 Project Structure

- **`learning_lab.py`**: The main frontend application. Start here.
- **`src/`**: The production-grade backend code.
    - `app.py`: The `LangChainAgent` class (Orchestrator).
    - `graph_agent.py`: The `LangGraph` workflows (Planner, Tool Synthesizer).
    - `ai_model.py`: The wrapper for LLM calls.
    - `vector_store.py`: Manages your data storage.
- **`roadmap/`**: Detailed markdown documentation for every single chapter.
    - Example: `roadmap/19_autonomous_planning.md` explains exactly how the Planner agent works.

## 🐛 Troubleshooting

**Q: I get an "ImportError: cannot import name 'BaseMemory'"?**
A: This usually means a version mismatch with `langchain`. Run:
```bash
pip install --upgrade langchain langchain-community langchain-core
```

**Q: The app says "OPENROUTER_API_KEY is missing"?**
A: Make sure you created the `.env` file and saved it in the MAIN folder (not in `src` or `roadmap`).

**Q: How do I test without the UI?**
A: We have included verification scripts:
```bash
python tests/verify_retrieval.py   # Tests RAG
python tests/verify_agents.py      # Tests Agents
```

---
*Built with ❤️ for the AI Engineering Community.*
