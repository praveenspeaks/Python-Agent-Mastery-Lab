"""
LangChain Curriculum - Step-by-step learning chapters

8 Chapters covering:
1. Document Loading
2. Text Splitting
3. Embeddings
4. Vector Stores
5. Chains (LCEL)
6. Tools
7. Agents
8. Memory
"""

from typing import List, Dict, Any
from src.curriculum.code_explorer import (
    get_document_loader_example,
    get_text_splitter_example,
    get_embedding_example,
    get_vector_store_example,
    get_chain_example,
    get_retrieval_chain_example
)


# ============================================================================
# CHAPTER 1: DOCUMENT LOADING
# ============================================================================

def demo_document_loading():
    """Interactive demo for document loading."""
    import streamlit as st
    import os
    
    st.markdown("### 📂 Try Document Loading")
    
    test_data_dir = "testdata"
    if os.path.exists(test_data_dir):
        files = [f for f in os.listdir(test_data_dir) if os.path.isfile(os.path.join(test_data_dir, f))]
        
        if files:
            selected_file = st.selectbox("Choose a file to load:", files)
            
            if st.button("Load Document"):
                from src.app import LangChainAgent
                agent = LangChainAgent()
                
                with st.spinner("Loading..."):
                    docs, logs = agent.parser.load_document(os.path.join(test_data_dir, selected_file))
                
                st.success(f"✅ Loaded {len(docs)} document(s)")
                
                # Show logs
                with st.expander("🔍 See what happened behind the scenes"):
                    for log in logs:
                        st.info(log["message"])
                
                # Show content preview
                with st.expander("📄 Document Preview"):
                    for i, doc in enumerate(docs[:3]):  # Show first 3
                        st.markdown(f"**Document {i+1}**")
                        st.text_area(f"Content {i+1}", doc.page_content[:500] + "...", height=100)
                        st.json(doc.metadata)
        else:
            st.warning("No test files found. Run `python generate_test_data.py` first.")
    else:
        st.error("testdata directory not found.")


LANGCHAIN_CURRICULUM: List[Dict[str, Any]] = [
    {
        "title": "Document Loading",
        "subtitle": "Reading files into AI-compatible formats",
        "emoji": "📄",
        "objectives": [
            "Understand what Document Loaders are and why they matter",
            "Load text files, PDFs, and other formats",
            "Access document content and metadata",
            "Choose the right loader for each file type"
        ],
        "content": """
        ## What is Document Loading?
        
        Before an AI can process your files, you need to **load** them into a format the AI understands.
        LangChain provides **Document Loaders** for this purpose.
        
        ### The Document Object
        Every loader returns `Document` objects with two key properties:
        - **page_content**: The actual text content (string)
        - **metadata**: Information about the source (dict)
        
        ### Common Loaders
        
        | Loader | File Type | Use Case |
        |--------|-----------|----------|
        | `TextLoader` | .txt, .md | Simple text files |
        | `PyPDFLoader` | .pdf | PDF documents |
        | `CSVLoader` | .csv | Spreadsheet data |
        | `JSONLoader` | .json | Structured JSON |
        | `Docx2txtLoader` | .docx | Word documents |
        | `BSHTMLLoader` | .html | Web pages |
        
        ### Why Different Loaders?
        Each file format stores text differently:
        - **PDFs**: Can have text layers, images, or be scanned
        - **Word docs**: Have formatting, headers, footers
        - **HTML**: Has tags, scripts, styles to filter out
        - **CSV**: Has rows and columns to parse
        
        Loaders handle these complexities so you don't have to!
        """,
        "code_examples": [get_document_loader_example()],
        "demo": demo_document_loading,
        "quiz": [
            {
                "question": "What does a Document Loader return?",
                "options": [
                    "Just the text content as a string",
                    "A list of Document objects",
                    "A JSON object",
                    "A pandas DataFrame"
                ],
                "correct": "A list of Document objects",
                "explanation": "All LangChain loaders return List[Document], where each Document has page_content and metadata."
            },
            {
                "question": "Which loader would you use for a PDF file?",
                "options": ["TextLoader", "PyPDFLoader", "CSVLoader", "JSONLoader"],
                "correct": "PyPDFLoader",
                "explanation": "PyPDFLoader is specifically designed for PDF files. It extracts text from each page."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 2: TEXT SPLITTING
    # ============================================================================
    {
        "title": "Text Splitting (Chunking)",
        "subtitle": "Breaking documents into AI-sized pieces",
        "emoji": "✂️",
        "objectives": [
            "Understand why chunking is necessary",
            "Use RecursiveCharacterTextSplitter effectively",
            "Configure chunk_size and chunk_overlap",
            "Preserve context across chunk boundaries"
        ],
        "content": """
        ## Why Split Text?
        
        LLMs have a **context window limit** (e.g., 4k, 8k, 128k tokens). You can't feed an entire book 
        to most models. Even if you could, smaller chunks often give better results for retrieval.
        
        ### The Chunking Problem
        
        Imagine splitting a recipe:
        ```
        Bad split:  "1. Preheat oven to 350. 2. Mix flour" | "and sugar in a bowl. 3. Add eggs..."
        Good split: "1. Preheat oven to 350." | "2. Mix flour and sugar in a bowl." | "3. Add eggs..."
        ```
        
        **The goal**: Split at natural boundaries (paragraphs, sentences) while keeping semantic meaning.
        
        ### RecursiveCharacterTextSplitter
        
        This is LangChain's most intelligent splitter:
        
        1. **First**, tries to split on paragraph breaks (`\\n\\n`)
        2. **If still too big**, splits on line breaks (`\\n`)
        3. **If still too big**, splits on spaces
        4. **Last resort**, splits on characters
        
        This preserves sentence structure as much as possible!
        
        ### Key Parameters
        
        - **chunk_size**: Maximum characters per chunk (e.g., 1000)
        - **chunk_overlap**: Characters shared between chunks (e.g., 200)
        
        **Why overlap?** So context isn't lost at chunk boundaries.
        
        ### Example
        If chunk 1 ends with "The quick brown", chunk 2 starts with "brown fox jumped."
        Both chunks have "brown" so the meaning carries over!
        """,
        "code_examples": [get_text_splitter_example()],
        "demo": None,  # Could add interactive chunking demo
        "quiz": [
            {
                "question": "Why do we need chunk_overlap?",
                "options": [
                    "To make chunks smaller",
                    "To preserve context at chunk boundaries",
                    "To increase processing speed",
                    "To reduce memory usage"
                ],
                "correct": "To preserve context at chunk boundaries",
                "explanation": "Overlap ensures that if a sentence is split across chunks, both chunks have enough context to understand it."
            },
            {
                "question": "What does RecursiveCharacterTextSplitter try first?",
                "options": [
                    "Splitting on individual characters",
                    "Splitting on paragraph breaks",
                    "Splitting on words",
                    "Splitting randomly"
                ],
                "correct": "Splitting on paragraph breaks",
                "explanation": "It tries the largest separator first (paragraphs), then falls back to smaller ones if needed."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 3: EMBEDDINGS
    # ============================================================================
    {
        "title": "Embeddings",
        "subtitle": "Converting text to vectors (numbers)",
        "emoji": "🧠",
        "objectives": [
            "Understand what embeddings are conceptually",
            "Convert text to vectors using embedding models",
            "Choose between local and cloud embedding providers",
            "Understand vector dimensions"
        ],
        "content": """
        ## What Are Embeddings?
        
        **Embeddings** are a way to represent text as numbers (vectors) that capture meaning.
        
        ### The Analogy: GPS Coordinates
        
        Just like every city has GPS coordinates (latitude, longitude):
        - New York: (40.7, -74.0)
        - London: (51.5, -0.1)
        
        Every piece of text has embedding coordinates in "meaning space":
        - "King": [0.1, -0.5, 0.3, ...] (384 numbers)
        - "Queen": [0.2, -0.4, 0.4, ...]
        
        ### The Magic
        
        Similar texts have similar vectors! This means:
        - "Cat" and "Kitten" are close together
        - "Python" (programming) and "JavaScript" are close
        - "Python" (snake) and "Cobra" are close
        - But "Python" (programming) and "Python" (snake) might be far apart based on context!
        
        ### Vector Math
        
        You can do arithmetic with embeddings:
        ```
        King - Man + Woman ≈ Queen
        Paris - France + Italy ≈ Rome
        ```
        
        ### Embedding Models
        
        | Model | Provider | Dimensions | Cost | Best For |
        |-------|----------|------------|------|----------|
        | all-MiniLM-L6-v2 | HuggingFace | 384 | Free | Quick start, privacy |
        | text-embedding-3-small | OpenAI | 1536 | $ | Higher quality |
        | text-embedding-3-large | OpenAI | 3072 | $$ | Maximum quality |
        
        ### HuggingFace vs OpenAI
        
        **HuggingFace (Local):**
        - ✅ Free forever
        - ✅ Runs on your computer (privacy)
        - ✅ Works offline
        - ❌ Lower quality than OpenAI
        - ❌ Uses your CPU/GPU
        
        **OpenAI (Cloud):**
        - ✅ Best quality embeddings
        - ✅ Fast (runs on OpenAI's servers)
        - ❌ Costs money
        - ❌ Sends data to cloud
        """,
        "code_examples": [get_embedding_example()],
        "demo": None,
        "quiz": [
            {
                "question": "What is an embedding vector?",
                "options": [
                    "A 3D model of text",
                    "A list of numbers representing text meaning",
                    "A compressed image of text",
                    "A type of database"
                ],
                "correct": "A list of numbers representing text meaning",
                "explanation": "Embeddings convert text into numerical vectors that capture semantic meaning."
            },
            {
                "question": "How many dimensions does all-MiniLM-L6-v2 output?",
                "options": ["100", "384", "768", "1536"],
                "correct": "384",
                "explanation": "all-MiniLM-L6-v2 is a lightweight model that outputs 384-dimensional vectors."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 4: VECTOR STORES
    # ============================================================================
    {
        "title": "Vector Stores",
        "subtitle": "Storing and searching vectors efficiently",
        "emoji": "💾",
        "objectives": [
            "Understand what vector stores do",
            "Store documents in FAISS",
            "Perform semantic similarity search",
            "Save and load indexes"
        ],
        "content": """
        ## What Are Vector Stores?
        
        **Vector stores** are specialized databases for finding similar vectors.
        
        ### The Problem
        
        You have 10,000 document chunks, each with a 384-dimensional vector.
        User asks: "What is machine learning?"
        
        **Naive approach:** Compare query vector to all 10,000 vectors (slow!)
        **Vector store approach:** Use special data structures for O(log n) search (fast!)
        
        ### How Similarity Search Works
        
        1. Convert user query to vector using same embedding model
        2. Vector store finds the k closest vectors using:
           - **Cosine similarity**: Angle between vectors
           - **Euclidean distance**: Straight-line distance
        3. Return the documents associated with those vectors
        
        ### FAISS (Facebook AI Similarity Search)
        
        **FAISS** is an open-source library optimized for:
        - High-dimensional vectors
        - Billions of vectors
        - GPU acceleration
        
        **Why FAISS?**
        - ✅ Completely free and open source
        - ✅ Runs locally (no cloud dependency)
        - ✅ Very fast (C++ backend)
        - ✅ Works great with LangChain
        
        ### The Workflow
        
        ```
        Documents → Chunks → Embeddings → FAISS Index
                                             ↓
        Query → Embedding → Search Index → Similar Chunks
        ```
        
        ### Index Persistence
        
        Creating embeddings takes time. You can save your FAISS index:
        ```python
        vectorstore.save_local("vector_db/my_index")
        
        # Later, load it instantly:
        vectorstore = FAISS.load_local("vector_db/my_index", embeddings)
        ```
        
        **⚠️ Security Note:** Only load indexes you created! 
        `allow_dangerous_deserialization=True` is required but potentially risky with untrusted files.
        """,
        "code_examples": [get_vector_store_example()],
        "demo": None,
        "quiz": [
            {
                "question": "What does a vector store do?",
                "options": [
                    "Stores text files",
                    "Finds similar vectors efficiently",
                    "Trains ML models",
                    "Renders web pages"
                ],
                "correct": "Finds similar vectors efficiently",
                "explanation": "Vector stores use specialized data structures to find similar vectors much faster than brute force."
            },
            {
                "question": "Why save a FAISS index to disk?",
                "options": [
                    "To save memory",
                    "To avoid re-embedding documents every time",
                    "To share with other users",
                    "To encrypt the data"
                ],
                "correct": "To avoid re-embedding documents every time",
                "explanation": "Creating embeddings is slow. Saving the index lets you reload instantly without re-processing."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 5: CHAINS (LCEL)
    # ============================================================================
    {
        "title": "Chains with LCEL",
        "subtitle": "Composing operations with the pipe operator",
        "emoji": "⛓️",
        "objectives": [
            "Understand what chains are",
            "Use the LCEL pipe operator",
            "Build multi-step pipelines",
            "Use prompt templates"
        ],
        "content": """
        ## What Are Chains?
        
        **Chains** are sequences of operations where the output of one step becomes the input of the next.
        
        ### Simple Chain Example
        
        ```
        User Input → Prompt Template → LLM → Output Parser → Final Output
        ```
        
        ### LCEL: LangChain Expression Language
        
        LCEL uses the **pipe operator (`|`)** to compose components, similar to Unix pipes:
        
        ```bash
        # Unix: cat file.txt | grep "hello" | wc -l
        # LCEL: prompt | llm | output_parser
        ```
        
        ### Why LCEL?
        
        1. **Readable**: Clear flow of data
        2. **Composable**: Easy to add/remove steps
        3. **Streaming**: Automatic support for streaming responses
        4. **Async**: Built-in async support
        5. **Parallel**: Automatically parallelizes where possible
        
        ### Prompt Templates
        
        Instead of hardcoding prompts, use templates with placeholders:
        
        ```python
        template = "Translate the following to French: {text}"
        prompt = ChatPromptTemplate.from_template(template)
        
        # Later:
        prompt.format(text="Hello world")  # "Translate the following to French: Hello world"
        ```
        
        ### Common Chain Components
        
        | Component | Purpose | Example |
        |-----------|---------|---------|
        | `ChatPromptTemplate` | Format prompts | `ChatPromptTemplate.from_template("Hello {name}")` |
        | `ChatOpenAI` | Call LLM | `ChatOpenAI(model="gpt-4")` |
        | `StrOutputParser` | Extract text | `StrOutputParser()` |
        | `RunnablePassthrough` | Pass through | `RunnablePassthrough()` |
        
        ### Building a Simple Chain
        
        ```python
        chain = prompt | llm | output_parser
        result = chain.invoke({"text": "Hello"})
        ```
        
        **What happens:**
        1. `prompt` receives `{"text": "Hello"}`, outputs formatted prompt
        2. `llm` receives prompt, outputs AI response object
        3. `output_parser` receives response, extracts text string
        """,
        "code_examples": [get_chain_example()],
        "demo": None,
        "quiz": [
            {
                "question": "What does the `|` operator do in LCEL?",
                "options": [
                    "Logical OR",
                    "Pipes output of left component to input of right",
                    "Bitwise OR",
                    "Creates a list"
                ],
                "correct": "Pipes output of left component to input of right",
                "explanation": "The pipe operator chains components together, passing output from one to the next."
            },
            {
                "question": "What is RunnablePassthrough used for?",
                "options": [
                    "To skip a component",
                    "To pass input through unchanged",
                    "To run async code",
                    "To parse output"
                ],
                "correct": "To pass input through unchanged",
                "explanation": "RunnablePassthrough passes data through without modification, useful for parallel branches."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 6: RAG CHAINS
    # ============================================================================
    {
        "title": "Building RAG Systems",
        "subtitle": "Retrieval-Augmented Generation",
        "emoji": "🔍",
        "objectives": [
            "Understand the RAG pattern",
            "Connect retrievers to LLMs",
            "Format retrieved context",
            "Build end-to-end RAG chains"
        ],
        "content": """
        ## What is RAG?
        
        **RAG (Retrieval-Augmented Generation)** combines:
        1. **Retrieval**: Finding relevant documents from your knowledge base
        2. **Augmentation**: Adding those documents to the LLM's context
        3. **Generation**: Having the LLM answer based on the retrieved context
        
        ### Why RAG?
        
        LLMs only know what they were trained on. RAG gives them access to:
        - Your private documents
        - Recent information (post-training)
        - Specific domain knowledge
        
        ### The RAG Pipeline
        
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │  1. User asks: "What is our refund policy?"                  │
        │                      ↓                                       │
        │  2. Query → Embedding → Search Vector Store                  │
        │                      ↓                                       │
        │  3. Retrieve top-k relevant chunks                           │
        │                      ↓                                       │
        │  4. Format: Context + Question → Prompt                      │
        │                      ↓                                       │
        │  5. LLM generates answer based on context                    │
        └─────────────────────────────────────────────────────────────┘
        ```
        
        ### The Prompt Template
        
        A good RAG prompt instructs the LLM to use the context:
        
        ```
        Answer the question based on the context below.
        If you can't answer, say "I don't know".
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        ```
        
        **Why this works:**
        - Gives the LLM specific information to use
        - Reduces hallucinations (grounding in facts)
        - Allows "I don't know" for irrelevant queries
        
        ### The Retriever
        
        ```python
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        ```
        
        This turns your vector store into a component that:
        1. Accepts a query string
        2. Embeds it
        3. Searches the vector store
        4. Returns relevant documents
        
        ### Formatting Documents
        
        Retriever returns List[Document]. We need to join them:
        
        ```python
def format_docs(docs):
            return "\\n\\n".join(doc.page_content for doc in docs)
        ```
        
        This function is used in the chain: `retriever | format_docs`
        """,
        "code_examples": [get_retrieval_chain_example()],
        "demo": None,
        "quiz": [
            {
                "question": "What does the 'R' in RAG stand for?",
                "options": ["Random", "Retrieval", "Recursive", "Reactive"],
                "correct": "Retrieval",
                "explanation": "RAG = Retrieval-Augmented Generation. First we retrieve relevant documents."
            },
            {
                "question": "Why is RAG better than just asking the LLM directly?",
                "options": [
                    "It's faster",
                    "It's cheaper",
                    "It gives access to private/recent information",
                    "It uses less memory"
                ],
                "correct": "It gives access to private/recent information",
                "explanation": "RAG retrieves from your documents, letting the LLM answer questions about information it wasn't trained on."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 7: TOOLS
    # ============================================================================
    {
        "title": "Tools",
        "subtitle": "Giving LLMs superpowers",
        "emoji": "🛠️",
        "objectives": [
            "Understand what tools are",
            "Create custom tools",
            "Use built-in LangChain tools",
            "Understand tool schemas"
        ],
        "content": """
        ## What Are Tools?
        
        **Tools** are functions that LLMs can call to interact with the outside world.
        
        ### Why Tools?
        
        LLMs are "brain in a jar" - they know things but can't **do** things. Tools let them:
        - Search the web
        - Query databases
        - Call APIs
        - Do calculations
        - Access your files
        
        ### Tool Anatomy
        
        A tool has three parts:
        1. **Name**: What to call it (e.g., "get_weather")
        2. **Description**: When to use it (e.g., "Get weather for a city")
        3. **Function**: The actual code that runs
        
        ### Creating a Tool
        
        The easiest way is using the `@tool` decorator:
        
        ```python
        from langchain.tools import tool

        @tool
        def get_weather(city: str) -> str:
            '''Get the current weather for a city.'''
            # In real code, call a weather API
            return f"The weather in {city} is sunny!"
        
        # The LLM sees:
        # Name: get_weather
        # Description: Get the current weather for a city.
        # Args: city (str)
        ```
        
        ### Built-in Tools
        
        LangChain has many pre-built tools:
        
        ```python
        from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
        
        search = DuckDuckGoSearchRun()
        result = search.run("Python programming language")
        ```
        
        ### Tool Schemas
        
        Tools use **function calling** (also called tool calling):
        1. LLM decides it needs a tool
        2. LLM outputs JSON with arguments
        3. LangChain parses JSON and calls your function
        4. Result goes back to LLM
        5. LLM generates final response
        
        ### Example Flow
        
        ```
        User: "What's the weather in Paris?"
        
        LLM: I need to use get_weather tool.
        Output: {"city": "Paris"}
        
        [Tool executes: get_weather(city="Paris")]
        [Returns: "Sunny, 25°C"]
        
        LLM: The weather in Paris is sunny and 25°C.
        ```
        """,
        "code_examples": [
            {
                "title": "🛠️ Creating and Using Tools",
                "code": '''from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. Define your tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Safe in this controlled context
        return f"Result: {result}"
    except:
        return "Invalid expression"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. Create tool list
tools = [calculator, get_current_time]

# 3. Create LLM that can use tools
llm = ChatOpenAI(model="gpt-4o-mini")

# 4. Create agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 5. Run
result = agent_executor.invoke({"input": "What is 123 * 456?"})
print(result["output"])''',
                "explanation": """
                **Tools** let LLMs perform actions and access external data.
                
                **Key Points:**
                - `@tool` decorator automatically creates the tool schema from function signature
                - Docstring becomes the tool description (LLM uses this to decide when to call it)
                - `create_tool_calling_agent` builds an agent that can use tools
                - `AgentExecutor` runs the agent loop (decide → call tool → observe → respond)
                """,
                "key_concepts": ["@tool decorator", "Tool Calling", "Agent", "AgentExecutor"],
                "line_explanations": {
                    "5": "@tool decorator converts a function into a LangChain tool",
                    "6": "Function docstring becomes the tool description - critical for LLM to know when to use it",
                    "7": "Type hints (str) tell the LLM what argument type to provide",
                    "20": "tools list is passed to the agent so it knows what actions are available",
                    "32": "create_tool_calling_agent: Creates an agent using the modern tool-calling API",
                    "33": "AgentExecutor: Runs the thought → action → observation loop"
                },
                "references": {
                    "@tool": {
                        "type": "Decorator",
                        "description": "Converts a Python function into a LangChain Tool with automatic schema detection",
                        "example": "@tool\\ndef my_func(arg: str) -> str: ..."
                    },
                    "create_tool_calling_agent": {
                        "type": "Function",
                        "description": "Creates an agent using OpenAI-style tool calling. More reliable than ReAct.",
                        "example": "agent = create_tool_calling_agent(llm, tools, prompt)"
                    },
                    "AgentExecutor": {
                        "type": "Class",
                        "description": "Executes the agent loop: LLM decides → Tool runs → Result to LLM → Repeat or respond",
                        "example": "executor = AgentExecutor(agent=agent, tools=tools)"
                    }
                },
                "practice": {
                    "starter_code": '''from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# TODO: Create a tool that searches a list of books

books = [
    {"title": "Python Crash Course", "author": "Eric Matthes"},
    {"title": "Clean Code", "author": "Robert Martin"},
    {"title": "The Pragmatic Programmer", "author": "Andrew Hunt"}
]

@tool
def search_books(query: str) -> str:
    """???"""  # Write a description
    # Your code here: search books by title or author
    pass

# Create agent with your tool
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_books]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful librarian."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Test it
result = executor.invoke({"input": "Find books by Eric Matthes"})
print(result["output"])
'''
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "What does the `@tool` decorator do?",
                "options": [
                    "Makes the function run faster",
                    "Converts a function to a LangChain Tool",
                    "Caches the function result",
                    "Makes the function async"
                ],
                "correct": "Converts a function to a LangChain Tool",
                "explanation": "The @tool decorator wraps a function and creates the necessary schema for LLM tool calling."
            },
            {
                "question": "Why is the docstring important for tools?",
                "options": [
                    "It's required by Python",
                    "The LLM uses it to decide when to call the tool",
                    "It makes code pretty",
                    "It's needed for type checking"
                ],
                "correct": "The LLM uses it to decide when to call the tool",
                "explanation": "The LLM sees the tool name and description to determine if it should use that tool for a given query."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 8: AGENTS
    # ============================================================================
    {
        "title": "Agents",
        "subtitle": "LLMs that think and act",
        "emoji": "🤖",
        "objectives": [
            "Understand agent architecture",
            "Build ReAct agents",
            "Use create_tool_calling_agent",
            "Understand agent loops"
        ],
        "content": """
        ## What Are Agents?
        
        **Agents** are LLM systems that can:
        1. **Reason** about what to do
        2. **Decide** which actions to take
        3. **Execute** those actions (via tools)
        4. **Observe** results and continue
        
        ### Agent vs Simple Chain
        
        **Chain:**
        ```
        Input → Step 1 → Step 2 → Step 3 → Output
        ```
        Fixed path, no decisions.
        
        **Agent:**
        ```
        Input → Think → Act → Observe → Think → Act → Output
        ```
        Dynamic path, decides what to do next.
        
        ### The ReAct Pattern
        
        **ReAct** = Reasoning + Acting
        
        ```
        Thought: I need to find the weather in Paris to answer this question.
        Action: get_weather(city="Paris")
        Observation: "Sunny, 25°C"
        Thought: I now have the weather information. I can answer the user.
        Final Answer: The weather in Paris is sunny and 25°C.
        ```
        
        ### Agent Components
        
        1. **LLM**: The "brain" that makes decisions
        2. **Tools**: Actions the agent can take
        3. **Prompt**: Instructions for the agent
        4. **Executor**: Runs the think-act-observe loop
        
        ### Types of Agents
        
        | Agent Type | Best For | Description |
        |------------|----------|-------------|
        | `create_tool_calling_agent` | Modern OpenAI models | Uses function calling API, very reliable |
        | `create_react_agent` | General use | Classic ReAct pattern with thought/action/observation |
        | `create_structured_chat_agent` | Multi-input tools | Supports tools with complex schemas |
        
        ### Creating an Agent
        
        ```python
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        
        # 1. Create LLM
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 2. Define tools
        tools = [search, calculator, get_weather]
        
        # 3. Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # 4. Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # 5. Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True  # See the agent's thought process!
        )
        
        # 6. Run
        result = agent_executor.invoke({"input": "What's 25 * 4 + weather in Paris?"})
        ```
        
        ### The Agent Loop
        
        1. User provides input
        2. Agent thinks: "Do I need a tool?"
        3. If yes: Calls tool, gets observation, goes back to step 2
        4. If no: Generates final answer
        
        **verbose=True** lets you see each step!
        """,
        "code_examples": [
            {
                "title": "🤖 Building a ReAct Agent",
                "code": '''from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Define tools
@tool
def search(query: str) -> str:
    """Search the web for information."""
    # In real use: return search_results
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

tools = [search, calculator]

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# ReAct prompt template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows the thought process!
    handle_parsing_errors=True
)

# Run
result = agent_executor.invoke({"input": "What is 123 * 456?"})''',
                "explanation": """
                **ReAct** (Reason + Act) is the classic agent pattern.
                
                **How it works:**
                1. Agent sees the prompt template with tool descriptions
                2. LLM outputs text following the format: Thought → Action → Observation
                3. Agent parses the Action and calls the tool
                4. Observation is added to the prompt
                5. Loop continues until Final Answer
                
                **Note:** Modern agents use `create_tool_calling_agent` instead, 
                which uses OpenAI's native function calling and is more reliable.
                """,
                "key_concepts": ["ReAct", "Agent Loop", "Thought-Action-Observation", "verbose"],
                "line_explanations": {
                    "30": "ReAct prompt template - instructs LLM on the format to use",
                    "32": "{tools} placeholder - gets replaced with tool descriptions",
                    "34": "{tool_names} placeholder - gets replaced with available tool names",
                    "36": "Format instructions tell LLM how to structure its reasoning",
                    "53": "create_react_agent: Classic ReAct agent using text parsing",
                    "57": "verbose=True: Prints each thought/action/observation step"
                },
                "references": {
                    "create_react_agent": {
                        "type": "Function",
                        "description": "Classic ReAct agent that parses text output for actions",
                        "example": "agent = create_react_agent(llm, tools, prompt)"
                    },
                    "AgentExecutor": {
                        "type": "Class",
                        "description": "Runs the agent loop until final answer or max iterations",
                        "example": "executor = AgentExecutor(agent=agent, tools=tools, verbose=True)"
                    },
                    "verbose": {
                        "type": "Parameter",
                        "description": "When True, prints each step of the agent's reasoning",
                        "example": "AgentExecutor(..., verbose=True)"
                    }
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "What does ReAct stand for?",
                "options": ["Read and Execute", "Reason and Act", "Recursive Action", "Reactive Agent"],
                "correct": "Reason and Act",
                "explanation": "ReAct = Reasoning + Acting. The agent reasons about what to do, then takes action."
            },
            {
                "question": "What does verbose=True do in AgentExecutor?",
                "options": [
                    "Makes the agent smarter",
                    "Shows the agent's thought process",
                    "Increases token limit",
                    "Enables async execution"
                ],
                "correct": "Shows the agent's thought process",
                "explanation": "verbose=True prints each step: Thought → Action → Observation, so you can debug the agent."
            }
        ]
    }
]
