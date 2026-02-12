"""
Code Explorer Component - Right-side panel for code explanations

This module provides detailed code annotation and explanation functionality
for the learning system. Every code example includes:
- What: The code itself
- Why: Why this approach is used
- How: Detailed explanation of each component
- Variables: Explanation of key variables and functions
"""

import streamlit as st
from typing import List, Dict, Any


def render_code_explorer(code_examples: List[Dict[str, Any]]):
    """
    Render the code explorer panel on the right side.
    
    Args:
        code_examples: List of code example dictionaries containing:
            - title: Code section title
            - code: The actual code
            - explanation: General explanation
            - line_explanations: Dict of line numbers to explanations
            - key_concepts: List of key concepts covered
    """
    st.markdown("""
    <div style="background: #0d1117; border-radius: 10px; padding: 15px; height: 100%;">
        <h3 style="color: #667eea; margin-bottom: 15px;">💻 Code Explorer</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not code_examples:
        st.info("📚 Select a chapter to see code examples here.")
        return
    
    # Code example selector
    example_titles = [ex['title'] for ex in code_examples]
    selected_idx = st.selectbox(
        "Select code example:",
        range(len(example_titles)),
        format_func=lambda i: example_titles[i],
        key="code_selector"
    )
    
    example = code_examples[selected_idx]
    
    # Key Concepts Tags
    if 'key_concepts' in example:
        st.markdown("**🏷️ Key Concepts:**")
        tags_html = ""
        for concept in example['key_concepts']:
            tags_html += f"""
            <span style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 3px 10px; border-radius: 12px; 
                        font-size: 11px; margin: 2px; display: inline-block;">
                {concept}
            </span>
            """
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("---")
    
    # The Code
    st.markdown("**📝 The Code:**")
    st.code(example['code'], language='python')
    
    # Explanation
    with st.expander("📖 Why This Code?", expanded=True):
        st.markdown(example.get('explanation', 'No explanation available.'))
    
    # Line-by-Line Breakdown
    if 'line_explanations' in example:
        with st.expander("🔍 Line-by-Line Breakdown"):
            lines = example['code'].split('\n')
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                    
                # Check if we have an explanation for this line
                explanation = example['line_explanations'].get(str(i)) or \
                             example['line_explanations'].get(line_stripped)
                
                if explanation:
                    col1, col2 = st.columns([0.6, 0.4])
                    with col1:
                        st.code(line, language='python')
                    with col2:
                        st.caption(f"💡 {explanation}")
                else:
                    st.code(line, language='python')
    
    # Variable/Function Reference
    if 'references' in example:
        with st.expander("📚 Reference Guide"):
            for ref_name, ref_info in example['references'].items():
                st.markdown(f"**`{ref_name}`** - {ref_info['type']}")
                st.caption(ref_info['description'])
                if 'example' in ref_info:
                    st.code(ref_info['example'], language='python')
                st.markdown("---")
    
    # Try It Yourself
    if 'practice' in example:
        with st.expander("🧪 Try It Yourself"):
            st.markdown("Modify the code below and run it:")
            user_code = st.text_area(
                "Your code:",
                value=example['practice'].get('starter_code', example['code']),
                height=200,
                key=f"practice_{selected_idx}"
            )
            
            if st.button("▶️ Run Code", key=f"run_{selected_idx}"):
                st.info("💡 In a real implementation, this would execute the code safely.")
                st.code(user_code, language='python')


def render_concept_card(concept: Dict[str, Any]):
    """
    Render a concept explanation card.
    
    Args:
        concept: Dictionary containing:
            - title: Concept name
            - description: Brief description
            - analogy: Real-world analogy
            - code_example: Optional code example
    """
    st.markdown(f"""
    <div style="background: #1e2130; border: 1px solid #2d3748; 
                border-radius: 12px; padding: 20px; margin: 10px 0;">
        <h4 style="color: #667eea;">💡 {concept['title']}</h4>
        <p>{concept['description']}</p>
        {f"<p style='color: #a0aec0; font-style: italic;'>🌍 Analogy: {concept['analogy']}</p>" if 'analogy' in concept else ""}
    </div>
    """, unsafe_allow_html=True)
    
    if 'code_example' in concept:
        st.code(concept['code_example'], language='python')


# ============================================================================
# PRE-BUILT CODE EXAMPLE TEMPLATES
# ============================================================================

def get_document_loader_example() -> Dict[str, Any]:
    """Return a code example for document loading."""
    return {
        "title": "📄 Loading Documents with LangChain",
        "code": '''from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Load a text file
text_loader = TextLoader("document.txt")
text_docs = text_loader.load()

# Load a PDF file
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

print(f"Loaded {len(text_docs)} text documents")
print(f"Loaded {len(pdf_docs)} PDF pages")''',
        "explanation": """
        **Document Loaders** are the entry point for bringing external data into your AI application.
        
        **Why this approach?**
        - LangChain provides specialized loaders for different file formats
        - Each loader handles the complexity of parsing that specific format
        - All loaders return a standard `Document` object for consistency
        
        **Key Insight:** Think of loaders as translators - they convert file formats into a language your AI understands.
        """,
        "key_concepts": ["Document Loaders", "TextLoader", "PyPDFLoader", "Document Object"],
        "line_explanations": {
            "1": "Import specific loaders from langchain_community (community-maintained integrations)",
            "4": "TextLoader: Simple loader for .txt, .md files. Reads entire file as one document.",
            "5": "load() method reads the file and returns a list of Document objects",
            "8": "PyPDFLoader: Handles PDF files, returns one Document per page",
            "11": "Each Document has .page_content (text) and .metadata (source info)"
        },
        "references": {
            "TextLoader": {
                "type": "Class",
                "description": "Loads plain text files. Good for .txt, .md, .csv files.",
                "example": "loader = TextLoader('file.txt', encoding='utf-8')"
            },
            "PyPDFLoader": {
                "type": "Class", 
                "description": "Extracts text from PDF files using PyPDF2 or pdfminer.",
                "example": "loader = PyPDFLoader('file.pdf', extract_images=False)"
            },
            "load()": {
                "type": "Method",
                "description": "Executes the loading process and returns List[Document]",
                "example": "documents = loader.load()  # Returns list of Document objects"
            },
            "Document": {
                "type": "Object",
                "description": "Core data structure with page_content (str) and metadata (dict)",
                "example": "doc.page_content  # The text\ndoc.metadata['source']  # File path"
            }
        },
        "practice": {
            "starter_code": '''from langchain_community.document_loaders import TextLoader

# TODO: Load a text file and print the first 100 characters
loader = TextLoader("your_file.txt")
docs = loader.load()

# Your code here:
'''
        }
    }


def get_text_splitter_example() -> Dict[str, Any]:
    """Return a code example for text splitting."""
    return {
        "title": "✂️ Splitting Text with RecursiveCharacterTextSplitter",
        "code": '''from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the splitter with parameters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Target size of each chunk
    chunk_overlap=200,      # Overlap between chunks for context
    length_function=len,    # How to measure length (characters)
    separators=["\\n\\n", "\\n", " ", ""]  # Priority order of split points
)

# Split documents into chunks
chunks = splitter.split_documents(documents)

# Or split raw text
text_chunks = splitter.split_text(long_text_string)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")''',
        "explanation": """
        **Text Splitting (Chunking)** is crucial because LLMs have context limits. 
        We break large documents into smaller pieces while preserving meaning.
        
        **Why RecursiveCharacterTextSplitter?**
        - Tries to split on paragraph boundaries first (\\n\\n)
        - Falls back to sentence boundaries (\\n)
        - Then words (space), then characters
        - This keeps semantically related text together
        
        **The Overlap Strategy:**
        - chunk_overlap ensures context isn't lost at boundaries
        - If chunk 1 ends with "The quick brown", chunk 2 starts with "brown fox jumped"
        - This maintains continuity for retrieval
        """,
        "key_concepts": ["Chunking", "RecursiveCharacterTextSplitter", "chunk_size", "chunk_overlap", "separators"],
        "line_explanations": {
            "4": "chunk_size: Maximum characters per chunk. Common values: 500-2000.",
            "5": "chunk_overlap: Characters shared between consecutive chunks. Usually 10-20% of chunk_size.",
            "6": "length_function: How to measure chunk size. 'len' = character count.",
            "7": "separators: Priority list. Tries first separator, if still too big, tries next.",
            "11": "split_documents(): Takes List[Document], returns List[Document] with smaller chunks",
            "14": "split_text(): Takes a string, returns List[str] of text chunks"
        },
        "references": {
            "RecursiveCharacterTextSplitter": {
                "type": "Class",
                "description": "Intelligently splits text trying to keep paragraphs/sentences together",
                "example": "splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)"
            },
            "chunk_size": {
                "type": "Parameter",
                "description": "Target maximum size for each chunk in characters",
                "example": "chunk_size=1000  # About 250 tokens for most text"
            },
            "chunk_overlap": {
                "type": "Parameter",
                "description": "Number of characters to overlap between chunks to maintain context",
                "example": "chunk_overlap=200  # 20% overlap is common"
            },
            "split_documents()": {
                "type": "Method",
                "description": "Splits a list of Document objects into smaller chunks",
                "example": "chunks = splitter.split_documents(docs)"
            }
        },
        "practice": {
            "starter_code": '''from langchain_text_splitters import RecursiveCharacterTextSplitter

# TODO: Create a splitter that produces chunks of ~500 chars with 50 char overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=?,  # Fill this in
    chunk_overlap=?  # Fill this in
)

# Test with this text
text = "First paragraph here.\\n\\nSecond paragraph here.\\n\\nThird paragraph here."
chunks = splitter.split_text(text)
print(f"Number of chunks: {len(chunks)}")
'''
        }
    }


def get_embedding_example() -> Dict[str, Any]:
    """Return a code example for embeddings."""
    return {
        "title": "🧠 Creating Vector Embeddings",
        "code": '''from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Option 1: Free local embeddings (HuggingFace)
local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Option 2: OpenAI embeddings (API key required)
openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create embeddings for text
texts = ["Hello world", "How are you?", "Machine learning is fascinating"]
vectors = local_embeddings.embed_documents(texts)

# Create embedding for a single query
query_vector = local_embeddings.embed_query("What is AI?")

print(f"Each vector has {len(vectors[0])} dimensions")
print(f"Query vector has {len(query_vector)} dimensions")''',
        "explanation": """
        **Embeddings** convert text into numbers (vectors) that capture semantic meaning.
        Similar texts have similar vectors, allowing semantic search.
        
        **What is a Vector?**
        Think of it as coordinates in a high-dimensional space. Just like (x, y) 
        describes a point on a 2D map, an embedding vector (384 numbers) describes 
        a point in "meaning space."
        
        **Choosing an Embedding Model:**
        - **HuggingFace (Free, Local):** Good for privacy, offline use. all-MiniLM-L6-v2 is 384-dim.
        - **OpenAI (Paid, API):** Higher quality, 1536-dim. Requires API key.
        
        **The Magic:**
        "King - Man + Woman ≈ Queen" works in vector space!
        """,
        "key_concepts": ["Embeddings", "Vectors", "Semantic Search", "HuggingFace", "Cosine Similarity"],
        "line_explanations": {
            "4": "HuggingFaceEmbeddings: Downloads model locally, runs on your CPU/GPU",
            "5": "all-MiniLM-L6-v2: Popular free model, 384 dimensions, fast and decent quality",
            "9": "OpenAIEmbeddings: Cloud-based, requires OPENAI_API_KEY environment variable",
            "14": "embed_documents(): Batch embed multiple texts efficiently",
            "18": "embed_query(): Embed a single search query (same model, same dimensions)"
        },
        "references": {
            "HuggingFaceEmbeddings": {
                "type": "Class",
                "description": "Local embedding model from HuggingFace. Free, privacy-preserving.",
                "example": "embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"
            },
            "embed_documents()": {
                "type": "Method",
                "description": "Converts list of texts to list of vectors. Efficient batch processing.",
                "example": "vectors = embeddings.embed_documents(['text1', 'text2', 'text3'])"
            },
            "embed_query()": {
                "type": "Method",
                "description": "Converts a single query string to a vector for searching",
                "example": "query_vec = embeddings.embed_query('search query')"
            },
            "384 dimensions": {
                "type": "Concept",
                "description": "all-MiniLM-L6-v2 outputs 384 numbers. Each number represents a feature of the text.",
                "example": "vector = [0.023, -0.156, 0.892, ...]  # 384 numbers total"
            }
        },
        "practice": {
            "starter_code": '''from langchain_huggingface import HuggingFaceEmbeddings

# TODO: Create embeddings for these sentences and find the most similar pair
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug", 
    "The stock market crashed today",
    "I love programming in Python"
]

# Your code here:
# 1. Create embeddings for all sentences
# 2. Calculate similarity between pairs
# 3. Find which pair is most similar
'''
        }
    }


def get_vector_store_example() -> Dict[str, Any]:
    """Return a code example for vector stores."""
    return {
        "title": "💾 Storing Vectors in FAISS",
        "code": '''from langchain_community.vectorstores import FAISS

# Create a FAISS index from documents
vectorstore = FAISS.from_documents(
    documents=chunks,           # Your chunked documents
    embedding=embeddings        # Your embedding model
)

# Save the index to disk
vectorstore.save_local("vector_db/my_index")

# Load the index from disk
loaded_vectorstore = FAISS.load_local(
    "vector_db/my_index",
    embeddings,
    allow_dangerous_deserialization=True  # Only load trusted indexes!
)

# Search for similar documents
results = vectorstore.similarity_search(
    query="What is machine learning?",
    k=4  # Return top 4 matches
)

# Search with scores
results_with_scores = vectorstore.similarity_search_with_score(
    query="What is machine learning?",
    k=4
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | Content: {doc.page_content[:100]}...")''',
        "explanation": """
        **Vector Stores** are databases optimized for finding similar vectors.
        Instead of exact keyword matching, they find semantically similar content.
        
        **What is FAISS?**
        Facebook AI Similarity Search - an open-source library for efficient 
        similarity search in high-dimensional spaces. It's fast and runs locally.
        
        **How Similarity Search Works:**
        1. Convert your query to a vector using the same embedding model
        2. FAISS compares this vector to all stored vectors
        3. Returns the k closest vectors using cosine similarity or Euclidean distance
        
        **The Score:**
        - Lower = more similar (for Euclidean distance)
        - Higher = more similar (for cosine similarity)
        - FAISS uses L2 (Euclidean) distance by default
        """,
        "key_concepts": ["FAISS", "Vector Store", "Similarity Search", "Index", "Cosine Similarity"],
        "line_explanations": {
            "4": "from_documents(): Creates FAISS index and embeds all documents in one step",
            "5": "documents: List of Document objects (from your chunks)",
            "6": "embedding: The embedding model instance to use for creating vectors",
            "10": "save_local(): Persists the index to disk so you don't need to recreate it",
            "14": "load_local(): Loads a saved index. Much faster than re-embedding everything!",
            "16": "allow_dangerous_deserialization=True: Required for loading. Only load trusted files!",
            "21": "similarity_search(): Main search method. Returns List[Document]",
            "22": "k: Number of results to return. Higher k = more context but slower.",
            "26": "similarity_search_with_score(): Returns (Document, score) tuples for ranking"
        },
        "references": {
            "FAISS": {
                "type": "Class",
                "description": "Facebook's vector similarity search library. Fast, local, open-source.",
                "example": "vectorstore = FAISS.from_documents(docs, embeddings)"
            },
            "from_documents()": {
                "type": "Method",
                "description": "Creates FAISS index from documents. Embeds them automatically.",
                "example": "FAISS.from_documents(chunks, embedding_model)"
            },
            "similarity_search()": {
                "type": "Method",
                "description": "Finds k most similar documents to the query string",
                "example": "docs = vectorstore.similarity_search('query', k=4)"
            },
            "k parameter": {
                "type": "Parameter",
                "description": "Number of top results to return. More results = more context for LLM.",
                "example": "k=4  # Return top 4 most similar chunks"
            }
        },
        "practice": {
            "starter_code": '''from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# TODO: Create a FAISS index and search it
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample chunks (normally these come from your documents)
chunks = [
    Document(page_content="Python is a programming language"),
    Document(page_content="JavaScript runs in browsers"),
    Document(page_content="Python is great for data science"),
    Document(page_content="HTML is a markup language")
]

# Your code here:
# 1. Create FAISS index from chunks
# 2. Search for "programming"
# 3. Print the top 2 results
'''
        }
    }


def get_chain_example() -> Dict[str, Any]:
    """Return a code example for LCEL chains."""
    return {
        "title": "⛓️ Building Chains with LCEL",
        "code": '''from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Create a prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
If you can't answer, say "I don't know".

Context: {context}

Question: {question}

Answer:""")

# 2. Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1  # Lower = more deterministic
)

# 3. Create output parser
output_parser = StrOutputParser()

# 4. Build the chain using LCEL (| operator)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

# 5. Run the chain
response = chain.invoke("What is LangChain?")
print(response)''',
        "explanation": """
        **LCEL (LangChain Expression Language)** lets you compose components with the pipe (`|`) operator.
        It's like Unix pipes - output of one step becomes input to the next.
        
        **The Chain Flow:**
        ```
        Input Question → Retrieve Context → Format Prompt → LLM → Parse Output
        ```
        
        **Why LCEL?**
        - Clean, readable syntax
        - Automatic streaming support
        - Parallel execution where possible
        - Easy to debug and modify
        
        **RunnablePassthrough:**
        Passes the input through unchanged. Used here to send the question 
        directly to the prompt while the retriever fetches context.
        """,
        "key_concepts": ["LCEL", "Chains", "Prompt Templates", "RunnablePassthrough", "Pipes"],
        "line_explanations": {
            "5": "ChatPromptTemplate: Template with {placeholders} for dynamic values",
            "18": "ChatOpenAI: Wrapper for OpenAI's chat models (gpt-4, gpt-3.5-turbo)",
            "19": "temperature: 0=deterministic, 1=very creative. RAG usually uses low values.",
            "25": "RunnablePassthrough: Passes input unchanged. Here it passes the user's question.",
            "27": "retriever: Your vectorstore.as_retriever() - fetches relevant documents",
            "28": "prompt: Formats the retrieved context and question into the final prompt",
            "29": "llm: Sends the formatted prompt to the language model",
            "30": "output_parser: Extracts the text from the LLM's response object",
            "34": "invoke(): Runs the entire chain with one input string"
        },
        "references": {
            "ChatPromptTemplate": {
                "type": "Class",
                "description": "Template for chat models with system/human/AI message roles",
                "example": "prompt = ChatPromptTemplate.from_template('Hello {name}')"
            },
            "RunnablePassthrough": {
                "type": "Class",
                "description": "Passes input through unchanged. Useful for parallel branches in chains.",
                "example": "{'question': RunnablePassthrough(), 'context': retriever}"
            },
            "| (pipe)": {
                "type": "Operator",
                "description": "LCEL pipe operator. Chains components together.",
                "example": "chain = prompt | llm | output_parser"
            },
            "invoke()": {
                "type": "Method",
                "description": "Executes the chain with input. Returns the final output.",
                "example": "result = chain.invoke('input question')"
            }
        },
        "practice": {
            "starter_code": '''from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# TODO: Build a simple chain that translates English to French

# 1. Create a prompt template
prompt = ChatPromptTemplate.from_template("""
???  # Create a template that translates {text} to French
""")

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Create output parser
output_parser = StrOutputParser()

# 4. Build the chain
chain = ???  # Use the | operator

# 5. Test it
result = chain.invoke({"text": "Hello, how are you?"})
print(result)
'''
        }
    }


def get_retrieval_chain_example() -> Dict[str, Any]:
    """Return a code example for retrieval chains."""
    return {
        "title": "🔍 Building a RAG Chain",
        "code": '''from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" for diversity
    search_kwargs={"k": 4}     # Return 4 documents
)

# 3. Create prompt
template = """Answer based on this context:
{context}

Question: {question}

If the context doesn't contain the answer, say "I don't know"."""
prompt = ChatPromptTemplate.from_template(template)

# 4. Format docs function
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# 5. Build RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Use it
answer = rag_chain.invoke("What is RAG?")
print(answer)''',
        "explanation": """
        **RAG (Retrieval-Augmented Generation)** combines vector search with LLMs.
        The system retrieves relevant documents, then uses them to answer questions.
        
        **The RAG Pipeline:**
        1. User asks a question
        2. System embeds the question and searches vector store
        3. Top-k documents are retrieved
        4. Documents + question are formatted into a prompt
        5. LLM generates answer based on the context
        
        **Key Components:**
        - **Retriever:** Fetches relevant documents from vector store
        - **format_docs:** Joins multiple documents into a single context string
        - **Prompt:** Instructs LLM to use the context
        
        **Why RAG Works:**
        - LLM only "knows" its training data
        - RAG gives it access to your private/custom documents
        - Reduces hallucinations by grounding answers in facts
        """,
        "key_concepts": ["RAG", "Retriever", "as_retriever", "format_docs", "Augmented Generation"],
        "line_explanations": {
            "14": "as_retriever(): Converts vectorstore into a retriever interface",
            "15": "search_type='similarity': Standard similarity search. 'mmr' adds diversity.",
            "16": "search_kwargs: Additional options like k (number of results)",
            "24": "format_docs(): Custom function to combine multiple docs into one context string",
            "28": "retriever | format_docs: Retrieves docs, then formats them. Result goes to {context}",
            "28": "RunnablePassthrough(): Passes the original question through to {question}"
        },
        "references": {
            "as_retriever()": {
                "type": "Method",
                "description": "Creates a retriever interface from vectorstore for use in chains",
                "example": "retriever = vectorstore.as_retriever(search_kwargs={'k': 4})"
            },
            "search_type": {
                "type": "Parameter",
                "description": "'similarity'=standard search, 'mmr'=Max Marginal Relevance for diverse results",
                "example": "search_type='mmr'  # Gets diverse results, not just most similar"
            },
            "format_docs": {
                "type": "Function",
                "description": "Helper to join multiple documents. You define this yourself.",
                "example": "def format_docs(docs): return '\\n'.join(d.page_content for d in docs)"
            },
            "k": {
                "type": "Parameter",
                "description": "Number of documents to retrieve. More = more context, but costs more tokens.",
                "example": "k=4  # Retrieve top 4 most relevant chunks"
            }
        },
        "practice": {
            "starter_code": '''# TODO: Build a complete RAG pipeline

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup components (use dummy data if needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = ... (load or create your index)
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Create retriever
retriever = ???

# 3. Create prompt
template = """Use this context: {context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 4. Build chain
rag_chain = ???

# 5. Test
answer = rag_chain.invoke("Your question here")
print(answer)
'''
        }
    }
