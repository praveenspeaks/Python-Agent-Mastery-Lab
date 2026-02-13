import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Tuple, Optional, Any
from langchain_core.documents import Document
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

# Load environment variables
load_dotenv()

# Load environment variables
load_dotenv()

class SemanticCache:
    """
    A simple in-memory semantic cache simulator.
    In production, this would use FAISS/Redis.
    """
    def __init__(self):
        self.cache: Dict[str, str] = {
            "hello": "Hello! I am your AI Mastery assistant. How can I help you today?",
            "what is rag?": "RAG stands for Retrieval-Augmented Generation. It's a technique to give LLMs access to specific, private data."
        }

    def lookup(self, query: str) -> str:
        # Simple exact match for demo, imagine fuzzy semantic match here
        return self.cache.get(query.lower().strip())

    def update(self, query: str, response: str):
        self.cache[query.lower().strip()] = response

class AIModelHandler:
    """
    Handles interactions with AI models via OpenRouter.
    """

    def __init__(self, model_name: str = None):
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        default_model = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")
        
        if not api_key:
            print("⚠️ WARNING: OPENROUTER_API_KEY not found. AI features will fail.")
            # We still initialize to avoid crashing immediately, but invoking will fail.
        
        self.model_name = model_name or default_model
        
        # OpenRouter is compatible with OpenAI API
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=api_key or "dummy-key-to-prevent-init-crash",
            openai_api_base=base_url,
            default_headers={
                "HTTP-Referer": "https://localhost:3000", # Required by OpenRouter
                "X-Title": "LangChain Learning Project",   # Required by OpenRouter
            }
        )
        self.output_parser = StrOutputParser()
        self.langfuse_handler = self._init_langfuse()
        self.last_run_traces = []
        
        # Immediate Trace for Initialization
        self.last_run_traces.append({
            "step": "AI Model Handler Initialization",
            "module": "src.ai_model.AIModelHandler",
            "command": f"ChatOpenAI(model='{self.model_name}', openai_api_base='{base_url}')",
            "variables": {
                "model_name": self.model_name,
                "base_url": base_url,
                "api_key": "SET" if api_key else "MISSING"
            },
            "input": "N/A",
            "output": f"Initialized ChatOpenAI with {self.model_name}",
            "explanation": "Connecting to the LLM provider (OpenRouter) with standard OpenAI compatibility."
        })

    def _init_langfuse(self):
        """Initializes the Langfuse callback handler if keys are present."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        if public_key and secret_key:
            return LangfuseCallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
        return None

    def get_callbacks(self, trace_name: str = "Learning Lab Task"):
        """Returns a list of callbacks for LangChain runs."""
        if self.langfuse_handler:
            return [self.langfuse_handler]
        return []

    def validate_setup(self):
        """Checks if the API key is set."""
        if not os.getenv("OPENROUTER_API_KEY"):
            return False, "❌ OPENROUTER_API_KEY is missing. Please set it in your .env file."
        return True, "✅ AI Engine utilizes OpenRouter/OpenAI API."

    def summarize_chunks(self, chunks: List[Document]):
        """
        Summarizes a list of chunks into a single summary.
        Returns (summary, traces)
        """
        traces = []
        if not chunks:
            return "No content to summarize.", [{"message": "⚠️ No chunks found.", "code": ""}]

        traces.append({
            "message": f"🧠 **AI Initialization: Loading Model Engine**",
            "code": (
                "from langchain_openai import ChatOpenAI\n"
                f"llm = ChatOpenAI(\n"
                f"    model='{self.llm.model_name}',\n"
                "    temperature=0.7 # Creativity control\n"
                ")\n"
                "# We are using an OpenAI-compatible interface locally or via OpenRouter."
            )
        })
        
        traces.append({
            "message": "📝 **LCEL Pipeline Construction** (The 'Chain')",
            "code": (
                "# LangChain Expression Language (LCEL) uses the Unix pipe '|' syntax.\n"
                "# It allows us to compose primitives into complex chains.\n\n"
                "template = 'Summarize: {text}'\n"
                "prompt = ChatPromptTemplate.from_template(template)\n"
                "output_parser = StrOutputParser() # Extracts string from AI Message objects\n\n"
                "# THE CHAIN DEFINITION:\n"
                "chain = prompt | llm | output_parser\n"
                "# Data flows:  Dictionary -> Prompt -> Model -> String"
            )
        })

        combined_text = "\n".join([doc.page_content for doc in chunks[:10]])
        chain = (ChatPromptTemplate.from_template("Summarize the following text briefly and concisely:\n\n{text}") 
                 | self.llm 
                 | self.output_parser)
        
        try:
            summary = chain.invoke({"text": combined_text})
            traces.append({
                "message": "✅ **Chain Execution Success**", 
                "code": (
                    "# Invoking the chain sends the payload to the API\n"
                    "response = chain.invoke({'text': combined_text})\n"
                    "# Result is pure text, stripped of 'role' and 'metadata'."
                )
            })
            return summary, traces
        except Exception as e:
            # Assuming 'traces' is intended to be 'all_logs' here, and the return type for summary on error is an empty string.
            # The original code returned str(e) as summary, but the instruction implies an empty list for summary.
            # Sticking to the original return type for summary (str) and using traces for logs.
            traces.append({"message": f"❌ **Compression failed:** {e}", "code": ""})
            return "", traces # Returning empty string for summary, and the traces.

    def ask_about_document(self, query: str, chunks: List[Document]) -> str:
        """
        Answers a question based on the provided document chunks.
        """
        context = "\n".join([doc.page_content for doc in chunks])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the provided context to answer the user's question."),
            ("user", "Context: {context}\n\nQuestion: {query}")
        ])
        
        chain = prompt | self.llm | self.output_parser
        
        try:
            return chain.invoke({"context": context, "query": query})
        except Exception as e:
            return f"Error during AI query: {e}"

    def generate_multi_queries(self, query: str, n: int = 3) -> List[str]:
        """
        Generates different versions of a user query to improve retrieval.
        """
        self.last_run_traces = [] # Clear history for new run
        prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant tasked with generating {n} different versions of a user search query. "
            "The goal is to overcome the limitations of distance-based similarity search by providing "
            "multiple perspectives on the same user question. "
            "Provide these alternative queries as a newline-separated list.\n\n"
            "Original Query: {query}"
        )
        chain = prompt | self.llm | self.output_parser
        try:
            response = chain.invoke({"query": query, "n": n})
            queries = [q.strip() for q in response.split("\n") if q.strip()]
            
            # Technical Trace
            self.last_run_traces.append({
                "step": "Multi-Query Generation",
                "module": "src.ai_model.AIModelHandler",
                "command": f"chain = prompt | self.llm | StrOutputParser(); chain.invoke({{'query': '{query}', 'n': {n}}})",
                "variables": {"n_variations": n, "model": self.model_name},
                "input": query,
                "output": queries,
                "explanation": "Generating multiple perspectives of the same question to increase the chance of finding relevant documents (improves Recall)."
            })
            return queries
        except Exception as e:
            print(f"Error generating multi-queries: {e}")
            return [query]

    def decompose_query(self, query: str) -> List[str]:
        """
        Breaks a complex query into simpler sub-questions.
        """
        self.last_run_traces = [] # Clear history for new run
        prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant that breaks down complex user questions into simpler sub-questions. "
            "Decompose the following complex query into a logical sequence of simpler steps or questions. "
            "Provide these as a newline-separated list.\n\n"
            "Complex Query: {query}"
        )
        chain = prompt | self.llm | self.output_parser
        try:
            response = chain.invoke({"query": query})
            sub_queries = [q.strip() for q in response.split("\n") if q.strip()]
            
            # Technical Trace
            self.last_run_traces.append({
                "step": "Query Decomposition",
                "module": "src.ai_model.AIModelHandler",
                "command": "chain = prompt | self.llm | StrOutputParser(); chain.invoke({'query': query})",
                "variables": {"strategy": "Step-by-step reasoning", "model": self.model_name},
                "input": query,
                "output": sub_queries,
                "explanation": "Breaking down a compound question into atomic steps that can be answered sequentially."
            })
            return sub_queries
        except Exception as e:
            print(f"Error decomposing query: {e}")
            return [query]

    def generate_hyde_document(self, query: str) -> str:
        """
        Generates a hypothetical document based on the query.
        """
        self.last_run_traces = [] # Clear history
        prompt = ChatPromptTemplate.from_template(
            "Please write a scientific-style paragraph that answers the question: {query}"
        )
        chain = prompt | self.llm | self.output_parser
        
        try:
            response = chain.invoke({"query": query}, config={"callbacks": self.get_callbacks("HyDE Generation")})
            
            # Capture Trace
            self.last_run_traces.append({
                "step": "HyDE Document Generation",
                "module": "src.ai_model.AIModelHandler",
                "command": "chain = prompt | self.llm | StrOutputParser(); chain.invoke({'query': query})",
                "variables": {"model": self.model_name, "query": query},
                "input": query,
                "output": response,
                "explanation": "Generating a 'ground truth' style paragraph to improve vector search recall. Comparing 'Answer to Answer' (HyDE) is often more accurate than 'Question to Answer'."
            })
            return response
        except Exception as e:
            print(f"Error generating HyDE document: {e}")
            return query

    def generate_summary(self, context: str, query: str) -> Tuple[str, List[Dict]]:
        """
        Generates a summary from retrieved documents, with optional caching.
        """
        self.last_run_traces = [] # Clear history
        # 1. Check Cache
        cached = self.cache.lookup(query)
        if cached:
            self.last_run_traces.append({
                "step": "Semantic Cache Lookup",
                "module": "src.cache.SemanticCache",
                "command": f"cache.lookup('{query}')",
                "variables": {"hit": True},
                "input": query,
                "output": cached,
                "explanation": "Instantly retrieving a previously computed response based on semantic similarity of the query."
            })
            return cached, []

        # 2. Proceed with LLM if no cache hit
        prompt = ChatPromptTemplate.from_template(
            "Use the following pieces of context to answer the user's question. "
            "If you don't know the answer, just say that you don't know. "
            "Context: {context}\nQuestion: {query}\nAnswer:"
        )
        chain = prompt | self.llm | self.output_parser
        
        try:
            summary = chain.invoke({"context": context, "query": query})
            self.cache.update(query, summary)
            
            # Technical Trace
            self.last_run_traces.append({
                "step": "RAG Summary Generation",
                "module": "src.ai_model.AIModelHandler",
                "command": "chain.invoke({'context': context, 'query': query})",
                "variables": {"model": self.model_name, "context_len": len(context)},
                "input": f"Query: {query} | Context: {context[:200]}...",
                "output": summary,
                "explanation": "Synthesizing an answer using the retrieved context window. This is the 'Generation' part of RAG."
            })
            return summary, []
        except Exception as e:
            return "", []

    def classify_query(self, query: str) -> str:
        """
        Classifies a user query to decide on routing.
        """
        self.last_run_traces = [] # Clear history
        prompt = ChatPromptTemplate.from_template(
            "Classify the following user query into one of these categories: "
            "['GREETING', 'TECHNICAL_QUESTION', 'DATABASE_SEARCH'].\n"
            "Respond with ONLY the category name.\n\nQuery: {query}"
        )
        chain = prompt | self.llm | self.output_parser
        try:
            category = chain.invoke({"query": query}).strip().upper()
            
            # Technical Trace
            self.last_run_traces.append({
                "step": "Intent Classification (Routing)",
                "module": "src.ai_model.AIModelHandler",
                "command": "chain.invoke({'query': query})",
                "variables": {"model": self.model_name, "detected_category": category},
                "input": query,
                "output": category,
                "explanation": "Using a cheap LLM call to decide which backend logic (Retrieval, Chat, or Agent) should handle the request."
            })
            return category if category in ['GREETING', 'TECHNICAL_QUESTION', 'DATABASE_SEARCH'] else 'DATABASE_SEARCH'
        except Exception as e:
            print(f"Error classifying query: {e}")
            return 'DATABASE_SEARCH'
