import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Tuple
from langchain_core.documents import Document

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
            return [q.strip() for q in response.split("\n") if q.strip()]
        except Exception as e:
            print(f"Error generating multi-queries: {e}")
            return [query]

    def decompose_query(self, query: str) -> List[str]:
        """
        Breaks a complex query into simpler sub-questions.
        """
        prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant that breaks down complex user questions into simpler sub-questions. "
            "Decompose the following complex query into a logical sequence of simpler steps or questions. "
            "Provide these as a newline-separated list.\n\n"
            "Complex Query: {query}"
        )
        chain = prompt | self.llm | self.output_parser
        try:
            response = chain.invoke({"query": query})
            return [q.strip() for q in response.split("\n") if q.strip()]
        except Exception as e:
            print(f"Error decomposing query: {e}")
            return [query]

    def generate_hyde_document(self, query: str) -> str:
        """
        Generates a hypothetical document based on the query (HyDE pattern).
        """
        prompt = ChatPromptTemplate.from_template(
            "Please write a short, one-paragraph technical answer to the following question. "
            "Even if you are unsure, provide a hypothetical response that looks like a valid document. "
            "This will be used to help search for real documents.\\n\\nQuestion: {query}"
        )
        chain = prompt | self.llm | self.output_parser
        try:
            return chain.invoke({"query": query})
        except Exception as e:
            print(f"Error generating HyDE document: {e}")
            return query

    def generate_summary(self, context: str, query: str) -> Tuple[str, List[Dict]]:
        """
        Generates a summary from retrieved documents, with optional caching.
        """
        # 1. Check Cache
        cached = self.cache.lookup(query)
        if cached:
            return cached, [{"message": "⚡ **Semantic Cache Hit!** Returning instant response.", "code": f"cache.lookup('{query}')"}]

        # 2. Proceed with LLM if no cache hit
        prompt = ChatPromptTemplate.from_template(
            "Use the following pieces of context to answer the user's question. "
            "If you don't know the answer, just say that you don't know. "
            "Context: {context}\\nQuestion: {query}\\nAnswer:"
        )
        chain = prompt | self.llm | self.output_parser
        traces = [{"message": "🤖 **Cache Miss.** Calling LLM for fresh reasoning...", "code": ""}]
        try:
            summary = chain.invoke({"context": context, "query": query})
            self.cache.update(query, summary)
            return summary, traces
        except Exception as e:
            traces.append({"message": f"❌ **Summary generation failed:** {e}", "code": ""})
            return "", traces

    def classify_query(self, query: str) -> str:
        """
        Classifies a user query to decide on routing.
        """
        prompt = ChatPromptTemplate.from_template(
            "Classify the following user query into one of these categories: "
            "['GREETING', 'TECHNICAL_QUESTION', 'DATABASE_SEARCH'].\\n"
            "Respond with ONLY the category name.\\n\\nQuery: {query}"
        )
        chain = prompt | self.llm | self.output_parser
        try:
            category = chain.invoke({"query": query}).strip().upper()
            return category if category in ['GREETING', 'TECHNICAL_QUESTION', 'DATABASE_SEARCH'] else 'DATABASE_SEARCH'
        except Exception as e:
            print(f"Error classifying query: {e}")
            return 'DATABASE_SEARCH'
