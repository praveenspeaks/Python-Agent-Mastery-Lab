from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from src.parser import DocumentParser
from src.processor import DocumentProcessor
from src.ai_model import AIModelHandler
from src.embeddings import EmbeddingProvider
from src.vector_store import VectorStoreManager
import os

class LangChainAgent:
    """
    Main Orchestrator for Document Parsing, Chunking, and AI Analysis.
    """
    
    def __init__(self, chunk_size=500, chunk_overlap=50, model_name=None, embedding_provider="huggingface"):
        # Chapter 2: Embeddings & Vector Store (Need these for semantic chunking)
        self.embeddings = EmbeddingProvider(provider=embedding_provider)
        
        self.parser = DocumentParser()
        self.processor = DocumentProcessor(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            embeddings=self.embeddings.embeddings
        )
        self.ai = AIModelHandler(model_name=model_name)
        self.vector_store = VectorStoreManager(embedding_provider=self.embeddings)

    def process_file(self, file_path: str, chunking_mode: str = "recursive"):
        """
        Parses a file and converts it into chunks.
        Returns (chunks, educational_logs)
        """
        all_logs = []
        try:
            documents, parse_logs = self.parser.load_document(file_path)
            all_logs.extend(parse_logs)
            chunks, process_logs = self._handle_chunks(documents, mode=chunking_mode)
            all_logs.extend(process_logs)
            return chunks, all_logs
        except Exception as e:
            error_msg = {"message": f"❌ **Error processing file:** {e}", "code": ""}
            all_logs.append(error_msg)
            return None, all_logs

    def process_sql(self, connection_string: str, query: str, summarize: bool = False):
        """
        Parses SQL results and converts them into chunks.
        """
        all_logs = [{"message": "🗄️ **Initiating SQL Query Extraction**", "code": ""}]
        try:
            documents = self.parser.load_sql(connection_string, query)
            all_logs.append({"message": f"✅ **SQL Success:** Retrieved {len(documents)} matching rows.", "code": ""})
            chunks, process_logs = self._handle_chunks(documents)
            all_logs.extend(process_logs)
            return chunks, all_logs
        except Exception as e:
            all_logs.append({"message": f"❌ **Error processing SQL:** {e}", "code": ""})
            return None, all_logs

    def index_file(self, file_path: str):
        """
        Loads, chunks, and indexes a file into the vector store.
        """
        all_logs = [{"message": f"📥 **Indexing File:** `{os.path.basename(file_path)}`", "code": ""}]
        chunks, logs = self.process_file(file_path)
        all_logs.extend(logs)
        
        if chunks:
            self.vector_store.add_documents(chunks)
            all_logs.append({"message": f"🚀 **Successfully indexed {len(chunks)} chunks.**", "code": ""})
        
        return all_logs

    def search_documents(self, query: str, k: int = 4):
        """
        Searches for relevant documents using semantic similarity.
        """
        all_logs = []
        try:
            results = self.vector_store.search(query, k=k)
            all_logs.append({
                "step": "Vector Similarity Search",
                "module": "src.vector_store.VectorStoreManager",
                "command": f"vector_store.search(query='{query}', k={k})",
                "variables": {"query": query, "k": k, "engine": "FAISS"},
                "input": query,
                "output": f"Found {len(results)} matches",
                "explanation": "Converting the query to a vector and measuring cosine similarity against all stored document vectors."
            })
            return results, all_logs
        except Exception as e:
            all_logs.append({
                "step": "Search Failure",
                "module": "src.app.LangChainAgent",
                "command": "search_documents()",
                "variables": {"error": str(e)},
                "input": query,
                "output": "Error",
                "explanation": "The semantic search operation failed, likely due to an uninitialized index."
            })
            return [], all_logs

    def hybrid_search(self, query: str, documents: List[Document], vector_weight: float = 0.5, keyword_weight: float = 0.5):
        """
        Performs a hybrid search combining Vector and Keyword retrieval.
        """
        all_logs = []
        try:
            retriever = self.vector_store.get_hybrid_retriever(
                documents, 
                vector_weight=vector_weight, 
                b_weight=keyword_weight
            )
            
            all_logs.append({
                "step": "Ensemble Retriever Initialization",
                "module": "langchain.retrievers.EnsembleRetriever",
                "command": "EnsembleRetriever(retrievers=[vector, bm25], weights=[v_w, k_w])",
                "variables": {
                    "vector_weight": vector_weight,
                    "keyword_weight": keyword_weight,
                    "fusion_strategy": "Reciprocal Rank Fusion (RRF)"
                },
                "input": f"{len(documents)} Context Docs",
                "output": "Initialized Hybrid Retriever",
                "explanation": "Combining the 'Deep Meaning' of vectors with the 'Exact Match' precision of BM25 (keyword) search."
            })
            
            results = retriever.invoke(query)
            all_logs.append({
                "step": "Hybrid Retrieval Execution",
                "module": "langchain.retrievers.EnsembleRetriever",
                "command": f"retriever.invoke('{query}')",
                "variables": {"result_count": len(results)},
                "input": query,
                "output": f"Ranked List of {len(results)} docs",
                "explanation": "Executing search across both internal indices and merging the results based on Reciprocal Rank Fusion."
            })
            return results, all_logs
        except Exception as e:
            all_logs.append({
                "step": "Hybrid Search Failure",
                "module": "src.app.LangChainAgent",
                "command": "hybrid_search()",
                "variables": {"error": str(e)},
                "input": query,
                "output": "Error",
                "explanation": "The hybrid search failed."
            })
            return [], all_logs

    def enhance_query(self, query: str, mode: str = "multi_query"):
        """
        Enhances a user query using AI strategies like Multi-Query or Decomposition.
        """
        all_logs = [{"message": f"✨ **Initiating Query Enhancement: `{mode}`**", "code": ""}]
        try:
            if mode == "multi_query":
                queries = self.ai.generate_multi_queries(query)
                all_logs.append({
                    "message": "🔄 **Multi-Query Result:** Generated alternative angles for better search recall.",
                    "code": "\\n".join([f"# Query {i+1}: {q}" for i, q in enumerate(queries)])
                })
                return queries, all_logs
            elif mode == "decomposition":
                sub_queries = self.ai.decompose_query(query)
                all_logs.append({
                    "message": "🧩 **Decomposition Result:** Broke complex goal into simpler sub-tasks.",
                    "code": "\\n".join([f"# Step {i+1}: {q}" for i, q in enumerate(sub_queries)])
                })
                return sub_queries, all_logs
            return [query], all_logs
        except Exception as e:
            all_logs.append({"message": f"❌ **Enhancement failed:** {e}", "code": ""})
            return [query], all_logs

    def hyde_search(self, query: str):
        """
        Performs HyDE search: Generate fake doc -> Search with fake doc.
        """
        all_logs = [{"message": "🔍 **Initiating HyDE Search**", "code": ""}]
        try:
            hyde_doc = self.ai.generate_hyde_document(query)
            all_logs.append({
                "message": "🏗️ **Hypothetical Document Generated:** Using this to search instead of the question.",
                "code": f"# HyDE Document:\\n{hyde_doc}"
            })
            return hyde_doc, all_logs
        except Exception as e:
            all_logs.append({"message": f"❌ **HyDE failed:** {e}", "code": ""})
            return query, all_logs

    def compressed_search(self, query: str):
        """
        Performs compressed search: 
        Base Retrieval -> LLM Filtering -> Compressed Chunks.
        """
        all_logs = [{"message": "🗜️ **Initiating Contextual Compression**", "code": ""}]
        try:
            retriever = self.vector_store.get_compressed_retriever(self.ai.llm)
            all_logs.append({
                "message": "🔬 **LLM Filter Active:** Extracting only relevant snippets from chunks.",
                "code": "compressor = LLMChainExtractor.from_llm(llm)\\ncompressed_retriever = ContextualCompressionRetriever(...)"
            })
            results = retriever.invoke(query)
            return results, all_logs
        except Exception as e:
            all_logs.append({"message": f"❌ **Compression failed:** {e}", "code": ""})
            return [], all_logs

    def route_query(self, query: str):
        """
        Intelligently routes the query.
        """
        all_logs = [{"message": "🚦 **Initiating Semantic Routing**", "code": ""}]
        try:
            category = self.ai.classify_query(query)
            all_logs.append({
                "message": f"🧭 **Route Selected: `{category}`**",
                "code": f"category = llm_classifier.invoke('{query}')"
            })
            return category, all_logs
        except Exception as e:
            all_logs.append({"message": f"❌ **Routing failed:** {e}", "code": ""})
            return "DATABASE_SEARCH", all_logs

    def cached_query(self, query: str):
        """
        Demonstrates semantic caching.
        """
        return self.ai.generate_summary("N/A", query)

    def _handle_chunks(self, documents, mode: str = "recursive"):
        # Split into Chunks with instrumentation
        return self.processor.split_documents(documents, mode=mode)

def main():
    # Example usage
    agent = LangChainAgent(chunk_size=1000, chunk_overlap=100)
    
    # Create a dummy text file for demonstration if it doesn't exist
    demo_file = "demo_document.txt"
    if not os.path.exists(demo_file):
        with open(demo_file, "w") as f:
            f.write("LangChain Project Demo\n" + "="*20 + "\n")
            f.write("LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).\n")
            f.write("As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.\n")
            f.write("\n" + "History\n" + "-"*10 + "\n")
            f.write("LangChain was launched in October 2022 as an open source project by Harrison Chase, while working at machine learning startup Robust Intelligence.\n")
            f.write("The project quickly garnered popularity, with hundreds of contributors on GitHub, trending discussions on Twitter, lively servers on Discord, many YouTube tutorials, and meetups in San Francisco and London.\n")
            f.write("In April 2023, LangChain had incorporated and the new startup raised over $20 million in funding at a valuation of at least $200 million from venture capital firm Sequoia Capital, weeks after announcing a $10 million seed investment from Benchmark.\n")
    
    agent.process_file(demo_file)

if __name__ == "__main__":
    main()
