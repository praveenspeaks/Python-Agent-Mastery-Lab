# LangChain & LangGraph Mastery Roadmap

This document outlines our step-by-step journey to building advanced Agentic AI systems. Every chapter will have its own code updates and architectural documentation.

## 🗺️ Learning Path

### Phase 1: Foundations
- [x] **01. Parsing Documents**: Multi-format loading (Updated: Robust PDF with PyMuPDF & OCR Fallback)
- [x] **02. Vector Embedding & Stores**: Turning text into numbers and storing them (FAISS, ChromaDB) [Documentation](./roadmap/02_embeddings_and_vector_stores.md)
- [x] **03. Advanced Chunking**: Semantic, Agentic, and Contextual splitting [Documentation](./roadmap/03_advanced_chunking.md)
- [x] **04. Hybrid Search**: Combining Keyword (BM25) with Vector Search [Documentation](./roadmap/04_hybrid_search.md)

### Phase 2: Retrieval Engineering
- [x] **05. Query Enhancement**: Re-writing and Expansion (Multi-query, De-composition) [Documentation](./roadmap/05_query_enhancement.md)
- [x] **06. Multi-Modal RAG**: Handling Images and Tables [Documentation](./roadmap/06_multi_modal.md)
- [x] **07. Transition to Agency**: When to move from chains to agents [Documentation](./roadmap/07_transition.md)

### Phase 3: LangGraph & Agentic Workflows
- [x] **08. LangGraph Basics**: Nodes, Edges, and State management [Documentation](./roadmap/08_langgraph_basics.md)
- [x] **09. Agentic Architecture**: Tool calling and Reasoning loops (ReAct) [Documentation](./roadmap/09_human_in_the_loop.md)
- [x] **10. Multi-Agent Workflows**: Designing collaborative agent teams [Documentation](./roadmap/10_multi_agent.md)
- [x] **11. Self-Correction (CRAG)**: Agents that reflect and correct their own errors [Documentation](./roadmap/11_self_correction.md)
- [x] **12. Persistence & Memory**: Advanced state management and long-term memory [Documentation](./roadmap/12_persistence.md)

### Phase 4: Advanced RAG Patterns
- [x] **13. Query Transformations (HyDE)**: Boosting search with hypothetical answers [Documentation](./roadmap/13_hyde_patterns.md)
- [x] **14. Contextual Compression**: LLM-based de-noising of retrieved chunks [Documentation](./roadmap/14_contextual_compression.md)
- [x] **15. Semantic Routing**: Intent-based branching [Documentation](./roadmap/15_semantic_routing.md)

### Phase 5: Production & Evaluation
- [x] **16. Reliable RAG**: Graph-based self-grading retrieval [Documentation](./roadmap/16_reliable_rag.md)
- [x] **17. Performance & Caching**: Optimizing with Semantic Caching [Documentation](./roadmap/17_performance_caching.md)
- [x] **18. RAG Evaluation**: Benchmarking with RAGAS [Documentation](./roadmap/18_rag_evaluation.md)

### Phase 6: DeepAgents Mastery (The Ultimate Agent)
- [x] **19. Autonomous Planning**: Hierarchical Planning and Task Decomposition [Documentation](./roadmap/19_autonomous_planning.md)
- [x] **20. Tool Synthesis**: Agents that write and debug their own tools/MCP Servers [Documentation](./roadmap/20_tool_synthesis.md)
- [x] **21. Multi-Agent Orchestration**: Boss/Worker patterns and Conflict Resolution [Completed]
- [x] **22. Long-term Context Management**: Vectorized memory vs. Episodic memory [Completed]
- [ ] **Chapter 20: Tool Synthesis & Execution** [20_tools.md](file:///d:/My%20Professional%20Projects/PythonAgent/roadmap/20_tools.md)
- [x] **Chapter 24: Final Graduation** [24_graduation.md](file:///d:/My%20Professional%20Projects/PythonAgent/roadmap/24_graduation.md)

### Phase 7: Observability & Production Architecture (Bonus)
- [ ] **Chapter 25: Langfuse Tracing & Observability** [25_langfuse.md](file:///d:/My%20Professional%20Projects/PythonAgent/roadmap/25_langfuse.md)

---

## 📂 Chapter Documentation
Each chapter's architecture is stored in the `/roadmap` folder:
1. [Chapter 01: Parsing Documents](./roadmap/01_parsing_documents.md)
2. [Chapter 02: Vector Embeddings & Stores](./roadmap/02_embeddings_and_vector_stores.md)
