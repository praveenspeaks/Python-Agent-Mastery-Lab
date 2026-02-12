# Chapter 17: Performance & Caching

In this chapter, we focus on the "Economics" and "Latency" of RAG using **Semantic Caching**.

## 1. The Cost of RAG
Every time a user asks a question:
1. We turn the question into a vector (API call).
2. We search the database (Computing cost).
3. We send a large prompt to the LLM (Expensive tokens).

If 1,000 users ask "What is RAG?", we pay those costs 1,000 times.

## 2. Semantic Caching
Unlike traditional caching (which looks for exact word matches), Semantic Caching looks for **meaningful** matches.
- **Query A**: "How do I use LangGraph?"
- **Query B**: "Tell me how to use LangGraph."
- Traditional cache would miss. Semantic cache identifies they are 99% similar and returns the cached answer instantly.

## 3. Implementation Logic
1. A new query arrives.
2. We search a small "Cache Index" for similar questions.
3. If a match is found with >0.95 similarity: Return saved answer.
4. Else: Proceed to full RAG and then save the new answer to the cache.

---

### Mastery Lab Interaction
In the **Chapter 17** tab, ask a question and click "Query AI". Then, ask the exact same question again. You will see the system transition from "Cache Miss" to a "Cache Hit!", returning the answer in milliseconds.
