# Chapter 18: RAG Evaluation (RAGAS)

This chapter concludes Phase 5 by answering the question: *"Is it actually good?"*

## 1. The Need for Metrics
Software engineers have unit tests. AI engineers have **Evaluations**.
If you change your chunking strategy, how do you know if the answers got better or worse? You can't read 1,000 answers manually every time.

## 2. The RAGAS Framework
RAGAS uses a "Judge" (a powerful LLM like GPT-4) to grade four specific metrics:
- **Faithfulness**: Is every claim in the answer supported by the context? (Catches Hallucination).
- **Answer Relevance**: Does the answer directly address the user's question? (Catches Divergence).
- **Context Precision**: Out of all the documents retrieved, how many were actually useful? (Catches Retrieval Noise).
- **Context Recall**: Did the retrieval find everything needed to answer the question? (Catches Retrieval Gaps).

## 3. The Continuous Improvement Loop
1. **Assess**: Run RAGAS on a "Golden Dataset" (a set of known good questions/answers).
2. **Analyze**: Identify which metric is low (e.g., if Recall is low, increase Chunk Size).
3. **Iterate**: Update the code and re-assess.

---

### Mastery Lab Interaction
In the **Chapter 18** tab, enter a question, a context snippet, and an AI answer. Click "Calculate RAGAS Scores" to see a simulated grade of how "Faithful" and "Relevant" that interaction was.
