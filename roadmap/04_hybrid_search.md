# Chapter 04: Hybrid Search

This chapter focuses on the "Ensemble" strategy—combining different search methods to overcome their individual weaknesses.

## 1. The Vector Weakness (Semantic Search)
Vector search is amazing at finding *meaning*, but it can struggle with:
- **Exact Names**: If you search for "Company X-Y-Z", vector search might find "Business A-B-C" because they are both tech startups, even if you wanted the exact name.
- **Part Numbers / Acronyms**: Rare codes often don't have strong semantic associations.

## 2. The Keyword Strength (BM25)
BM25 (Best Matching 25) is the classic search engine algorithm (like Lucene/ElasticSearch). It looks for exact word matches.
- **Great for**: Specialized jargon, names, and specific identifier codes.

## 3. The Hybrid solution (Ensemble)
We use an **EnsembleRetriever** to run both searches.
1.  **Run Vector Search** -> Get top 4 results.
2.  **Run BM25 Search** -> Get top 4 results.
3.  **Merge Results**: Use **Reciprocal Rank Fusion (RRF)** to score them. If a document appeared at the top of *both* lists, it gets a massive score boost!

---

### Mastery Lab Interaction
In the **Chapter 4** tab, you can adjust the "Weights" for Vector vs. Keyword.
- Give Keyword **1.0** weight if you are looking for a specific name.
- Give Vector **1.0** weight if you are asking a broad conceptual question.
- Use **0.5 / 0.5** for the best all-around performance.
