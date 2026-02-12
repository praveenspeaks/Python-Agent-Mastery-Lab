# Chapter 14: Contextual Compression

This chapter covers how to refine what we find. It's about **Efficiency** and **Focus**.

## 1. The Retrieval Noise Problem
When we search for documents, we retrieve "Chunks" (e.g., 500 words).
- Often, only 10% of that chunk is actually useful.
- The other 90% is "Noise" that confuses the AI and costs extra tokens.

## 2. Contextual Compression
This is a post-retrieval step. After finding 4 chunks, we send them to a "Compressor" (usually a smaller/cheaper LLM).
- **Extraction**: The compressor reads the chunk and the user's question.
- **Filtering**: It throws away all the sentences that don't help answer the question.
- **Result**: The final AI only sees the "Gold" information.

## 3. Benefits
- **Saves Money**: Fewer tokens sent to the main model (like GPT-4).
- **Better Accuracy**: The AI isn't distracted by irrelevant text in the context window.

---

### Mastery Lab Interaction
In the **Chapter 14** tab, run a search. You will see how the original, bulky document chunks are "squeezed" down into just the relevant facts.
