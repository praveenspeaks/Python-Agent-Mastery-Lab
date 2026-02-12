# Chapter 03: Advanced Chunking

This chapter explores moving beyond simple character count splitting to **Intelligent Segmenting**.

## 1. Semantic Chunking
Instead of cutting text at a fixed number (e.g., 500 characters), semantic chunking uses AI embeddings to analyze the meaning of sentences.

### How it works:
1.  **Sentence Splitting**: The text is broken into individual sentences.
2.  **Embedding**: Each sentence is converted into a vector.
3.  **Breakpoint Detection**: The system calculates the 'distance' (difference in meaning) between consecutive sentences.
4.  **Grouping**: If the difference is below a threshold, the sentences stay together. If it jumps, a new chunk is started.

## 2. Contextual Chunking (The 'Missing' Piece)
When we split a document, small chunks often lose their context.
- **Problem**: A chunk says "He was born in 1955." (Who is 'He'?)
- **Solution**: We use an LLM to add a small prefix to every chunk (e.g., "This text is about Steve Jobs: He was born in 1955.")

## Why use Advanced Chunking?
- **Better Search**: Finding relevant info is easier if the chunks represent complete thoughts.
- **Higher LLM Accuracy**: The AI gets cleaner, more relevant context to answer your questions.

---

### Mastery Lab Interaction
In the **Chapter 3** tab of the Mastery Lab, you can compare **Recursive** vs. **Semantic** splitting side-by-side to see how topics are better preserved.
