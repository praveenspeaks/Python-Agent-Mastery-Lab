# Chapter 06: Multi-Modal RAG

In this chapter, we explore how to make our AI "see" beyond plain text.

## 1. Handling Tables
Most documents (PDFs, PPTs) have tables. If we treat a table as a flat string, the AI loses the relationship between columns and rows.
- **Solution**: We use models like `Unstructured` to parse tables into HTML or Markdown format, which the AI understands much better.

## 2. Handling Images
A picture of a diagram or a chart contains massive info.
- **Visual Embedding**: Using models like CLIP to turn images into vectors, just like text.
- **Multi-Modal LLMs**: Using models like GPT-4o or Claude 3.5 Sonnet to "describe" the image and add that description to our search index.

## 3. The Multi-Vector Retriever
Instead of storing one vector per chunk, we store:
1. A summary of the image (text vector).
2. The raw image (stored in a database).
When the search finds the summary, it pulls the actual image for the AI to "look" at.

---

### Mastery Lab Interaction
Chapter 6 in the Mastery Lab provides stubs for these complex workflows, explaining the architecture required for vision-capable RAG.
