# Chapter 05: Query Enhancement

This chapter explores how to make our AI smarter by "re-thinking" the user's question before searching.

## 1. Multi-Query (Expansion)
The user might ask: "How much did LangChain raise?"
Multi-query rephrases this into:
- "LangChain seed funding amount"
- "Capital raised by LangChain in 2023"
- "Harrison Chase startup valuation"

**Why?** This catches documents that might use different words but have the same meaning, significantly boosting search recall.

## 2. Decomposition (Planning)
For complex questions like: "Compare the 2023 and 2024 funding rounds of LangChain."
Decomposition breaks it into:
1. "Search for LangChain 2023 funding details."
2. "Search for LangChain 2024 funding details."
3. "Compare the results of both years."

**Why?** It transforms a big, scary prompt into a series of easy, manageable steps.

---

### Mastery Lab Interaction
In the **Chapter 5** tab, try entering a long, multi-part question and select "Decomposition" to see how the AI plans to solve it step-by-step.
