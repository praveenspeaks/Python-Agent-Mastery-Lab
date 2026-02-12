# Chapter 07: Transition to Agency

This chapter marks the most important shift in the course: **From Linear to Loops.**

## 1. What is a "Chain"? (The Past)
Up until now, we've used Chains.
- `Prompt -> LLM -> Search -> Output`
- **Problem**: If the Search fails, the output is wrong. There is no "retry".

## 2. What is an "Agent"? (The Future)
An Agent is an entity that uses an LLM to **Reason** and **Act**.
- it has **Tools**: Google Search, Database, Calculator.
- it has a **Loop**: It looks at the result of its action. If it's not good enough, it tries again with a different tool.

## 3. When to use which?
- **Use Chains** for predictable, fast, and repetitive tasks (e.g., summarizing every incoming email).
- **Use Agents** for unpredictable, complex research (e.g., "Find the best flight and book it if it's under $500").

---

### Mastery Lab Interaction
Chapter 7 allows you to compare a simple Chain with an Agentic loop, demonstrating why agency is required for high-level AI mastery.
