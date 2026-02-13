# Chapter 25: Langfuse Architecture & Observability

Observability is the difference between a prototype and a production-grade AI system. In this chapter, we explore how to monitor, trace, and evaluate our agents using **Langfuse**.

## 🎯 Learning Objectives
- Understand the "Black Box" problem in LLM applications.
- Implement **Tracing** to see exactly how data flows through chains and agents.
- Monitor **Latency**, **Cost**, and **Token Usage** in real-time.
- Learn about **Evaluation** (Human-in-the-loop and LLM-as-a-judge).

## 🏗️ The Observability Stack
In a production agentic RAG system, you need to answer:
1. "Why did the agent give this answer?" (**Tracing**)
2. "Which search result was actually used?" (**Retrieval Evaluation**)
3. "How much did this 5-turn conversation cost?" (**Cost Tracking**)

## 🛡️ Implementation: The Callback Pattern
LangChain uses a **Callback System**. Every time a node runs, it emits an event. Langfuse listens to these events and builds a tree (trace) of the execution.

```python
from langfuse.langchain import CallbackHandler

# 1. Initialize the handler
handler = CallbackHandler(
    public_key="...",
    secret_key="...",
    host="https://cloud.langfuse.com"
)

# 2. Pass it to your agent or chain
agent_executor.invoke(
    {"input": "What is LangGraph?"},
    config={"callbacks": [handler]}
)
```

## 🚀 Key Production Metrics
- **P95 Latency**: How slow are the slowest 5% of your requests?
- **Prompt vs. Completion Tokens**: Are your prompts getting too bloated?
- **Success Rate**: How often does the agent reach the 'FINISH' node without errors?

---
> [!TIP]
> **Pro Tip**: Use Langfuse to version your prompts. If a new prompt version performs better, you can swap it without changing code.
