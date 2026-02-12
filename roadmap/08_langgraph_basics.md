# Chapter 08: LangGraph Basics

In this chapter, we leave behind linear "Chains" and embrace **Stateful Graphs**.

## 1. What is LangGraph?
LangGraph is a library for building stateful, multi-actor applications with LLMs. It uses a **Graph** structure where:
- **Nodes** are snippets of logic (Python functions).
- **Edges** define the flow between those nodes.

## 2. The Power of "State"
Unlike a chain, which passes data once and forgets, LangGraph maintains a **State** object.
- Every node can see the state.
- Every node can update the state.
- This allows for **Persistence** and **Cycles** (trying again if things go wrong).

## 3. Nodes & Edges
- **Entry Point**: Where the graph starts.
- **Conditional Edges**: An LLM-driven "Router" that decides which node to visit next based on the current state.
- **END Node**: A special signal that the workflow is finished.

---

### Mastery Lab Interaction
In the **Chapter 8** tab, enter a goal and click **Start Graph Loop**. You will see the graph jump between a "Reason" node and an "Act" node until it achieves the goal or hits a limit.
