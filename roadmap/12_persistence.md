# Chapter 12: Persistence & Memory

This chapter marks the completion of our LangGraph foundations. We focus on **Long-term Memory**.

## 1. What is Persistence?
In most AI apps, once you refresh the page, the AI "forgets" who you are. This is because the context is stored in temporary session memory.
**Persistence** solves this by saving the AI's state (its "Brain") to a database.

## 2. Checkpointers
LangGraph uses **Checkpointers** to save the state.
- **Short-term memory**: The current conversation history.
- **Long-term memory**: The state of the entire graph, including intermediate variables and routing decisions.

## 3. The Thread ID
The "Key" to the memory is the **Thread ID**. 
- It allows us to distinguish between different users or different conversations.
- By providing the same Thread ID, the AI can "Recall" everything from the previous session.

---

### Mastery Lab Interaction
In the **Chapter 12** tab, tell the AI a secret using a specific Thread ID. Refresh the page (or click again) using the *same* ID, and you'll see the AI still knows your secret!
