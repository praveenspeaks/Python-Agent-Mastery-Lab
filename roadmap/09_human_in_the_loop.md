# Chapter 09: Human-In-The-Loop

This chapter demonstrates how to build "Safe AI" by requiring human intervention for critical tasks.

## 1. Why Human-In-The-Loop (HITL)?
Even the best LLMs can "hallucinate" or make mistakes. HITL ensures that an agent doesn't send a broken email, delete a database, or spend a budget without permission.

## 2. Using `interrupt_before`
LangGraph makes this easy with a single configuration parameter:
- We tell the graph to "Wait" before a specific node.
- The graph saves its entire **State** (memory) to a checkpointer.
- The thread ID allows us to find that exact state later and "Resume".

## 3. Approval and Persistence
Because the state is persisted, the "Human" doesn't need to be in the same session.
- AI starts task -> State saved to DB -> Human receives notification.
- Human clicks 'Approve' 2 hours later -> AI resumes exactly where it left off.

---

### Mastery Lab Interaction
In the **Chapter 9** tab, try starting a "Sensitive Task". The graph will pause and show a warning. Click **Approve AI Action** to see it finish.
