# Chapter 10: Multi-Agent Workflows

In this chapter, we level up from a single agent to a **Collaborative Team**.

## 1. Why Multi-Agent?
Large, complex prompts often lead to the "Jack of all trades, master of none" problem.
- A single LLM trying to research, write, code, and test might lose focus.
- **Solution**: Break the work into specialized roles (agents).

## 2. The Hand-off Pattern
This is the simplest multi-agent architecture.
1. **Researcher Agent**: Takes the topic, performs search/retrieval, and writes facts to the shared **State**.
2. **Writer Agent**: Reads the facts from the state and transforms them into a summary.
3. The graph defines a rigid arrow (edge) from one to the other.

## 3. Shared State
The key to multi-agent collaboration is the **State**. 
- It acts as a "Team Blackboard".
- Every agent can read what the previous agents wrote and build upon it.

---

### Mastery Lab Interaction
In the **Chapter 10** tab, deploy a Researcher/Writer team. You will see the Researcher finish their work before handing the baton to the Writer.
