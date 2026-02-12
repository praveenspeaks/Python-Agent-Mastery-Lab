# Chapter 19: Autonomous Planning

This chapter marks our entry into **Level 5 Autonomy: DeepAgents**.

## 1. Chains vs. Plans
- In a **Chain**, the developer defines every step. The AI has no choice.
- In a **Plan**, the developer defines the goal. The AI creates the steps.

## 2. The Plan-and-Execute Pattern
We use a specialized LangGraph architecture with two core loops:
1. **The Planner**: Receives the goal and outputs a numbered list of sub-tasks.
2. **The Executor**: Takes one sub-task at a time and performs it.
3. **The Re-planner**: After each step, it reviews the progress and updates the plan if needed.

## 3. Why it matters
- **Complexity**: Fixed chains break when a task is too large. Planning decomposes it into manageable bits.
- **Resilience**: If Step 1 fails, the Re-planner can pivot to a different Step 1, rather than the whole system crashing.
- **Traceability**: You can see exactly what the AI *intended* to do before it did it.

---

### Mastery Lab Interaction
In the **Chapter 19** tab, enter a high-level goal. You will see the Planner generate the steps first, followed by the Executor hitting each node until the goal is achieved.
