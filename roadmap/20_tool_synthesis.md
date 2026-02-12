# Chapter 20: Tool Synthesis (AI as a Developer)

This chapter explores the final frontier of Agency: **The Self-Extending Agent**.

## 1. The Bottleneck of Pre-Defined Tools
Most agents can only use tools the developer built for them (e.g., Google Search, Database query).
- If the developer forgot to build a "Calculator" tool, the AI is stuck.
- **Tool Synthesis** removes this bottleneck.

## 2. Dynamic Code Generation
In this pattern, we give the model access to a **Sandboxed Python Runtime**.
- When a problem arises:
    1. The AI identifies it doesn't have a tool for the job.
    2. The AI writes the Python code to solve the problem.
    3. The AI executes the code and uses the result.
    4. The AI can even "save" this tool for future use in its memory.

## 3. Security and Safety
Writing and executing code dynamically is high-risk. In production systems:
- Code must run in **Isolated Containers** (like Docker or gVisor).
- No access to the host file system or network (unless explicitly allowed).
- Timeouts and resource limits must be strictly enforced.

---

### Mastery Lab Interaction
In the **Chapter 20** tab, give the AI a problem that requires logic or calculation. You will see it generate a Python block, which it conceptually 'executes' to give you the final answer.
