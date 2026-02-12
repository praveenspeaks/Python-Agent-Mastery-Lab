# Chapter 11: Self-Correction (Reflection)

This chapter focuses on a critical agentic pattern: **The Reflective Loop**.

## 1. What is AI Reflection?
Reflection is the process where an AI reviews its own work. 
- In a common chain, the AI answers and the job is done.
- in a Reflective Graph, the AI acts as its own **Editor**.

## 2. The Multi-Role Loop
The graph consists of two main roles:
1. **The Drafter**: Writes the initial answer.
2. **The Critic**: Reviews the answer for errors, tone, and accuracy. 
- If the Critic finds a problem, it sends the "Bat-Signal" (a conditional edge) back to the Drafter with feedback.

## 3. Benefits of Self-Correction
- **Higher Quality**: The second draft is almost always better than the first.
- **Error Reduction**: The model catches its own hallucinations by "fact-checking" itself in the critique stage.
- **Autonomous Polishing**: The system doesn't stop until the critic says "READY".

---

### Mastery Lab Interaction
In the **Chapter 11** tab, start a Reflection loop. You will see the "Critic" provide feedback and the "Drafter" address it in a second iteration.
