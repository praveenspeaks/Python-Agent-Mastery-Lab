# Chapter 15: Semantic Routing

This chapter concludes Phase 4 with **Intelligent Navigation**.

## 1. Routing vs. Chaining
In a simple chain, every user message follows the exact same path.
- In a **Routed System**, the AI decides which path to take based on the message's meaning (semantics).

## 2. Decision logic
The first step of the system is a classification call.
- **Greeting?** -> Route to a simple chat component.
- **Fact-finding?** -> Route to a retrieval-augmented chain.
- **Complex Research?** -> Route to a deep-search agent.

## 3. Why it matters
- **Speed**: Simple greetings get answered in milliseconds because we skip the database search.
- **Cost**: We don't waste LLM tokens or vector store hits on queries that don't need them.
- **User Experience**: The system feels smarter because it "understands" the type of input it received.

---

### Mastery Lab Interaction
In the **Chapter 15** tab, try entering a simple "Hello" versus a complex "What is a vector embedding?". You will see how the system identifies your intent and chooses the appropriate track.
