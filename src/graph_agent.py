from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.ai_model import AIModelHandler

class AgentState(TypedDict):
    """
    Represents the state of our LangGraph agent.
    
    In LangGraph, 'State' is a shared dictionary that all nodes (functions) can access and modify.
    It's like the 'memory' of the agent during a conversation.
    """
    # use `add_messages` to append new messages to history instead of overwriting
    # 'Annotated' tells LangGraph: "When a node returns 'messages', don't overwrite this list. Add to it."
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 'next_step' helps us decide where to go next (Conditional Edge)
    next_step: str
    iterations: int

class LangGraphMastery:
    """
    Educational implementation of LangGraph basics.
    """
    def __init__(self):
        self.ai = AIModelHandler()
        self.workflow = StateGraph(AgentState)
        self.checkpointer = MemorySaver()
        
        # Define Nodes (The steps of our workflow)
        # Think of a Node as a function that does one specific job.
        self.workflow.add_node("reason", self.reason_node)
        self.workflow.add_node("act", self.act_node)
        
        # Define Edges (The logic of how to move between nodes)
        self.workflow.set_entry_point("reason") # Start here
        
        # Conditional Edges allow the agent to make a choice.
        # "If 'should_continue' returns 'end', go to END. Else go to 'act'."
        self.workflow.add_conditional_edges(
            "reason",
            self.should_continue,
            {
                "continue": "act",
                "end": END
            }
        )
        self.workflow.add_edge("act", "reason")
        
        # Compile into a Runnable
        # This converts our graph definition into an executable Python object.
        # 'checkpointer' allows us to save the state (for resuming later).
        # 'interrupt_before' pauses the agent before a specific node (great for human-in-the-loop).
        self.app = self.workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["act"] # Stop BEFORE acting for human approval
        )

    def reason_node(self, state: AgentState):
        """Decides if the goal is met or more action is needed."""
        last_message = state['messages'][-1].content
        prompt = f"Review this progress and decide if we are done. Respond with 'FINISH' if done, else 'ACT'.\\n\\Progress: {last_message}"
        decision = self.ai.llm.invoke(prompt).content
        
        return {
            "next_step": "end" if "FINISH" in decision.upper() else "continue",
            "iterations": state.get("iterations", 0) + 1
        }

    def act_node(self, state: AgentState):
        """Simple action node for demonstration."""
        return {
            "messages": [AIMessage(content="I am performing a task... done.")],
        }

    def should_continue(self, state: AgentState):
        if state["next_step"] == "end" or state.get("iterations", 0) > 3:
            return "end"
        return "continue"

    def run(self, input_text: str, thread_id: str = "1"):
        """Runs the graph with educational tracking and thread isolation."""
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=input_text)],
            "iterations": 0
        }
        
        # Check if we are resuming
        state = self.app.get_state(config)
        if state.values:
            # Resume from where we left off (e.g., after approval)
            iterator = self.app.stream(None, config)
        else:
            # Start fresh
            iterator = self.app.stream(initial_state, config)

        logs = []
        for output in iterator:
            for key, value in output.items():
                logs.append({"node": key, "data": value})
        
        # Check for interrupt
        next_state = self.app.get_state(config)
        if next_state.next:
            logs.append({"node": "INTERRUPT", "data": {"message": "PAUSED: Waiting for Human Approval", "next": next_state.next}})
            
        return logs

class MultiAgentMastery:
    """
    Implements a simple Multi-Agent team: Researcher + Writer.
    """
    def __init__(self):
        self.ai = AIModelHandler()
        self.workflow = StateGraph(AgentState)
        
        # Nodes
        self.workflow.add_node("researcher", self.research_node)
        self.workflow.add_node("writer", self.write_node)
        
        # Edges
        self.workflow.set_entry_point("researcher")
        self.workflow.add_edge("researcher", "writer")
        self.workflow.add_edge("writer", END)
        
        self.app = self.workflow.compile()

    def research_node(self, state: AgentState):
        """Simulates a researcher gathering facts."""
        topic = state['messages'][0].content
        prompt = f"Provide 3 key facts about {topic}."
        facts = self.ai.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=f"RESEARCH_DATA: {facts}")]}

    def write_node(self, state: AgentState):
        """Simulates a writer turning facts into a summary."""
        research = state['messages'][-1].content
        prompt = f"Turn these facts into a professional summary:\\n{research}"
        summary = self.ai.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=f"FINAL_WRITING: {summary}")]}

    def run(self, topic: str):
        logs = []
        initial_state = {"messages": [HumanMessage(content=topic)], "iterations": 0}
        for output in self.app.stream(initial_state):
            for key, value in output.items():
                logs.append({"node": key, "data": value})
        return logs

class ReflectiveMastery:
    """
    Implements a Self-Correction loop: Drafter -> Critique -> Redraft.
    """
    def __init__(self):
        self.ai = AIModelHandler()
        self.workflow = StateGraph(AgentState)
        
        # Nodes
        self.workflow.add_node("drafter", self.draft_node)
        self.workflow.add_node("critique", self.critique_node)
        
        # Edges
        self.workflow.set_entry_point("drafter")
        self.workflow.add_edge("drafter", "critique")
        self.workflow.add_conditional_edges(
            "critique",
            self.should_redraft,
            {
                "redraft": "drafter",
                "finish": END
            }
        )
        
        self.app = self.workflow.compile()

    def draft_node(self, state: AgentState):
        """Generates a draft or refines it based on critique."""
        topic = state['messages'][0].content
        critique = state['messages'][-1].content if len(state['messages']) > 1 else "No critique yet."
        prompt = f"Write a summary about {topic}.\\n\\If there is critique, address it:\\nCritique: {critique}"
        content = self.ai.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=content)], "iterations": state.get("iterations", 0) + 1}

    def critique_node(self, state: AgentState):
        """Critiques the current draft."""
        draft = state['messages'][-1].content
        prompt = (
            f"Review this draft and provide critical feedback to improve it. "
            f"If it is perfect, respond ONLY with 'READY'.\\n\\Draft: {draft}"
        )
        critique = self.ai.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=critique)], "next_step": "finish" if "READY" in critique.upper() else "redraft"}

    def should_redraft(self, state: AgentState):
        if state["next_step"] == "finish" or state.get("iterations", 0) >= 2:
            return "finish"
        return "redraft"

    def run(self, topic: str):
        logs = []
        initial_state = {"messages": [HumanMessage(content=topic)], "iterations": 0}
        for output in self.app.stream(initial_state):
            for key, value in output.items():
                logs.append({"node": key, "data": value})
        return logs

class PersistentMastery:
    """
    Demonstrates long-term memory using Thread IDs.
    """
    def __init__(self):
        self.ai = AIModelHandler()
        self.workflow = StateGraph(AgentState)
        self.checkpointer = MemorySaver()
        
        self.workflow.add_node("remember", self.remember_node)
        self.workflow.set_entry_point("remember")
        self.workflow.add_edge("remember", END)
        
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def remember_node(self, state: AgentState):
        """Stores information in the persistent state."""
        new_msg = state['messages'][-1].content
        # In a real app, you might summarize all history here
        return {"messages": [AIMessage(content=f"MEMORY_UPDATED: {new_msg}")]}

    def get_history(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(config)
        return state.values.get("messages", []) if state.values else []

    def run(self, input_text: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        self.app.invoke({"messages": [HumanMessage(content=input_text)]}, config)
        return self.get_history(thread_id)

class ReliableRAG:
    """
    Implements a self-grading RAG pipeline: Retrieve -> Grade -> Generate/Retry.
    """
    def __init__(self):
        self.ai = AIModelHandler()
        self.workflow = StateGraph(AgentState)
        
        # Nodes
        self.workflow.add_node("retrieve", self.retrieve_node)
        self.workflow.add_node("grade", self.grade_node)
        self.workflow.add_node("generate", self.generate_node)
        
        # Edges
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade")
        self.workflow.add_conditional_edges(
            "grade",
            self.decide_to_generate,
            {
                "yes": "generate",
                "no": END # In a real app, this would go to Web Search
            }
        )
        self.workflow.add_edge("generate", END)
        self.app = self.workflow.compile()

    def retrieve_node(self, state: AgentState):
        query = state['messages'][0].content
        # Simulated retrieval
        docs = ["Document about RAG basics", "Irrelevant document about cooking"]
        return {"messages": [AIMessage(content=f"RETRIEVED: {docs}")]}

    def grade_node(self, state: AgentState):
        query = state['messages'][0].content
        retrieval = state['messages'][-1].content
        prompt = (
            f"Given this query: {query}\\n"
            f"And these documents: {retrieval}\\n"
            "Grade the documents as 'RELEVANT' or 'IRRELEVANT' based on the query.\\n"
            "Respond with ONLY the grade."
        )
        grade = self.ai.llm.invoke(prompt).content.upper()
        return {"next_step": "yes" if "RELEVANT" in grade else "no"}

    def generate_node(self, state: AgentState):
        query = state['messages'][0].content
        retrieval = state['messages'][-2].content
        prompt = f"Answer the question: {query} using these documents: {retrieval}"
        answer = self.ai.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=answer)]}

    def decide_to_generate(self, state: AgentState):
        return state["next_step"]

    def run(self, topic: str):
        logs = []
        initial_state = {"messages": [HumanMessage(content=topic)], "iterations": 0}
        for output in self.app.stream(initial_state):
            for key, value in output.items():
                logs.append({"node": key, "data": value})
        return logs

class PlanAndExecuteMastery:
    """
    Implements a Plan-and-Execute pattern.
    """
    def __init__(self):
        self.ai = AIModelHandler()
        self.workflow = StateGraph(AgentState)
        
        # Nodes
        self.workflow.add_node("planner", self.planner_node)
        self.workflow.add_node("executor", self.executor_node)
        
        # Edges
        self.workflow.set_entry_point("planner")
        self.workflow.add_edge("planner", "executor")
        self.workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {
                "continue": "executor",
                "replan": "planner",
                "finish": END
            }
        )
        self.app = self.workflow.compile()

    def planner_node(self, state: AgentState):
        topic = state['messages'][0].content
        prompt = f"Create a 3-step plan to explain {topic}. Return ONLY the steps as a numbered list."
        plan = self.ai.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=f"PLAN: {plan}")]}

    def executor_node(self, state: AgentState):
        # In a real app, this would execute one step of the plan
        # Here we simulate finishing one step
        steps_left = state.get("iterations", 3) - 1
        return {"messages": [AIMessage(content=f"EXECUTED_STEP. {steps_left} steps remaining.")], "iterations": steps_left}

    def should_continue(self, state: AgentState):
        if state.get("iterations", 3) <= 0:
            return "finish"
        return "continue"

    def run(self, topic: str):
        logs = []
        # We use iterations to track remaining steps in the plan
        initial_state = {"messages": [HumanMessage(content=topic)], "iterations": 3}
        for output in self.app.stream(initial_state):
            for key, value in output.items():
                logs.append({"node": key, "data": value})
        return logs

class ToolSynthesizer:
    """
    Implements a Tool Synthesis pattern (AI writing its own tools).
    """
    def __init__(self):
        self.ai = AIModelHandler()
        self.workflow = StateGraph(AgentState)
        
        # Nodes
        self.workflow.add_node("writer", self.writer_node)
        self.workflow.add_node("executor", self.executor_node)
        
        # Edges
        self.workflow.set_entry_point("writer")
        self.workflow.add_edge("writer", "executor")
        self.workflow.add_edge("executor", END)
        
        self.app = self.workflow.compile()

    def writer_node(self, state: AgentState):
        problem = state['messages'][0].content
        prompt = (
            f"Write a Python function to solve this: {problem}\\n"
            "Return ONLY the code within triple backticks."
        )
        code = self.ai.llm.invoke(prompt).content
        return {"messages": [AIMessage(content=f"CODE_GENERATED: {code}")]}

    def executor_node(self, state: AgentState):
        code = state['messages'][-1].content
        # In a real app, this would use PythonREPL or a sandboxed exec()
        # Here we simulate the successful execution
        return {"messages": [AIMessage(content="SUCCESS. Tool synthesized and executed successfully.")]}

    def run(self, problem: str):
        logs = []
        initial_state = {"messages": [HumanMessage(content=problem)]}
        for output in self.app.stream(initial_state):
            for key, value in output.items():
                logs.append({"node": key, "data": value})
        return logs
