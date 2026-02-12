"""
LangGraph Curriculum - Stateful agent workflows

6 Chapters covering:
1. LangGraph Basics (Nodes, Edges, State)
2. State Management
3. Conditional Edges
4. Multi-Agent Systems
5. Human-in-the-Loop
6. Persistence & Memory
"""

from typing import List, Dict, Any


LANGGRAPH_CURRICULUM: List[Dict[str, Any]] = [
    
    # ============================================================================
    # CHAPTER 1: LANGGRAPH BASICS
    # ============================================================================
    {
        "title": "LangGraph Basics",
        "subtitle": "Nodes, Edges, and State",
        "emoji": "🕸️",
        "objectives": [
            "Understand why LangGraph exists",
            "Learn the core concepts: Nodes, Edges, State",
            "Build your first graph",
            "Understand cyclic vs acyclic flows"
        ],
        "content": """
        ## What is LangGraph?
        
        **LangGraph** is a library for building stateful, multi-actor applications with LLMs.
        
        ### Why LangGraph?
        
        Regular chains are **linear**:
        ```
        A → B → C → D
        ```
        
        Real applications need **cycles**:
        ```
        A → B → C → (check condition)
                ↑_________|
        ```
        
        LangGraph lets you build graphs with loops for:
        - Agent reasoning cycles
        - Multi-step approval workflows
        - Human-in-the-loop interactions
        - Multi-agent collaboration
        
        ### Core Concepts
        
        **1. State**
        A shared data structure that all nodes can read and write.
        ```python
        class State(TypedDict):
            messages: list
            counter: int
        ```
        
        **2. Nodes**
        Python functions that perform work. They receive state and return updates.
        ```python
        def my_node(state: State):
            return {"counter": state["counter"] + 1}
        ```
        
        **3. Edges**
        Connections between nodes. Define the flow of execution.
        ```python
        graph.add_edge("node_a", "node_b")
        ```
        
        ### LangGraph vs LangChain
        
        | Feature | LangChain | LangGraph |
        |---------|-----------|-----------|
        | Flow | Linear chains | Cyclic graphs |
        | State | Implicit | Explicit |
        | Persistence | Limited | Built-in |
        | Human-in-loop | Hard | Easy |
        | Multi-agent | Challenging | Native |
        
        ### Your First Graph
        
        ```python
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        # 1. Define state
        class State(TypedDict):
            message: str

        # 2. Define nodes
        def node_a(state: State):
            return {"message": state["message"] + " from A"}

        def node_b(state: State):
            return {"message": state["message"] + " from B"}

        # 3. Build graph
        builder = StateGraph(State)
        builder.add_node("node_a", node_a)
        builder.add_node("node_b", node_b)
        builder.add_edge("node_a", "node_b")
        builder.add_edge("node_b", END)
        builder.set_entry_point("node_a")

        # 4. Compile and run
        graph = builder.compile()
        result = graph.invoke({"message": "Hello"})
        # Result: {"message": "Hello from A from B"}
        ```
        """,
        "code_examples": [
            {
                "title": "🕸️ Building Your First LangGraph",
                "code": '''from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# 1. Define the state schema
class AgentState(TypedDict):
    """State that persists across node executions."""
    messages: Annotated[list, operator.add]  # Append behavior
    steps: int
    should_continue: bool

# 2. Define nodes (these are just Python functions!)
def start_node(state: AgentState):
    """First node - initializes the workflow."""
    print(f"Starting with {len(state['messages'])} messages")
    return {
        "messages": [{"role": "assistant", "content": "Hello!"}],
        "steps": 1
    }

def process_node(state: AgentState):
    """Processing node - does some work."""
    print(f"Processing step {state['steps']}")
    return {
        "messages": [{"role": "assistant", "content": f"Step {state['steps']} complete"}],
        "steps": state["steps"] + 1
    }

def should_continue(state: AgentState):
    """Conditional logic node."""
    return "continue" if state["steps"] < 3 else "stop"

# 3. Build the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("start", start_node)
builder.add_node("process", process_node)

# Add edges
builder.set_entry_point("start")
builder.add_edge("start", "process")

# Conditional edge: can go to process again or end
builder.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Loop back to process
        "stop": END             # End the graph
    }
)

# 4. Compile
graph = builder.compile()

# 5. Execute
result = graph.invoke({
    "messages": [],
    "steps": 0
})

print(f"Final state: {result}")''',
                "explanation": """
                **LangGraph** builds workflows as graphs with explicit state management.
                
                **Key Concepts:**
                - **TypedDict State**: Defines the shape of data passed between nodes
                - **Annotated**: Controls how state updates merge (here, operator.add appends to lists)
                - **Nodes**: Regular Python functions that receive state and return partial state updates
                - **Edges**: Define flow. Regular edges are direct. Conditional edges branch based on logic.
                
                **The Flow:**
                1. start → process (first time)
                2. should_continue checks steps
                3. If steps < 3: loops back to process
                4. If steps >= 3: goes to END
                """,
                "key_concepts": ["StateGraph", "TypedDict", "Annotated", "Nodes", "Conditional Edges", "END"],
                "line_explanations": {
                    "7": "TypedDict defines the state schema - what data flows through the graph",
                    "8": "Annotated with operator.add means new messages get appended, not replaced",
                    "13": "Nodes are just functions that take state dict and return updates dict",
                    "34": "StateGraph is the builder class for constructing graphs",
                    "37": "add_node registers a function as a named node in the graph",
                    "46": "add_conditional_edges routes to different nodes based on function return value",
                    "49": "END is a special node that terminates graph execution"
                },
                "references": {
                    "StateGraph": {
                        "type": "Class",
                        "description": "Builder for creating stateful graphs. Define nodes, edges, and state schema.",
                        "example": "builder = StateGraph(StateSchema)"
                    },
                    "TypedDict": {
                        "type": "Type",
                        "description": "Python type that defines dictionary structure with specific keys and types",
                        "example": "class State(TypedDict):\\n    messages: list"
                    },
                    "Annotated": {
                        "type": "Type Modifier",
                        "description": "Adds metadata to type hints. In LangGraph, controls how state updates merge.",
                        "example": "messages: Annotated[list, operator.add]  # Append, don't replace"
                    },
                    "add_conditional_edges": {
                        "type": "Method",
                        "description": "Adds branching logic. Function return value determines which path to take.",
                        "example": "builder.add_conditional_edges('node', routing_func, {'a': 'node_a', 'b': 'node_b'})"
                    }
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "What is a Node in LangGraph?",
                "options": [
                    "A visual diagram element",
                    "A Python function that processes state",
                    "A database table",
                    "A type of LLM"
                ],
                "correct": "A Python function that processes state",
                "explanation": "Nodes are regular Python functions that receive the current state and return state updates."
            },
            {
                "question": "What does END represent?",
                "options": [
                    "The last node added",
                    "A special node that terminates execution",
                    "An error state",
                    "The starting point"
                ],
                "correct": "A special node that terminates execution",
                "explanation": "END is a sentinel value that tells LangGraph to stop executing the graph."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 2: STATE MANAGEMENT
    # ============================================================================
    {
        "title": "State Management",
        "subtitle": "Reducers and persistence patterns",
        "emoji": "💾",
        "objectives": [
            "Understand state reducers",
            "Use Annotated types effectively",
            "Manage complex state objects",
            "Handle list vs replace semantics"
        ],
        "content": """
        ## Understanding State in LangGraph
        
        In LangGraph, **State** is the single source of truth that flows through your graph.
        
        ### The State Problem
        
        Multiple nodes might update the same key:
        ```python
        # Node A returns
        {"messages": [msg1]}
        
        # Node B returns  
        {"messages": [msg2]}
        
        # What should the final state be?
        # Option 1: Replace → {"messages": [msg2]}
        # Option 2: Append → {"messages": [msg1, msg2]}
        ```
        
        ### Reducers to the Rescue
        
        **Reducers** define HOW state updates should merge.
        
        ```python
        from typing import Annotated
        import operator

        # Replace (default): New value overwrites old
        name: str

        # Append: New items added to list
        messages: Annotated[list, operator.add]

        # Custom reducer
        def merge_dicts(existing, new):
            return {**existing, **new}
        
        config: Annotated[dict, merge_dicts]
        ```
        
        ### Common Reducers
        
        | Reducer | Behavior | Use Case |
        |---------|----------|----------|
        | (none) | Replace | Simple values, counters |
        | `operator.add` | Append | Lists of messages |
        | Custom function | Custom logic | Merging nested objects |
        
        ### State Design Patterns
        
        **Pattern 1: Message History**
        ```python
        class State(TypedDict):
            messages: Annotated[list, operator.add]  # Keeps all messages
        ```
        
        **Pattern 2: Working Memory**
        ```python
        class State(TypedDict):
            scratchpad: str  # Overwritten each step
            final_answer: str  # Set once at end
        ```
        
        **Pattern 3: Multi-Agent State**
        ```python
        class State(TypedDict):
            # Shared state
            messages: Annotated[list, operator.add]
            
            # Agent-specific state (using namespaces)
            researcher_notes: str
            writer_draft: str
        ```
        
        ### State Persistence
        
        LangGraph can automatically save and resume state:
        
        ```python
        from langgraph.checkpoint.sqlite import SqliteSaver

        # Add persistence
        checkpointer = SqliteSaver.from_conn_string(":memory:")
        graph = builder.compile(checkpointer=checkpointer)

        # Each run gets a thread_id
        config = {"configurable": {"thread_id": "conversation_1"}}
        
        # State is automatically saved
        result = graph.invoke({"messages": []}, config)
        
        # Resume from where you left off
        result = graph.invoke({"messages": [new_message]}, config)
        ```
        """,
        "code_examples": [
            {
                "title": "💾 State Reducers in Practice",
                "code": '''from typing import TypedDict, Annotated
import operator

# Custom reducer: keep the maximum value
def keep_max(existing, new):
    return max(existing, new)

# Custom reducer: merge dictionaries
def merge_metadata(existing: dict, new: dict):
    merged = existing.copy()
    merged.update(new)
    return merged

class WorkflowState(TypedDict):
    # Default: Replace - new value overwrites old
    current_step: str
    
    # operator.add: Append to list
    messages: Annotated[list, operator.add]
    
    # Custom: Keep maximum score
    best_score: Annotated[float, keep_max]
    
    # Custom: Merge dictionaries
    metadata: Annotated[dict, merge_metadata]

# Example node showing different update patterns
def example_node(state: WorkflowState):
    return {
        # Overwrites current_step
        "current_step": "processing",
        
        # Appends to messages list
        "messages": [{"role": "assistant", "content": "Hello"}],
        
        # Keeps max of existing vs new
        "best_score": 95.5,
        
        # Merges into existing metadata dict
        "metadata": {"processed_at": "2024-01-01"}
    }

# Demonstration
from langgraph.graph import StateGraph, END

builder = StateGraph(WorkflowState)
builder.add_node("example", example_node)
builder.set_entry_point("example")
builder.add_edge("example", END)

graph = builder.compile()

# First run
result1 = graph.invoke({
    "current_step": "start",
    "messages": [{"role": "user", "content": "Hi"}],
    "best_score": 80.0,
    "metadata": {"created_by": "user"}
})

print("After first run:")
print(f"  current_step: {result1['current_step']}")  # "processing" (replaced)
print(f"  messages: {len(result1['messages'])} msgs")  # 2 (appended)
print(f"  best_score: {result1['best_score']}")  # 95.5 (max kept)
print(f"  metadata: {result1['metadata']}")  # Has both created_by and processed_at''',
                "explanation": """
                **Reducers** control how state updates from different nodes merge together.
                
                **Default Behavior (No Reducer):**
                - New value completely replaces the old value
                - Good for: status flags, single values, current state
                
                **operator.add Reducer:**
                - Appends new items to existing list
                - Good for: message history, event logs
                
                **Custom Reducers:**
                - Define your own merge logic
                - Good for: complex objects, conflict resolution, aggregations
                
                **Important:** All nodes that run in parallel see the SAME state snapshot.
                Their updates are merged AFTER all parallel nodes complete.
                """,
                "key_concepts": ["Reducers", "Annotated", "operator.add", "State Merging", "TypedDict"],
                "line_explanations": {
                    "4": "Custom reducers are just functions: (existing_value, new_value) → merged_value",
                    "15": "Default behavior (no Annotated): New value replaces old",
                    "18": "Annotated with operator.add: New items appended to list",
                    "21": "Custom reducer function: Implements keep_max logic",
                    "24": "Another custom reducer: Merges dictionaries instead of replacing",
                    "46": "Node returns partial state - only includes keys being updated"
                },
                "references": {
                    "operator.add": {
                        "type": "Reducer",
                        "description": "Built-in Python operator. For lists, it concatenates (extends) them.",
                        "example": "messages: Annotated[list, operator.add]  # Appends new messages"
                    },
                    "Annotated": {
                        "type": "Type Hint",
                        "description": "Attaches metadata to types. In LangGraph, the second argument is the reducer function.",
                        "example": "field: Annotated[type, reducer_function]"
                    },
                    "keep_max": {
                        "type": "Custom Reducer",
                        "description": "Example custom reducer that keeps the larger of two values",
                        "example": "best_score: Annotated[float, keep_max]"
                    }
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "What happens by default (no reducer) when two nodes update the same key?",
                "options": [
                    "Values are appended",
                    "Last write wins (replacement)",
                    "Error is raised",
                    "Values are averaged"
                ],
                "correct": "Last write wins (replacement)",
                "explanation": "Without a reducer, the last update to a key overwrites any previous updates."
            },
            {
                "question": "What does `Annotated[list, operator.add]` do?",
                "options": [
                    "Adds numbers in the list",
                    "Appends new items to the existing list",
                    "Sorts the list",
                    "Removes duplicates"
                ],
                "correct": "Appends new items to the existing list",
                "explanation": "operator.add as a reducer concatenates lists, effectively appending new messages."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 3: CONDITIONAL EDGES
    # ============================================================================
    {
        "title": "Conditional Edges",
        "subtitle": "Routing and branching logic",
        "emoji": "🔀",
        "objectives": [
            "Create branching workflows",
            "Implement loops and cycles",
            "Build decision trees",
            "Handle error routing"
        ],
        "content": """
        ## Conditional Edges in LangGraph
        
        **Conditional edges** let your graph make decisions about where to go next.
        
        ### Types of Edges
        
        ```python
        # Normal edge: Always goes from A to B
        builder.add_edge("A", "B")
        
        # Conditional edge: Goes to different nodes based on a condition
        builder.add_conditional_edges("A", routing_function, {
            "path_1": "node_b",
            "path_2": "node_c"
        })
        
        # Entry point: Where the graph starts
        builder.set_entry_point("start_node")
        ```
        
        ### Building a Decision Tree
        
        ```python
        def route_by_intent(state: State):
            intent = classify_intent(state["message"])
            if intent == "question":
                return "answer"
            elif intent == "order":
                return "process_order"
            else:
                return "clarify"
        
        builder.add_conditional_edges(
            "classify",
            route_by_intent,
            {
                "answer": "answer_node",
                "process_order": "order_node",
                "clarify": "clarify_node"
            }
        )
        ```
        
        ### Loops and Cycles
        
        LangGraph shines when you need to loop:
        
        ```python
        def should_retry(state: State):
            if state["attempts"] < 3 and not state["success"]:
                return "retry"
            return "finish"
        
        builder.add_conditional_edges(
            "process",
            should_retry,
            {
                "retry": "process",  # Loop back!
                "finish": END
            }
        )
        ```
        
        ### Common Routing Patterns
        
        **Pattern 1: Intent Classification**
        ```
        classify_intent → route → [answer, search, escalate]
        ```
        
        **Pattern 2: Quality Check**
        ```
        generate → evaluate → [accept, regenerate, human_review]
        ```
        
        **Pattern 3: Error Handling**
        ```
        try → [success → continue, error → retry → (max 3) → fallback]
        ```
        """,
        "code_examples": [
            {
                "title": "🔀 Building a Router Graph",
                "code": '''from langgraph.graph import StateGraph, END
from typing import TypedDict
import random

class RouterState(TypedDict):
    query: str
    query_type: str
    response: str
    attempts: int

# Node: Classify the query
def classify_node(state: RouterState):
    """Determine what type of query this is."""
    query = state["query"].lower()
    
    if any(word in query for word in ["weather", "temperature", "rain"]):
        query_type = "weather"
    elif any(word in query for word in ["price", "cost", "buy"]):
        query_type = "shopping"
    elif any(word in query for word in ["help", "support", "problem"]):
        query_type = "support"
    else:
        query_type = "general"
    
    return {"query_type": query_type}

# Node: Handle weather queries
def weather_node(state: RouterState):
    return {"response": f"Weather info for: {state['query']}"}

# Node: Handle shopping queries
def shopping_node(state: RouterState):
    return {"response": f"Shopping results for: {state['query']}"}

# Node: Handle support queries
def support_node(state: RouterState):
    return {"response": f"Support ticket created for: {state['query']}"}

# Node: Handle general queries
def general_node(state: RouterState):
    return {"response": f"General answer for: {state['query']}"}

# Router function
def route_query(state: RouterState):
    """Return the node name to route to based on query_type."""
    routing_map = {
        "weather": "weather",
        "shopping": "shopping",
        "support": "support",
        "general": "general"
    }
    return routing_map.get(state["query_type"], "general")

# Build graph
builder = StateGraph(RouterState)

# Add nodes
builder.add_node("classify", classify_node)
builder.add_node("weather", weather_node)
builder.add_node("shopping", shopping_node)
builder.add_node("support", support_node)
builder.add_node("general", general_node)

# Add edges
builder.set_entry_point("classify")

# The magic: conditional routing!
builder.add_conditional_edges(
    "classify",
    route_query,
    {
        "weather": "weather",
        "shopping": "shopping", 
        "support": "support",
        "general": "general"
    }
)

# All paths lead to END
for node in ["weather", "shopping", "support", "general"]:
    builder.add_edge(node, END)

# Compile and test
graph = builder.compile()

# Test different queries
test_queries = [
    "What's the weather today?",
    "How much does this cost?",
    "I need help with my account",
    "Tell me a joke"
]

for query in test_queries:
    result = graph.invoke({"query": query, "query_type": "", "response": "", "attempts": 0})
    print(f"Q: {query}")
    print(f"  Type: {result['query_type']}")
    print(f"  Response: {result['response']}\\n")''',
                "explanation": """
                **Conditional edges** enable routing decisions based on state.
                
                **The Pattern:**
                1. A node analyzes state and categorizes/routes
                2. The routing function returns a string key
                3. add_conditional_edges maps keys to destination nodes
                
                **Use Cases:**
                - **Intent classification**: Route user queries to specialized handlers
                - **Quality gates**: Retry, accept, or escalate based on quality checks
                - **A/B testing**: Route to different implementations
                - **Error handling**: Retry loops with max attempts
                
                **Important:** The routing function must return a key that exists in the mapping dict,
                or one of the reserved values like END.
                """,
                "key_concepts": ["add_conditional_edges", "Routing", "Branching", "Decision Trees"],
                "line_explanations": {
                    "60": "route_query returns a string that determines which path to take",
                    "69": "add_conditional_edges takes: source node, routing function, and mapping dict",
                    "70": "Mapping dict: returned string → destination node name",
                    "71": "If route_query returns 'weather', go to weather node",
                    "72": "If route_query returns 'shopping', go to shopping node",
                    "79": "All destination nodes eventually connect to END to terminate"
                },
                "references": {
                    "add_conditional_edges": {
                        "type": "Method",
                        "description": "Adds branching logic to graph. Routing function return value determines path.",
                        "example": "builder.add_conditional_edges('node', router, {'a': 'node_a', 'b': 'node_b'})"
                    },
                    "route_query": {
                        "type": "Function Pattern",
                        "description": "Routing function takes state and returns a routing key",
                        "example": "def router(state): return 'path_a' if condition else 'path_b'"
                    }
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "What does a routing function return?",
                "options": [
                    "A boolean",
                    "A string key mapping to a destination node",
                    "The next state",
                    "A node function"
                ],
                "correct": "A string key mapping to a destination node",
                "explanation": "The routing function returns a string that matches a key in the conditional edges mapping dict."
            },
            {
                "question": "How do you create a loop in LangGraph?",
                "options": [
                    "Use a while loop in Python",
                    "Route a conditional edge back to the same node",
                    "Use recursion",
                    "You can't create loops"
                ],
                "correct": "Route a conditional edge back to the same node",
                "explanation": "In add_conditional_edges, map a condition back to the same node name to create a loop."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 4: MULTI-AGENT SYSTEMS
    # ============================================================================
    {
        "title": "Multi-Agent Systems",
        "subtitle": "Collaborative AI agents",
        "emoji": "👥",
        "objectives": [
            "Understand multi-agent architecture",
            "Design agent communication patterns",
            "Implement supervisor patterns",
            "Handle agent delegation"
        ],
        "content": """
        ## Multi-Agent Systems
        
        Instead of one agent doing everything, use **specialized agents** that collaborate.
        
        ### Why Multiple Agents?
        
        **Single Agent Problems:**
        - Too many tools = confused decisions
        - Context gets cluttered
        - Hard to optimize for different tasks
        
        **Multi-Agent Benefits:**
        - Each agent specializes (research, write, code)
        - Clear separation of concerns
        - Easier to test and improve individually
        - Parallel execution where possible
        
        ### Common Architectures
        
        **1. Supervisor Pattern**
        ```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ↓               ↓               ↓
        ┌──────┐      ┌────────┐      ┌────────┐
        │AgentA│      │ AgentB │      │ AgentC │
        └──────┘      └────────┘      └────────┘
        ```
        A supervisor agent decides which specialist to call.
        
        **2. Pipeline Pattern**
        ```
        Research → Write → Edit → Publish
        ```
        Each agent passes work to the next.
        
        **3. Collaborative Pattern**
        ```
            ┌───────┐
            │Agent A│←────┐
            └───┬───┘     │
                ↓         │
            ┌───────┐     │
            │Agent B│─────┘
            └───────┘
        ```
        Agents iterate together on a shared artifact.
        
        ### Implementing Supervisor Pattern
        
        ```python
        def supervisor(state: State):
            '''Decides which agent should act next.'''
            decision = llm.invoke(f"Given: {state}, which agent should act?")
            return {"next_agent": decision}
        
        def route_to_agent(state: State):
            return state["next_agent"]
        
        builder.add_conditional_edges(
            "supervisor",
            route_to_agent,
            {
                "researcher": "researcher",
                "writer": "writer",
                "FINISH": END
            }
        )
        ```
        """,
        "code_examples": [
            {
                "title": "👥 Multi-Agent Blog Writing Team",
                "code": '''from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class BlogTeamState(TypedDict):
    topic: str
    research_notes: str
    outline: str
    draft: str
    final_blog: str
    messages: Annotated[list, operator.add]
    next: str

# Agent 1: Researcher
def researcher_node(state: BlogTeamState):
    """Gathers information on the topic."""
    topic = state["topic"]
    # In real use: search APIs, databases, etc.
    research = f"Research on {topic}: Key points include..."
    
    return {
        "research_notes": research,
        "messages": [{"role": "researcher", "content": f"Completed research on {topic}"}],
        "next": "outliner"  # Pass to next agent
    }

# Agent 2: Outliner
def outliner_node(state: BlogTeamState):
    """Creates an outline from research."""
    research = state["research_notes"]
    outline = f"Outline based on research:\\n1. Introduction\\n2. Main Points\\n3. Conclusion"
    
    return {
        "outline": outline,
        "messages": [{"role": "outliner", "content": "Created outline"}],
        "next": "writer"
    }

# Agent 3: Writer
def writer_node(state: BlogTeamState):
    """Writes the blog post from outline."""
    outline = state["outline"]
    draft = f"Blog post based on outline: {outline}\\n\\n[Full content here...]"
    
    return {
        "draft": draft,
        "messages": [{"role": "writer", "content": "Draft completed"}],
        "next": "editor"
    }

# Agent 4: Editor
def editor_node(state: BlogTeamState):
    """Reviews and finalizes the blog."""
    draft = state["draft"]
    final = f"EDITED: {draft}\\n\\n[Polished and ready to publish]"
    
    return {
        "final_blog": final,
        "messages": [{"role": "editor", "content": "Editing complete"}],
        "next": "FINISH"
    }

# Router: Determines next agent
def router(state: BlogTeamState):
    return state["next"]

# Build the team graph
builder = StateGraph(BlogTeamState)

# Add all agents as nodes
builder.add_node("researcher", researcher_node)
builder.add_node("outliner", outliner_node)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)

# Entry point
builder.set_entry_point("researcher")

# Each agent routes to the next
for node in ["researcher", "outliner", "writer", "editor"]:
    builder.add_conditional_edges(
        node,
        router,
        {
            "outliner": "outliner",
            "writer": "writer",
            "editor": "editor",
            "FINISH": END
        }
    )

# Compile
graph = builder.compile()

# Run the team
result = graph.invoke({
    "topic": "AI in Healthcare",
    "research_notes": "",
    "outline": "",
    "draft": "",
    "final_blog": "",
    "messages": [],
    "next": ""
})

print("Team workflow complete!")
print(f"Final blog: {result['final_blog'][:100]}...")
print(f"\\nMessage history:")
for msg in result["messages"]:
    print(f"  - {msg['role']}: {msg['content']}")''',
                "explanation": """
                **Multi-agent systems** use specialized agents working together.
                
                **This Example:**
                A blog writing team with 4 specialists:
                1. **Researcher** → Gathers information
                2. **Outliner** → Structures the content
                3. **Writer** → Creates the draft
                4. **Editor** → Polishes and finalizes
                
                **Key Patterns:**
                - Each agent updates state and sets `next` for routing
                - Shared state lets agents see each other's work
                - Message history tracks who did what
                - Sequential pipeline (could be parallel in some cases)
                
                **Variations:**
                - **Parallel**: Multiple researchers working on different aspects
                - **Iterative**: Writer → Editor → (if not good) → Writer again
                - **Dynamic**: Supervisor decides which agents to call based on task
                """,
                "key_concepts": ["Multi-Agent", "Pipeline", "Specialization", "Agent Teams"],
                "line_explanations": {
                    "17": "Each agent is just a node function that updates state",
                    "25": "Agent updates research_notes and sets next='outliner' to route",
                    "78": "Router function reads the 'next' field from state to determine path",
                    "85": "All agents use the same conditional edge pattern for routing"
                },
                "references": {
                    "Multi-Agent Pattern": {
                        "type": "Architecture",
                        "description": "Multiple specialized agents collaborating on a shared state",
                        "example": "Researcher → Writer → Editor pipeline"
                    },
                    "Specialization": {
                        "type": "Design Principle",
                        "description": "Each agent handles one specific task, making them simpler and more reliable",
                        "example": "Researcher only searches, Writer only writes"
                    }
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "What's the main benefit of multi-agent systems?",
                "options": [
                    "They're faster than single agents",
                    "Each agent can specialize in one task",
                    "They use less memory",
                    "They don't need tools"
                ],
                "correct": "Each agent can specialize in one task",
                "explanation": "Specialized agents are simpler, easier to maintain, and often perform better than one agent doing everything."
            },
            {
                "question": "In the Supervisor pattern, who decides which agent acts?",
                "options": [
                    "The user",
                    "The supervisor agent",
                    "Random selection",
                    "The first agent"
                ],
                "correct": "The supervisor agent",
                "explanation": "The supervisor agent analyzes the state and decides which specialist agent should act next."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 5: HUMAN-IN-THE-LOOP
    # ============================================================================
    {
        "title": "Human-in-the-Loop",
        "subtitle": "Integrating human oversight",
        "emoji": "👤",
        "objectives": [
            "Add human approval checkpoints",
            "Implement human fallback",
            "Handle human input during execution",
            "Design review workflows"
        ],
        "content": """
        ## Human-in-the-Loop
        
        Not everything should be fully automated. **Human oversight** is crucial for:
        - Sensitive decisions
        - Quality approval
        - Edge cases
        - Learning from feedback
        
        ### Types of Human Intervention
        
        **1. Approval Gate**
        ```
        Agent generates → Human approves → Continue
                            ↓ (if reject)
                        Revise and retry
        ```
        
        **2. Human Fallback**
        ```
        Agent tries → (if fails after 3x) → Escalate to human
        ```
        
        **3. Interactive Input**
        ```
        Agent asks question → Human provides answer → Continue
        ```
        
        ### Implementing in LangGraph
        
        LangGraph supports breakpoints and human input:
        
        ```python
        # Add an interrupt before a sensitive node
        builder.add_node("human_approval", human_approval_node)
        
        # The human_approval node can:
        # - Show current state to user
        # - Wait for user input
        # - Update state based on decision
        ```
        
        ### Breakpoint Pattern
        
        ```python
        def sensitive_action(state: State):
            # Check if we have approval
            if not state.get("approved"):
                return {"waiting_for_approval": True}
            
            # Proceed with action
            return {"result": perform_action(state)}
        
        def approval_node(state: State):
            if state.get("waiting_for_approval"):
                # In real app: show UI, wait for human
                # For demo, auto-approve
                return {"approved": True, "waiting_for_approval": False}
        ```
        """,
        "code_examples": [
            {
                "title": "👤 Approval Workflow",
                "code": '''from langgraph.graph import StateGraph, END
from typing import TypedDict

class ApprovalState(TypedDict):
    proposal: str
    approved: bool
    rejected: bool
    feedback: str
    final_decision: str
    stage: str  # 'draft', 'pending_approval', 'approved', 'rejected'

def generate_proposal(state: ApprovalState):
    """Generate a proposal."""
    return {
        "proposal": "Proposal: Implement new feature X with budget $10,000",
        "stage": "pending_approval"
    }

def human_review(state: ApprovalState):
    """Simulate human review of the proposal."""
    proposal = state["proposal"]
    
    # In a real app, this would:
    # 1. Show proposal to user via UI
    # 2. Wait for user to click Approve/Reject
    # 3. Collect feedback
    
    # Simulating: Auto-approve if budget < $50,000
    if "$10,000" in proposal:
        return {
            "approved": True,
            "rejected": False,
            "feedback": "Budget is reasonable. Approved!",
            "stage": "approved"
        }
    else:
        return {
            "approved": False,
            "rejected": True,
            "feedback": "Budget too high. Please revise.",
            "stage": "rejected"
        }

def implement(state: ApprovalState):
    """Execute the approved proposal."""
    return {
        "final_decision": f"✅ IMPLEMENTED: {state['proposal']}\\nFeedback: {state['feedback']}"
    }

def revise(state: ApprovalState):
    """Send back for revision."""
    return {
        "final_decision": f"❌ REJECTED: {state['proposal']}\\nFeedback: {state['feedback']}"
    }

def route_approval(state: ApprovalState):
    """Route based on approval decision."""
    if state["approved"]:
        return "implement"
    else:
        return "revise"

# Build graph
builder = StateGraph(ApprovalState)

builder.add_node("generate", generate_proposal)
builder.add_node("review", human_review)
builder.add_node("implement", implement)
builder.add_node("revise", revise)

builder.set_entry_point("generate")
builder.add_edge("generate", "review")
builder.add_conditional_edges(
    "review",
    route_approval,
    {
        "implement": "implement",
        "revise": "revise"
    }
)
builder.add_edge("implement", END)
builder.add_edge("revise", END)

graph = builder.compile()

# Run
result = graph.invoke({
    "proposal": "",
    "approved": False,
    "rejected": False,
    "feedback": "",
    "final_decision": "",
    "stage": "draft"
})

print(result["final_decision"])''',
                "explanation": """
                **Human-in-the-loop** adds checkpoints for human oversight.
                
                **The Pattern:**
                1. Agent generates output (proposal, code, content)
                2. Flow pauses at review node for human decision
                3. Human approves → continues to implementation
                4. Human rejects → goes to revision/fallback
                
                **In Real Applications:**
                - Use `interrupt()` to pause graph execution
                - Store thread ID to resume later
                - Show state in UI for human review
                - Use `graph.invoke()` with updated state to resume
                
                **Use Cases:**
                - Content moderation
                - Financial approvals
                - Code review
                - Medical diagnosis support
                """,
                "key_concepts": ["Human-in-the-loop", "Approval", "Breakpoints", "Interrupts"],
                "line_explanations": {
                    "22": "In real apps, this node would show UI and wait for human input",
                    "46": "route_approval determines path based on human's approve/reject decision",
                    "60": "Conditional edges let us branch to implement (approved) or revise (rejected)"
                },
                "references": {
                    "interrupt": {
                        "type": "Function",
                        "description": "Pauses graph execution and returns control to the application for human input",
                        "example": "if not approved: return interrupt('Waiting for approval')"
                    }
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "When should you use human-in-the-loop?",
                "options": [
                    "For every agent action",
                    "For sensitive decisions or when confidence is low",
                    "Only at the end",
                    "Never, agents should be fully autonomous"
                ],
                "correct": "For sensitive decisions or when confidence is low",
                "explanation": "Human oversight is valuable for sensitive operations, edge cases, or when the agent's confidence is below a threshold."
            }
        ]
    },
    
    # ============================================================================
    # CHAPTER 6: PERSISTENCE & MEMORY
    # ============================================================================
    {
        "title": "Persistence & Memory",
        "subtitle": "Long-running and resumable workflows",
        "emoji": "📝",
        "objectives": [
            "Enable state persistence",
            "Resume interrupted workflows",
            "Build long-term memory",
            "Handle errors gracefully"
        ],
        "content": """
        ## Persistence in LangGraph
        
        **Persistence** lets you:
        - Save workflow state to disk/database
        - Resume after crashes or restarts
        - Build long-running conversations
        - Debug by inspecting past states
        
        ### Checkpointing
        
        LangGraph automatically saves state at each step when configured:
        
        ```python
        from langgraph.checkpoint.sqlite import SqliteSaver

        # Create checkpointer
        checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
        
        # Compile with persistence
        graph = builder.compile(checkpointer=checkpointer)
        
        # Run with thread_id for tracking
        config = {"configurable": {"thread_id": "user_123"}}
        result = graph.invoke(initial_state, config)
        ```
        
        ### Thread IDs
        
        **Thread ID** identifies a unique conversation/workflow:
        - Resume: Use same thread_id to continue
        - New: Use new thread_id for fresh start
        - List: Query all threads for a user
        
        ```python
        # Continue existing conversation
        config = {"configurable": {"thread_id": "user_123"}}
        result = graph.invoke({"message": "Follow-up question"}, config)
        ```
        
        ### Error Recovery
        
        ```python
        from langgraph.pregel import RetryPolicy

        builder.add_node(
            "unreliable_api_call",
            unreliable_function,
            retry=RetryPolicy(max_attempts=3)
        )
        ```
        
        ### Time Travel (Debugging)
        
        ```python
        # Get state history
        history = list(graph.get_state_history(config))
        
        # Replay from a specific point
        past_state = history[5]  # Go back 5 steps
        result = graph.invoke(past_state.values, config)
        ```
        """,
        "code_examples": [
            {
                "title": "📝 Persistent Conversation",
                "code": '''from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    user_name: str
    session_id: str

def chatbot_node(state: ConversationState):
    """Simple chatbot that echoes back (in real use, call LLM)."""
    last_message = state["messages"][-1] if state["messages"] else "Hello!"
    response = f"Echo: {last_message}"
    
    return {
        "messages": [{"role": "assistant", "content": response}]
    }

# Build graph
builder = StateGraph(ConversationState)
builder.add_node("chatbot", chatbot_node)
builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")

# Add persistence - saves to SQLite database
checkpointer = SqliteSaver.from_conn_string(":memory:")  # Use file for real persistence
graph = builder.compile(checkpointer=checkpointer)

# Conversation 1: Thread "user_123"
print("=== Conversation 1 (Thread: user_123) ===")
config1 = {"configurable": {"thread_id": "user_123"}}

result1 = graph.invoke({
    "messages": [{"role": "user", "content": "Hi, I'm Alice"}],
    "user_name": "Alice",
    "session_id": "session_1"
}, config1)

print(f"Turn 1: {result1['messages'][-1]}")

# Continue same conversation
result2 = graph.invoke({
    "messages": [{"role": "user", "content": "What's my name?"}]
}, config1)

print(f"Turn 2: {result2['messages'][-1]}")
print(f"Total messages: {len(result2['messages'])}\\n")

# Conversation 2: Thread "user_456" (new user)
print("=== Conversation 2 (Thread: user_456) ===")
config2 = {"configurable": {"thread_id": "user_456"}}

result3 = graph.invoke({
    "messages": [{"role": "user", "content": "Hi, I'm Bob"}],
    "user_name": "Bob",
    "session_id": "session_2"
}, config2)

print(f"Turn 1: {result3['messages'][-1]}")
print(f"Total messages: {len(result3['messages'])}\\n")

# Resume Conversation 1
print("=== Resuming Conversation 1 ===")
result4 = graph.invoke({
    "messages": [{"role": "user", "content": "Remember me?"}]
}, config1)

print(f"Turn 3: {result4['messages'][-1]}")
print(f"Total messages in conv 1: {len(result4['messages'])}")

# Show state history
print("\\n=== State History for Conversation 1 ===")
history = list(graph.get_state_history(config1))
print(f"Number of saved states: {len(history)}")''',
                "explanation": """
                **Persistence** enables long-running conversations and fault tolerance.
                
                **Key Concepts:**
                - **Checkpointer**: Saves state after each step to a database
                - **Thread ID**: Unique identifier for a conversation/workflow
                - **Resume**: Use same thread_id to continue from where you left off
                
                **How It Works:**
                1. Each node execution saves state to the checkpointer
                2. Thread ID tracks which conversation the state belongs to
                3. If app restarts, state is loaded from database
                4. Can replay, fork, or time-travel through states
                
                **Use Cases:**
                - Long conversations with memory
                - Multi-step approvals that span days
                - Error recovery and retries
                - Audit trails and debugging
                """,
                "key_concepts": ["Checkpointer", "Thread ID", "Persistence", "State History"],
                "line_explanations": {
                    "21": "SqliteSaver persists state to SQLite database",
                    "22": ":memory: is for testing. Use a file path for real persistence.",
                    "25": "checkpointer parameter enables automatic state saving",
                    "29": "thread_id identifies this specific conversation",
                    "50": "Same thread_id = continues existing conversation with full history",
                    "60": "Different thread_id = fresh conversation, no shared history",
                    "78": "get_state_history returns all saved states for replay/debugging"
                },
                "references": {
                    "SqliteSaver": {
                        "type": "Class",
                        "description": "Saves graph state to SQLite database after each step",
                        "example": "checkpointer = SqliteSaver.from_conn_string('app.db')"
                    },
                    "thread_id": {
                        "type": "Config Parameter",
                        "description": "Unique identifier for a conversation. Same ID = resume, New ID = fresh start",
                        "example": "config = {'configurable': {'thread_id': 'conv_123'}}"
                    },
                    "get_state_history": {
                        "type": "Method",
                        "description": "Returns all saved states for a thread, enabling time-travel debugging",
                        "example": "states = list(graph.get_state_history(config))"
                    }
                }
            }
        ],
        "demo": None,
        "quiz": [
            {
                "question": "What does a checkpointer do?",
                "options": [
                    "Checks code syntax",
                    "Saves state after each step",
                    "Validates user input",
                    "Monitors performance"
                ],
                "correct": "Saves state after each step",
                "explanation": "A checkpointer persists the graph state to storage after each node execution."
            },
            {
                "question": "How do you resume a conversation?",
                "options": [
                    "Call resume()",
                    "Use the same thread_id",
                    "Reload the page",
                    "Start over"
                ],
                "correct": "Use the same thread_id",
                "explanation": "Using the same thread_id tells LangGraph to load the saved state for that conversation."
            }
        ]
    }
]
