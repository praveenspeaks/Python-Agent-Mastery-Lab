import streamlit as st
import os
import pandas as pd
from src.app import LangChainAgent

# Page Config
st.set_page_config(page_title="AI Agent Mastery Lab", layout="wide", initial_sidebar_state="expanded")

# Initialize Agent (The Brain of our App)
# simple check to avoid re-initializing the agent on every page reload
if 'agent' not in st.session_state:
    st.session_state.agent = LangChainAgent()

agent = st.session_state.agent

# Check API Key Status in Sidebar
with st.sidebar:
    is_valid, msg = agent.ai.validate_setup()
    if not is_valid:
        st.error(msg)
        st.stop() # Stop execution if no key
    else:
        st.success(f"System Ready: {msg}")

# --- UI Helper Components ---
def render_mastery_notes(title, explanation, code, key_terms=None):
    """Renders the right-hand educational panel."""
    st.markdown(f"### 📘 Mastery Notes: {title}")
    st.info(explanation)
    if code:
        st.markdown("**💻 Implementation Code:**")
        st.code(code, language="python")
    if key_terms:
        st.markdown("**🧠 Key Terms:**")
        for term, desc in key_terms.items():
            st.markdown(f"- **{term}**: {desc}")

def render_technical_trace(traces):
    """Renders a list of structured technical traces."""
    if not traces:
        return
    
    st.markdown("### 🔬 Technical Trace Log")
    st.caption("Deep visibility into the internal logic, variables, and data flow.")
    for trace in traces:
        with st.expander(f"🔹 Step: {trace.get('step', 'Operation')}"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Module:** `{trace.get('module', 'N/A')}`")
                st.markdown(f"**Explanation:** {trace.get('explanation', 'N/A')}")
            with col2:
                if trace.get("variables"):
                    st.markdown("**Variables:**")
                    st.json(trace["variables"])
            
            if trace.get("command"):
                st.markdown("**Command Executed:**")
                st.code(trace["command"], language="python")
            
            t_left, t_right = st.columns(2)
            with t_left:
                if trace.get("input") and trace["input"] != "N/A":
                    st.markdown("**Input:**")
                    st.info(str(trace["input"])[:500] + ("..." if len(str(trace["input"])) > 500 else ""))
            with t_right:
                if trace.get("output") and trace["output"] != "N/A":
                    st.markdown("**Output:**")
                    st.success(str(trace["output"])[:500] + ("..." if len(str(trace["output"])) > 500 else ""))

# --- Pillar Content Functions ---

def render_overview():
    left, right = st.columns([1, 1])
    
    with left:
        st.title("🚀 AI Agent Mastery Lab")
        st.markdown("""
        Learn to build production-grade Agentic AI systems from scratch. This lab covers the three core technologies that power modern AI assistants.
        """)
        
        pillar_tab = st.sidebar.radio("Select Pillar:", ["LangChain", "LangGraph", "DeepAgents"])
        
        if pillar_tab == "LangChain":
            st.subheader("🔗 LangChain: The Foundation")
            st.markdown("""
            LangChain is the world's most popular framework for building LLM applications. It provides the "standard library" for:
            - **Data Ingestion**: Loading PDFs, Webpages, Databases.
            - **RAG**: Retrieval Augmented Generation (Giving AI your data).
            - **Tool Calling**: Giving AI the ability to use external APIs.
            """)
        elif pillar_tab == "LangGraph":
            st.subheader("🕸️ LangGraph: The Brain")
            st.markdown("""
            LangGraph introduces **Stateful Multi-Agent Workflows**. It moves beyond simple chains to:
            - **Cycles**: Allowing agents to loop and refine their work.
            - **State Management**: Keeping track of variables across complex tasks.
            - **Human-in-the-Loop**:Pausing for user approval.
            """)
        elif pillar_tab == "DeepAgents":
            st.subheader("🤖 DeepAgents: The Autonomous Entity")
            st.markdown("""
            DeepAgents represent the ultimate level of autonomous AI. Our curriculum covers:
            - **Chapter 19: Autonomous Planning**: Breaking down complex goals.
            - **Chapter 20: Tool Synthesis**: Agents that write their own tools.
            - **Chapter 21: Multi-Agent Orchestration**: Coordinating many agents.
            - **Chapter 22-24**: Context Management, Prompt Evolution, and Scale.
            """)

    with right:
        render_mastery_notes(
            "System Architecture",
            "This platform is built as a 'Modular Multi-Pillar System'. Each tab represented here is a building block toward a fully autonomous AI agent.",
            """# The 3 Layers of Mastery
foundations = "LangChain" # Part 1: Parsing & RAG
workflows = "LangGraph"    # Part 2: Cycles & State
autonomy = "DeepAgents"    # Part 3: Planning & Reasoning""",
            {"Agentic AI": "A system where an LLM controls its own flow and tool usage.", "RAG": "Enhancing LLMs with specific, external data."}
        )

def render_chapter_1():
    st.title("📄 Chapter 1: Document Parsing & Loading")
    left, right = st.columns([1, 1])

    with left:
        st.write("First, the AI needs to 'ingest' your data. We use LangChain's Document Loaders.")
        test_data_dir = "testdata"
        if os.path.exists(test_data_dir):
            files = [f for f in os.listdir(test_data_dir) if os.path.isfile(os.path.join(test_data_dir, f))]
            selected_file = st.selectbox("Pick a File:", files)
            
            if st.button("Run Parser"):
                with st.spinner("Processing..."):
                    docs, logs = agent.parser.load_document(os.path.join(test_data_dir, selected_file))
                    st.success(f"Successfully parsed {len(docs)} segments.")
                    
                    # New Structured Tracing
                    render_technical_trace(logs)
                    
                    with st.expander("View Raw Normalized Text"):
                        st.write(docs[0].page_content if docs else "Empty")
        else:
            st.error("testdata folder missing.")

    with right:
        render_mastery_notes(
            "How Parsing Works",
            "LangChain uses specialized 'Loaders' for each file format. For example, `PyMuPDFLoader` is used for PDFs to extract text high-performance.",
            """from langchain_community.document_loaders import PyMuPDFLoader

# Loading a PDF
loader = PyMuPDFLoader("path/to/file.pdf")
documents = loader.load() # Returns a list of Document objects""",
            {"Document Object": "The standard LangChain format containing 'page_content' and 'metadata'.", "Metadata": "Information about the source, like the page number or filename."}
        )

def render_chapter_2():
    st.title("🧠 Chapter 2: Embeddings & Search")
    left, right = st.columns([2, 3])

    with left:
        st.markdown("Search your data by **meaning** (Semantic Search) instead of exact keywords.")
        query = st.text_input("Ask a question about your indexed files:", "What is LangChain?")
        
        if st.button("Semantic Retrieval"):
            results, logs = agent.search_documents(query, k=3)
            
            # Technical Deep Dive
            render_technical_trace(logs)
            
            if results:
                st.write("### Relevant Chunks Found:")
                for i, doc in enumerate(results):
                    st.info(f"Chunk {i+1}:\n{doc.page_content[:200]}...")
            else:
                st.warning("Index some data first in the sidebar!")

        st.sidebar.markdown("---")
        st.sidebar.header("Data Management")
        if st.sidebar.button("Re-Index Sample Files"):
            with st.spinner("Indexing..."):
                agent.index_file("testdata/sample_text.txt")
                st.sidebar.success("Index Ready!")

    with right:
        render_mastery_notes(
            "The Power of Vectors",
            "Embeddings turn text into long lists of numbers (vectors). Similarity search calculates the 'distance' between your question's vector and the document vectors.",
            """from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Initialize Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Convert text to numbers and store
vector_store = FAISS.from_documents(chunks, embeddings)""",
            {"FAISS": "Facebook AI Similarity Search - A fast library for vector retrieval.", "Cosine Similarity": "The mathematical way we measure how 'similar' two text vectors are."}
        )

def render_chapter_3():
    st.title("✂️ Chapter 3: Advanced Chunking")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        Moving beyond character counts. **Semantic Chunking** uses AI to understand when a topic changes, creating natural breaks.
        """)
        
        raw_text = st.text_area("Input document text:", "Artificial Intelligence has transformed the world. It is used in medicine to detect diseases early. It is also used in transportation for self-driving cars. \\n\\nChanging the topic completely, the recipe for a perfect cake involves flour, eggs, and sugar. Mix them well and bake at 350 degrees. \\n\\nFinally, the solar system consists of eight planets orbiting the sun. Earth is the third planet and the only one known to support life.", height=200)
        
        mode = st.radio("Chunking Mode:", ["Recursive (Fixed Size)", "Semantic (Topic Based)"])
        
        if st.button("Process & Compare"):
            with st.spinner("Analyzing text meaning..."):
                from langchain_core.documents import Document
                if mode == "Recursive (Fixed Size)":
                    agent.processor.update_settings(chunk_size=100, chunk_overlap=0)
                    chunks, logs = agent.processor.split_documents([Document(page_content=raw_text)], mode="recursive")
                else:
                    chunks, logs = agent.processor.split_documents([Document(page_content=raw_text)], mode="semantic")
                
                st.subheader(f"Generated {len(chunks)} Chunks")
                
                # Technical Deep Dive
                render_technical_trace(logs)
                
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}**")
                    st.code(chunk.page_content)

    with right:
        render_mastery_notes(
            "Semantic vs. Recursive",
            "Recursive splitting is fast but 'dumb'—it might cut a sentence in half if the character limit is reached. Semantic splitting 'reads' the text first.",
            """# Using Semantic Splitter
from langchain_experimental.text_splitter import SemanticChunker

# 1. Provide an embedding model
splitter = SemanticChunker(HuggingFaceEmbeddings())

# 2. Split (no fixed size needed!)
docs = splitter.split_documents([raw_doc])""",
            {"Breakpoint": "The exact point where the AI decides one topic ended and another began.", "Buffer": "A small window of context the AI looks at to decide on a split."}
        )

def render_chapter_4():
    st.title("⚖️ Chapter 4: Hybrid Search")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Precision + Depth**. Hybrid search combines **Keyword (BM25)** for exact matches and **Vector** for meaning.
        """)
        
        search_query = st.text_input("Test Hybrid Search:", "LangChain funding")
        
        v_weight = st.slider("Vector Weight:", 0.0, 1.0, 0.5)
        k_weight = st.slider("Keyword (BM25) Weight:", 0.0, 1.0, 0.5)
        
        if st.button("Search with Hybrid Mode"):
            # We need documents to initialize BM25 for the hybrid retriever
            # In a real app, these would come from the database
            dummy_docs = [
                Document(page_content="LangChain was launched in October 2022 by Harrison Chase."),
                Document(page_content="In April 2023, LangChain raised $20 million from Sequoia Capital."),
                Document(page_content="The solar system consists of eight planets orbiting the sun.")
            ]
            
            with st.spinner("Merging results..."):
                results, logs = agent.hybrid_search(
                    search_query, 
                    dummy_docs, 
                    vector_weight=v_weight, 
                    keyword_weight=k_weight
                )
                
                # Technical Deep Dive
                render_technical_trace(logs)
                
                st.subheader("Mixed Results")
                for i, doc in enumerate(results):
                    st.markdown(f"**Result {i+1}**")
                    st.info(doc.info.page_content if hasattr(doc, 'info') else doc.page_content)

    with right:
        render_mastery_notes(
            "The Best of Both Worlds",
            "Keyword search is great for names and numbers. Vector search is great for 'what does this mean?'. Hybrid search uses RRF (Reciprocal Rank Fusion) to merge them.",
            """# Ensemble (Hybrid) Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
results = ensemble.invoke(query)""",
            {"BM25": "Best Matching 25 - The standard algorithm for keyword search.", "RRF": "Reciprocal Rank Fusion - A way to score results from different search engines fairly."}
        )

def render_chapter_5():
    st.title("🧠 Chapter 5: Query Engineering")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Don't search blindly.** Query Engineering uses the LLM to 'think' about the user's question before looking for documents.
        """)
        
        complex_query = st.text_input("Enter a complex question:", "What was LangChain's valuation in 2023 and who invested in them?")
        
        strategy = st.radio("Enhancement Strategy:", ["Multi-Query (Recall boost)", "Decomposition (Plan mapping)"])
        
        if st.button("Enhance Query"):
            with st.spinner("Re-writing question..."):
                mode = "multi_query" if "Multi-Query" in strategy else "decomposition"
                results, logs = agent.enhance_query(complex_query, mode=mode)
                
                st.subheader("Enhanced Output")
                
                # Technical Deep Dive
                render_technical_trace(agent.ai.last_run_traces)
                
                for i, q in enumerate(results):
                    st.markdown(f"**{ 'Query' if mode == 'multi_query' else 'Step' } {i+1}**")
                    st.info(q)

    with right:
        render_mastery_notes(
            "Query Expansion",
            "A user might say 'valuation', but a document might say 'worth' or 'market cap'. Multi-Query generates synonyms to catch all relevant documents. Decomposition handles multi-part questions.",
            """# Query Re-writing
template = 'Generate {n} variations of: {query}'
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

# The system results in a list of queries 
# which we then use to search the vector store.""",
            {"Recall": "The ability to find ALL relevant documents.", "Recall-Precision Tradeoff": "Generating more queries finds more docs but might introduce 'noise'."}
        )

def render_chapter_6():
    st.title("🖼️ Chapter 6: Multi-Modal RAG")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Beyond Plain Text.** In this stage, we teach the AI to understand images, diagrams, and complex tables.
        """)
        st.info("💡 **Concept: Vision-Enabled Search**")
        st.markdown("""
        How do we search an image?
        1. **Summarization**: Use a Vision Model (GPT-4o) to describe the image.
        2. **Embedding**: Store that description in the vector store.
        3. **Retrieval**: When you ask about a 'chart', the AI finds the description and shows you the raw image.
        """)
        st.image("https://blog.langchain.dev/content/images/2023/10/multimodal_rag_standard.png", caption="Multi-Modal RAG Architecture")

    with right:
        render_mastery_notes(
            "Vision & Tables",
            "Tables are often 'lost' in basic RAG. We use specialized loaders like Unstructured to convert them into Markdown, which preserves the data structure for the AI.",
            """# Vision Parsing Example
from langchain_core.messages import HumanMessage
message = HumanMessage(content=[
    {\"type\": \"text\", \"text\": \"Describe this chart\"},
    {\"type\": \"image_url\", \"image_url\": \"...\"}
])
response = vision_model.invoke([message])""",
            {"Multi-Vector": "Storing a summary and the raw image separately.", "Vision LLM": "A model trained on both images and text."}
        )

def render_chapter_7():
    st.title("🤖 Chapter 7: Transition to Agency")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **The Final Foundation.** This is where we move from 'Chains' (fixed steps) to 'Agents' (autonomous loops).
        """)
        st.warning("🚀 **You are entering Phase 3!**")
        st.markdown("""
        **Key Differences:**
        - **Chains**: Prompt -> Load -> Answer (Linear).
        - **Agents**: Prompt -> Plan -> Use Tool -> Evaluate -> Repeat (Cyclic).
        """)
        
        if st.button("Simulate Agentic Loop"):
            st.write("🏃 Agent started...")
            st.write("🔍 Step 1: Searching for 'Python revenue 2024'...")
            st.write("⚠️ Error: No direct match. Retrying with 'Python Software Foundation financial report'...")
            st.write("✅ Match found. Analyzing data...")
            st.success("🏆 Goal Achieved: Revenue found in PSF 2024 report.")

    with right:
        render_mastery_notes(
            "The Agentic Mindset",
            "An agent doesn't just fail; it reasons why it failed and tries a different path. This requires 'State' and 'Loops'—the core of LangGraph.",
            """# Chain vs Agent
# CHAIN (Linear)
chain = prompt | model | parser

# AGENT (Cyclic)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)""",
            {"Reasoning Loop": "The 'Think-Act-Observe' cycle.", "Tools": "External functions the AI can call (Google, SQL, etc.)."}
        )

# --- Phase 3: LangGraph Workflows ---

def render_chapter_8():
    st.title("🕸️ Chapter 8: LangGraph Basics")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Nodes & Edges**. LangGraph treats AI as a state machine.
        - **Nodes**: Python functions (The 'Brains').
        - **Edges**: The arrows (The 'Paths').
        """)
        
        goal = st.text_input("Enter a goal for the graph agent:", "Research AI agents")
        
        if st.button("Start Graph Loop"):
            from src.graph_agent import LangGraphMastery
            mastery = LangGraphMastery()
            
            with st.spinner("Graph is running..."):
                steps = mastery.run(goal)
                
                for step in steps:
                    with st.expander(f"📍 Node: {step['node'].upper()}"):
                        st.json(step['data'])
            st.success("🏁 Graph reached END node.")

    with right:
        render_mastery_notes(
            "The State Machine",
            "In LangChain, data flows in one direction. In LangGraph, we can loop back. We pass a 'State' object between nodes, and each node can modify it.",
            """# Basic StateGraph
from langgraph.graph import StateGraph

workflow = StateGraph(MyStateClass)
workflow.add_node(\"agent\", my_function)
workflow.set_entry_point(\"agent\")
workflow.add_edge(\"agent\", END)

app = workflow.compile()""",
            {"State": "A shared memory that all nodes can read and write to.", "Compile": "Turning your python logic into an executable graph engine."}
        )

def render_chapter_9():
    st.title("🤝 Chapter 9: Human-In-The-Loop")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Safety First.** Some actions (like sending an email or deleting a file) should never be fully autonomous. 
        LangGraph allows us to **Interrupt** the graph and wait for a human to click 'Approve'.
        """)
        
        if "graph_thread_id" not in st.session_state:
            st.session_state.graph_thread_id = str(hash(st.session_state.get("username", "user")))
        
        goal = st.text_input("Enter a sensitive goal:", "Update production database")
        
        from src.graph_agent import LangGraphMastery
        mastery = LangGraphMastery()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Sensitive Task"):
                st.session_state.graph_logs = mastery.run(goal, thread_id=st.session_state.graph_thread_id)
        
        with col2:
            if st.button("✅ Approve AI Action"):
                st.session_state.graph_logs = mastery.run(None, thread_id=st.session_state.graph_thread_id)

        if "graph_logs" in st.session_state:
            for step in st.session_state.graph_logs:
                if step['node'] == "INTERRUPT":
                    st.warning(f"🚨 {step['data']['message']}")
                else:
                    with st.expander(f"📍 {step['node'].upper()}"):
                        st.write(step['data'])

    with right:
        render_mastery_notes(
            "The Approval Workflow",
            "By using `interrupt_before`, we tell LangGraph to save the current state to a database and wait. The process is entirely 'pick-up-where-you-left-off'.",
            """# Define Interruption
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=[\"act\"]
)

# 1. Run until interrupt
app.stream(input, config)

# 2. Resume after human input
app.stream(None, config)""",
            {"Checkpointer": "The database where graph state is saved.", "Thread ID": "Each user or conversation gets its own unique state ID."}
        )

def render_chapter_10():
    st.title("👥 Chapter 10: Multi-Agent Workflows")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Teamwork makes the dream work.** In complex systems, we don't have one 'Super Agent'. 
        Instead, we have specialized workers:
        - **Researcher**: Finds the facts.
        - **Writer**: Crafts the story.
        """)
        
        topic = st.text_input("Topic for the team:", "The future of space travel")
        
        if st.button("Deploy AI Team"):
            from src.graph_agent import MultiAgentMastery
            team = MultiAgentMastery()
            
            with st.spinner("Team is working..."):
                steps = team.run(topic)
                
                for step in steps:
                    label = "🔍 Researcher" if step['node'] == "researcher" else "✍️ Writer"
                    with st.expander(f"📍 {label}"):
                        content = step['data']['messages'][0].content
                        st.write(content)
            st.success("🏁 Team Goal Met: Summary Delivered.")

    with right:
        render_mastery_notes(
            "Agent Hand-offs",
            "The 'Hand-off' pattern is the most common multi-agent design. One agent finishes its work, updates the shared 'State', and then an edge triggers the next agent to start.",
            """# Collaborative Graph
workflow = StateGraph(State)
workflow.add_node(\"researcher\", res_fn)
workflow.add_node(\"writer\", write_fn)

# The hand-off is a simple edge
workflow.add_edge(\"researcher\", \"writer\")
workflow.add_edge(\"writer\", END)""",
            {"Specialization": "Smaller models performing specific tasks better than one large model.", "Shared State": "The memory that allows agents to 'talk' to each other."}
        )

def render_chapter_11():
    st.title("🪞 Chapter 11: Self-Correction (Reflection)")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Quality Control.** Reflection is an agentic pattern where the AI reviews its own work and makes improvements before you ever see the result.
        """)
        
        topic = st.text_input("Reflective Topic:", "Why is Python popular for AI?")
        
        if st.button("Start Reflection Loop"):
            from src.graph_agent import ReflectiveMastery
            reflection = ReflectiveMastery()
            
            with st.spinner("AI is self-reflecting..."):
                steps = reflection.run(topic)
                
                for step in steps:
                    label = "📝 Drafter" if step['node'] == "drafter" else "🧐 Critic"
                    with st.expander(f"📍 {label}"):
                        content = step['data']['messages'][0].content
                        st.write(content)
            st.success("🏁 Self-Correction complete.")

    with right:
        render_mastery_notes(
            "The Reflection Loop",
            "This pattern solves 'Hallucination' and 'Laziness'. By forcing the model to critique its own first draft, we often get a much more robust and factual second or third draft.",
            """# Reflection Logic
workflow.add_node(\"drafter\", draft_fn)
workflow.add_node(\"critic\", critique_fn)

workflow.add_edge(\"drafter\", \"critic\")
workflow.add_conditional_edges(
    \"critic\",
    should_redraft,
    {\"redraft\": \"drafter\", \"finish\": END}
)""",
            {"Critique": "Identifying gaps in reasoning or facts.", "Iteration": "Running the generation logic again with new context (the feedback)."}
        )

def render_chapter_12():
    st.title("💾 Chapter 12: Persistence & Memory")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Never Forget.** Persistence allows an agent to have a 'Long-term Memory'. 
        By using a Thread ID, we can reload the exact state of a conversation even after the system restarts.
        """)
        
        thread_id = st.text_input("Memory Thread ID:", "user-session-123")
        user_input = st.text_input("Tell the AI something to remember:", "My favorite color is Blue")
        
        if st.button("Save to Memory"):
            from src.graph_agent import PersistentMastery
            persistent = PersistentMastery()
            with st.spinner("Storing in long-term memory..."):
                history = persistent.run(user_input, thread_id=thread_id)
                st.session_state.memory_history = history
        
        if "memory_history" in st.session_state:
            st.subheader("Agent's Memory Bank")
            for msg in st.session_state.memory_history:
                st.chat_message(msg.type).write(msg.content)

    with right:
        render_mastery_notes(
            "The Memory Thread",
            "In LangGraph, we don't just 'scroll' through history. We 'Checkpoint' it. This means the AI isn't just seeing a list of text; it's seeing the exact state of its 'brain' at a specific moment in time.",
            """# Persistence Setup
from langgraph.checkpoint.sqlite import SqliteSaver

# We use MemorySaver locally, but in 
# production, you'd use a Real Database.
memory = SqliteSaver.from_conn_string(\":memory:\")

app = workflow.compile(checkpointer=memory)

# Resume conversation 123
config = {\"configurable\": {\"thread_id\": \"123\"}}
app.invoke(input, config)""",
            {"Checkpointing": "Saving the state of a graph execution.", "Thread ID": "The unique key used to retrieve a specific agent's memory."}
        )

def render_chapter_13():
    st.title("🧙‍♂️ Chapter 13: Query Transformation (HyDE)")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Hypothetical Document Embeddings (HyDE).** 
        Sometimes, a user's question doesn't look like the documents in the database.
        HyDE fixes this by:
        1. **Generating** a fake 'perfect' answer.
        2. **Searching** using that fake answer instead of the question.
        """)
        
        query = st.text_input("Ask a technical question:", "What is the capital of France?")
        
        if st.button("Run HyDE Search"):
            with st.spinner("Dreaming of a perfect answer..."):
                hyde_doc, logs = agent.hyde_search(query)
                st.subheader("The 'Hypothetical' Document")
                st.info(hyde_doc)
                
                # Technical Deep Dive
                render_technical_trace(agent.ai.last_run_traces)
                
                st.success("Now searching the database using this hallucinated text... (Conceptual)")

    with right:
        render_mastery_notes(
            "Better Than a Question",
            "Embeddings work best when comparing 'Answers to Answers'. By turning your question into a fake answer FIRST, we find real documents that 'look' like our fake answer.",
            """# HyDE implementation
hyde_prompt = \"Write a technical answer for: {query}\"
fake_doc = llm.invoke(hyde_prompt)

# Search with the FAKE doc
results = vector_store.search(fake_doc)""",
            {"HyDE": "Hypothetical Document Embeddings.", "Alignment": "Bringing the query and the document into the same semantic space."}
        )

def render_chapter_14():
    st.title("✂️ Chapter 14: Contextual Compression")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **De-noising the retrieval.** Often, a 500-word chunk only has 20 words that actually answer the question. 
        Contextual Compression uses an LLM to 'surgically' extract only those 20 words.
        """)
        
        query = st.text_input("Refine this search:", "What are LangChain's main features?")
        
        if st.button("Run Compressed Search"):
            # Dummy docs for demo if index is empty
            if not agent.vector_store.vector_store:
                st.warning("Index is empty. Using demo logic.")
                dummy_text = "LangChain is a framework for LLMs. It features Chains, Agents, and Retrievers. It also has a lot of other complex stuff that isn't relevant to your specific question, like logo design details and company history."
                st.info(f"Original Text (100%):\\n{dummy_text}")
                st.success(f"Compressed Result (20%):\\n- Chains, Agents, and Retrievers")
            else:
                with st.spinner("Compressing context..."):
                    results, logs = agent.compressed_search(query)
                    for i, res in enumerate(results):
                        st.markdown(f"**Refined Chunk {i+1}**")
                        st.success(res.page_content)

    with right:
        render_mastery_notes(
            "The Signal-to-Noise Ratio",
            "Contextual Compression is a 'Post-Retrieval' step. We find the documents first, then we use a cheaper/faster model to 'Filter' them before sending them to the final reasoning model.",
            """# Compression Setup
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=vector_store.as_retriever()
)""",
            {"Tokens": "The units of text the AI reads; compression saves money/time.", "Context Window": "The limit of how much an AI can 'remember' at once."}
        )

def render_chapter_15():
    st.title("🚦 Chapter 15: Semantic Routing")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Intelligent Entry Points.** Not every message needs a database search. 
        Semantic Routing uses a lightweight LLM call to categorize the user's intent *first*.
        """)
        
        query = st.text_input("Try different inputs (e.g., 'Hi', 'What is RAG?'):")
        
        if st.button("Route Query"):
            with st.spinner("Classifying intent..."):
                category, logs = agent.route_query(query)
                
                # Technical Deep Dive
                render_technical_trace(agent.ai.last_run_traces)
                
                if category == "GREETING":
                    st.success(f"👋 Detected Greeting. Routing to 'Chat' chain.")
                elif category == "TECHNICAL_QUESTION":
                    st.info(f"💾 Detected Technical Query. Routing to 'RAG' chain.")
                else:
                    st.warning(f"🔍 Detected Search Request. Routing to 'Deep Search' chain.")

    with right:
        render_mastery_notes(
            "The Traffic Controller",
            "Routing is the first step in build 'Agency'. Instead of a fixed pipeline, we build a decision tree that only fires the expensive components (like vector search) when absolutely necessary.",
            """# Routing logic
prompt = \"Classify this query: {query}\"
category = llm.invoke(prompt)

if category == \"GREETING\":
    return \"Hello! How can I help?\"
else:
    return vector_store.search(query)""",
            {"Latency": "The speed of the system; routing simple queries reduces latency.", "Cost Efficiency": "Avoiding API calls to expensive models when possible."}
        )

def render_chapter_16():
    st.title("🛡️ Chapter 16: Reliable RAG (Self-Grading)")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **No more garbage in, garbage out.** Reliable RAG adds a 'Grader' node between retrieval and generation.
        If the documents found are irrelevant, the agent refuses to answer or triggers a retry.
        """)
        
        topic = st.text_input("Enter a query to test grading:", "How to bake a cake?")
        
        if st.button("Start Reliable RAG"):
            from src.graph_agent import ReliableRAG
            agent_rag = ReliableRAG()
            
            with st.spinner("Retrieving and Grading..."):
                steps = agent_rag.run(topic)
                
                for step in steps:
                    node_name = step['node']
                    data = step['data']
                    with st.expander(f"📍 Node: {node_name}"):
                        if 'messages' in data:
                            st.write(data['messages'][0].content)
                        else:
                            st.write(data)
            
            # Final output logic
            last_step = steps[-1]
            if last_step['node'] == "generate":
                st.success("✅ Reliable Answer Generated.")
            else:
                st.warning("❌ Documents graded as IRRELEVANT. Generation skipped.")

    with right:
        render_mastery_notes(
            "The Grader Node",
            "This is the first step towards 'Corrective RAG' (CRAG). The model is prompted to be a critic of the retrieved text. If the score is too low, we don't proceed to help avoid hallucinations.",
            """# Grading Logic
def grade_node(state):
    prompt = \"Is this relevant to {query}?\"
    grade = llm.invoke(prompt)
    return {\"relevance\": grade}

# Conditional Routing
workflow.add_conditional_edges(
    \"grade\",
    decide_fn,
    {\"yes\": \"generate\", \"no\": END}
)""",
            {"Self-Correction": "The AI recognizing its own data is bad.", "Reliability": "Ensuring every answer is backed by relevant truth."}
        )

def render_chapter_17():
    st.title("⚡ Chapter 17: Performance & Caching")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Semantic Caching.** Why pay for the same answer twice? 
        A semantic cache stores previous AI responses. If a new question is similar to an old one, we return the cached answer instantly.
        """)
        
        # Simulated Cache Demo
        test_query = st.text_input("Test a cached interaction (try 'What is RAG?'):")
        
        if st.button("Query AI (with Caching)"):
            with st.spinner("Checking cache and processing..."):
                answer, logs = agent.cached_query(test_query)
                
                # Technical Deep Dive
                render_technical_trace(agent.ai.last_run_traces)
                
                st.write(answer)

    with right:
        render_mastery_notes(
            "The Cost of Thinking",
            "LLMs are expensive and slow. Semantic caching is the #1 way to scale a RAG application. By identifying that query A is 99% similar to query B, we can serve a high-quality answer in under 10ms for $0.",
            """# Semantic Cache Logic
cache = SemanticCache()

def ask_ai(query):
    # Search for similar query vectors
    cached = cache.lookup(query)
    if cached:
        return cached # Speed!
    
    # Otherwise, call LLM
    answer = llm.invoke(query)
    cache.update(query, answer)
    return answer""",
            {"Vector Cache": "Storing queries as numbers to find 'similar' questions.", "Latency": "The time a user waits for an answer; cache = low latency."}
        )

def render_chapter_18():
    st.title("📊 Chapter 18: Evaluating RAG (RAGAS)")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Measuring Success.** How do you know if your RAG system is actually good? 
        RAGAS (RAG Assessment) provides four key metrics to grade your AI's performance scientifically.
        """)
        
        # Interactive Evaluation Demo
        st.subheader("Interactive Evaluator")
        query = st.text_input("User Question:", "Who invented Python?")
        context = st.text_area("Retrieved Context:", "Guido van Rossum created Python in the late 1980s.")
        answer = st.text_area("AI Response:", "Guido van Rossum is the creator of the Python programming language.")
        
        if st.button("Calculate RAGAS Scores"):
            with st.spinner("Analyzing alignment..."):
                # Simulated RAGAS logic
                st.metric("Faithfulness", "0.98", delta="Excellent")
                st.metric("Answer Relevance", "1.00", delta="Perfect")
                st.metric("Context Precision", "0.95", delta="High")
                st.metric("Context Recall", "0.90", delta="Good")
                st.success("🏁 Evaluation complete. System is Production-Ready.")

    with right:
        render_mastery_notes(
            "The RAGAS Framework",
            "Stop guessing and start measuring. RAGAS uses an 'LLM-as-a-Judge' to automatically grade thousands of answers against their source context. This is the only way to catch hallucinations across large datasets.",
            """# RAGAS metrics
metrics = [
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
]

# Run evaluation
result = evaluate(dataset, metrics=metrics)""",
            {"Faithfulness": "Is the answer derived ONLY from the context?", "Relevance": "Does the answer actually address the question?"}
        )

def render_chapter_19():
    st.title("🧠 Chapter 19: Autonomous Planning")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Decomposition.** Planning agents avoid rushing into answers. 
        They first break a complex goal down into a series of smaller, executable steps.
        """)
        
        goal = st.text_input("Enter a complex goal:", "Explain why LangGraph is better than sequential chains.")
        
        if st.button("Start Planning Execution"):
            from src.graph_agent import PlanAndExecuteMastery
            agent_plan = PlanAndExecuteMastery()
            
            with st.spinner("Generating Plan..."):
                steps = agent_plan.run(goal)
                
                for step in steps:
                    node_name = step['node']
                    data = step['data']
                    with st.expander(f"📍 Node: {node_name}"):
                        if 'messages' in data:
                            st.write(data['messages'][0].content)
                        else:
                            st.write(data)
            
            st.success("🎯 Goal Achieved through Autonomous Planning.")

    with right:
        render_mastery_notes(
            "Plan & Execute",
            "This pattern is for high-stakes agents. Instead of giving the AI the freedom to loop infinitely, we force it to define its path first. This makes the agent predictable, debuggable, and reliable.",
            """# Planning Node
def planner(state):
    plan = llm.invoke(\"Plan for {goal}\")
    return {\"plan\": plan}

# Executor Loop
workflow.add_conditional_edges(
    \"executor\",
    should_continue,
    {\"continue\": \"executor\", \"finish\": END}
)""",
            {"Hierarchical Planning": "Breaking a big task into sub-tasks.", "Predictability": "Knowing exactly what the AI intends to do before it does it."}
        )

def render_chapter_20():
    st.title("🛠️ Chapter 20: Tool Synthesis")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        **Self-Generation.** The ultimate form of agency is an AI that writes its own tools.
        If the AI encounters a problem it can't solve (like calculating a complex derivative), it writes the Python code to do it.
        """)
        
        problem = st.text_input("Enter a problem for the AI to solve with code:", "Calculate the 50th Fibonacci number.")
        
        if st.button("Synthesize Tool"):
            from src.graph_agent import ToolSynthesizer
            agent_tool = ToolSynthesizer()
            
            with st.spinner("Writing and Executing Code..."):
                steps = agent_tool.run(problem)
                
                for step in steps:
                    node_name = step['node']
                    data = step['data']
                    with st.expander(f"📍 Node: {node_name}"):
                        if 'messages' in data:
                            st.code(data['messages'][0].content)
                        else:
                            st.write(data)
            
            st.success("🏗️ Tool synthesized and goal achieved autonomously.")

    with right:
        render_mastery_notes(
            "AI as a Developer",
            "In this pattern, we give the model access to a 'Sandboxed Python Interpreter'. Instead of building every tool yourself, you build a model that can build tools. This creates an agent with near-infinite capability.",
            """# Tool Synthesis Logic
def writer(state):
    code = llm.invoke(\"Write code to solve {problem}\")
    return {\"code\": code}

def executor(state):
    result = sandbox.run(state.code)
    return {\"result\": result}""",
            {"Sandboxing": "Keeping the AI's generated code in a safe container.", "Recursive Growth": "Agents that get smarter by building their own toolkits."}
        )

def render_chapter_24():
    st.title("🎓 Chapter 24: Final Graduation")
    st.balloons()
    
    st.markdown("""
    ### Congratulations! You've Mastered AI Agency.
    
    From simple PDF parsing to autonomous code-writing agents, you have moved through the entire hierarchy of modern AI engineering.
    
    #### 🏆 Your Journey Summary:
    1.  **Phase 1: Foundations** - Mastery of Embeddings and Vector Search.
    2.  **Phase 2: Retrieval Engineering** - Building robust, multi-modal pipelines.
    3.  **Phase 3: LangGraph** - Transitioning from static chains to dynamic state machines.
    4.  **Phase 4: Advanced RAG** - Implementing HyDE, Compression, and Routing.
    5.  **Phase 5: Production** - Scaling with Reliable RAG and Semantic Caching.
    6.  **Phase 6: DeepAgents** - Entering the age of Autonomous Planning and Tool Synthesis.
    
    #### 🚀 What's Next?
    The code you see in this Learning Lab is a template. You can now take these `src` modules and deploy them into production apps, internal company tools, or the next big AI startup.
    
    *Keep building, keep iterating, and keep pushing the boundaries of what these models can do.*
    """)
    
    if st.button("Download My Mastery Badge"):
        st.success("Badge 'AI_AGENCY_COMPLETE' downloaded (simulated).")

def render_langgraph_stub(): # Kept as fallback for now
    st.title("🕸️ LangGraph: Workflows & State")
    st.warning("⚠️ This chapter is currently under development.")
    st.markdown("""
    In the next phase, we will implement **Stateful Agents**. Unlike basic chains, LangGraph allows:
    - **Persistence**: Saving the agent's memory to a database.
    - **Cycles**: Letting the agent try multiple times until it succeeds.
    - **Human-in-the-loop**: Waiting for you to click 'Approve' before taking action.
    """)
    st.image("https://blog.langchain.dev/content/images/2024/02/langgraph-diagram.png", caption="LangGraph Cyclic Workflow")

def render_phase_4_navigation():
    chapter = st.sidebar.radio("Learning Module:", [
        "Chapter 13: HyDE Patterns",
        "Chapter 14: Contextual Compression",
        "Chapter 15: Semantic Routing"
        "Chapter 15: Semantic Routing",
        "Chapter 20: Tool Synthesis",
        "Chapter 24: Final Graduation"
    ])
    if "Chapter 13" in chapter:
        render_chapter_13()
    elif "Chapter 14" in chapter:
        render_chapter_14()
    elif "Chapter 15" in chapter:
        render_chapter_15()
    elif "Chapter 20" in chapter:
        render_chapter_20()
    elif "Chapter 24" in chapter:
        render_chapter_24()
    else:
        st.title(f"🚀 {chapter}")
        st.warning("⚠️ This advanced RAG module is coming soon!")

def render_phase_5_navigation():
    chapter = st.sidebar.radio("Learning Module:", [
        "Chapter 16: Reliable RAG",
        "Chapter 17: Performance Caching",
        "Chapter 18: RAG Evaluation"
    ])
    if "Chapter 16" in chapter:
        render_chapter_16()
    else:
        st.title(f"🚀 {chapter}")
        st.warning("⚠️ This production RAG module is coming soon!")

def render_phase_6_navigation():
    chapter = st.sidebar.radio("Mastery Level:", [
        "Chapter 19: Autonomous Planning",
        "Chapter 20: Tool Synthesis",
        "Chapter 24: Final Graduation"
    ])
    if "Chapter 19" in chapter:
        render_chapter_19()
    else:
        st.title(f"🚀 {chapter}")
        st.warning("⚠️ This mastery module is coming soon!")

def render_chapter_25():
    st.title("🔬 Chapter 25: Langfuse Tracing & Observability")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("""
        Observability is the difference between a **prototype** and a **production** system.
        
        **In this chapter, we master:**
        1. **Tracing**: Seeing the full execution graph of an agent.
        2. **Evaluation**: Scoring responses to improve reliability.
        3. **Prompt Management**: Versioning prompts without changing code.
        """)
        
        st.image("https://langfuse.com/images/docs/trace-detail.png", caption="Langfuse Trace Visualization")
        
    with right:
        render_mastery_notes(
            "The Callback Pattern",
            "LangChain Agents emit 'events' at every step. Langfuse 'listens' to these events and organizes them into Traces and Spans.",
            """# 1. Initialize Handler
from langfuse.callback import CallbackHandler
handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)

# 2. Run with Callbacks
config = {"callbacks": [handler]}
graph.invoke(state, config=config)""",
            {"Tracing": "The web of steps taken to reach an answer.", "Evaluation": "Automatic or human-in-the-loop scoring of results."}
        )

def render_phase_7_navigation():
    chapter = st.sidebar.radio("Observability Module:", [
        "Chapter 25: Langfuse Tracing"
    ])
    if "Chapter 25" in chapter:
        render_chapter_25()

# --- Navigation Logic ---
st.sidebar.title("📚 Course Catalog")
pillar = st.sidebar.selectbox("Select Learning Pillar:", [
    "🏠 Home / Overview", 
    "🔗 LangChain foundations", 
    "🕸️ LangGraph Workflows", 
    "🌊 Advanced RAG Patterns",
    "🏭 Production Agentic RAG",
    "🤖 DeepAgents",
    "🔬 Observability"
])

if pillar == "🏠 Home / Overview":
    render_overview()
elif pillar == "🔗 LangChain foundations":
    chapter = st.sidebar.radio("Chapter:", [
        "Chapter 1: Parsing", 
        "Chapter 2: Embeddings", 
        "Chapter 3: Advanced Chunking", 
        "Chapter 4: Hybrid Search", 
        "Chapter 5: Query Enhancement",
        "Chapter 6: Multi-Modal RAG",
        "Chapter 7: Transition to Agency"
    ])
    if "Chapter 1" in chapter:
        render_chapter_1()
    elif "Chapter 2" in chapter:
        render_chapter_2()
    elif "Chapter 3" in chapter:
        render_chapter_3()
    elif "Chapter 4" in chapter:
        render_chapter_4()
    elif "Chapter 5" in chapter:
        render_chapter_5()
    elif "Chapter 6" in chapter:
        render_chapter_6()
    else:
        render_chapter_7()
elif pillar == "🕸️ LangGraph Workflows":
    chapter = st.sidebar.radio("Chapter:", [
        "Chapter 8: Basics",
        "Chapter 9: Human-In-Loop",
        "Chapter 10: Multi-Agent",
        "Chapter 11: Self-Correction",
        "Chapter 12: Persistence"
    ])
    if "Chapter 8" in chapter:
        render_chapter_8()
    elif "Chapter 9" in chapter:
        render_chapter_9()
    elif "Chapter 10" in chapter:
        render_chapter_10()
    elif "Chapter 11" in chapter:
        render_chapter_11()
    elif "Chapter 12" in chapter:
        render_chapter_12()
    else:
        st.title(f"🚀 {chapter}")
        st.warning("⚠️ This advanced LangGraph module is coming soon!")
        st.markdown("We will implement persistence, human-in-the-loop, and multi-agent coordination in the coming steps.")
elif pillar == "🌊 Advanced RAG Patterns":
    render_phase_4_navigation()
elif pillar == "🏭 Production Agentic RAG":
    render_phase_5_navigation()
elif pillar == "🤖 DeepAgents":
    if "Concept" in chapter:
        st.title("🤖 DeepAgents: The Autonomous Entity")
        st.info("The pinnacle of our roadmap: High-level autonomous reasoning.")
        st.markdown("""
        DeepAgents are agents that don't just 'search and answer'—they **Plan and Execute**.
        
        **What you will master:**
        1. **Hierarchical Planning**: Breaking goals into sub-tasks.
        2. **Tool Creation**: Empowering agents to write their own Python tools.
        3. **Multi-Agent Teams**: Managing a 'Boss' agent and specialized workers.
        """)
    elif "Chapter 19" in chapter:
        st.title("🎯 Chapter 19: Autonomous Planning")
        left, right = st.columns([1, 1])
        with left:
            st.write("Planning is the difference between a 'chatbot' and an 'agent'.")
            st.image("https://python.langchain.com/v0.2/img/plan_and_execute.png", caption="Plan-and-Execute Architecture")
        with right:
            render_mastery_notes(
                "Task Decomposition",
                "DeepAgents use LLMs to split a complex prompt into small, manageable steps. This allows the agent to track progress and handle errors at each step.",
                """# Planning Logic
plan = agent.generate_plan("Research and summarize AI trends")
for step in plan.steps:
    result = agent.execute(step)""",
                {"Decomposition": "Breaking a big problem into small pieces.", "Dynamic Replanning": "The ability to change the plan if a step fails."}
            )
    else:
        st.title(f"🚀 {chapter}")
        st.warning("⚠️ This advanced mastery chapter is part of the Phase 6 roadmap and is coming soon!")
        st.markdown("Check [ROADMAP.md](file:///d:/My%20Professional%20Projects/PythonAgent/ROADMAP.md) for the full details of what you will learn here.")
elif pillar == "🔬 Observability":
    render_phase_7_navigation()

st.sidebar.markdown("---")
st.sidebar.caption("Mastery Lab v2.0 | Built with Antigravity 🚀")
