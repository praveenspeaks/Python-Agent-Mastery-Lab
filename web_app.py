import streamlit as st
import os
import time
from src.app import LangChainAgent
from dotenv import load_dotenv

# Page Config
st.set_page_config(
    page_title="LangChain Learning Lab - Chapter 1",
    page_icon="🎓",
    layout="wide"
)

# Load environment
load_dotenv()

# Custom CSS for a premium look
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; color: #fafafa; }
    .stAlert { border-radius: 10px; }
    .log-container { 
        background-color: #1e2130; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .step-header { color: #4CAF50; font-weight: bold; font-size: 1.2em; }
    .educational-tag { 
        background-color: #2e3148; 
        color: #ffcc00; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 0.8em; 
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title & Educational Welcome
st.title("🎓 LangChain & Agentic AI: Learning Lab")
st.markdown("""
### Chapter 1: The Ingestion Pipeline
Welcome, Developer! This interface isn't just a tool; it's a window into the **LangChain Ingestion Pipeline**. 
In this chapter, you will learn how raw data is transformed into structured AI context through **Parsing** and **Chunking**.
""")

# Sidebar settings
with st.sidebar:
    st.header("🛠️ Learning Parameters")
    st.info("Adjust these settings to see how the 'behind the scenes' logic changes.")
    chunk_size = st.slider("Chunk Size", 100, 2000, 1000, help="Total characters per chunk.")
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, help="How many characters repeat from the previous chunk for context.")
    
    st.divider()
    st.markdown("### 📚 Current Roadmap")
    st.write("✅ **01. Parsing & Chunking**")
    st.write("⏳ 02. Vector Embeddings")
    st.write("⏳ 03. Agentic Workflows")
    
    st.divider()
    if st.button("🗑️ Reset Learning Lab", help="Clear all uploaded data and logs."):
        st.rerun()

# Initialize Session State
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "ai_logs" not in st.session_state:
    st.session_state.ai_logs = []

# Initialize Orchestrator
agent = LangChainAgent(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Layout
col_main, col_logs = st.columns([3, 2])

with col_main:
    st.subheader("📤 Step 1: Input Document")
    uploaded_file = st.file_uploader("Upload a file to see it parsed in real-time", 
                                     type=["pdf", "txt", "docx", "csv", "xlsx", "json", "html", "xml", "md"])

    if uploaded_file:
        # Save temp
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Attached: `{uploaded_file.name}`")
        
        if st.button("🚀 Execute Ingestion Pipeline"):
            with st.spinner("Executing Pipeline..."):
                # Clear previous state
                st.session_state.summary = None
                st.session_state.ai_logs = []
                # Orchestrate
                st.session_state.chunks, st.session_state.logs = agent.process_file(file_path)
                st.rerun()

    # 3. RENDER RESULTS (Persists after reruns)
    if st.session_state.chunks:
        st.subheader(f"🧩 Step 2: Generated Chunks ({len(st.session_state.chunks)})")
        num_to_show = min(3, len(st.session_state.chunks))
        st.info(f"Ingestion successful! Here are the first {num_to_show} chunks:")
        
        for i in range(num_to_show):
            with st.expander(f"📦 Chunk #{i+1} Viewer"):
                st.markdown(f"**Metadata:** `{st.session_state.chunks[i].metadata}`")
                st.text_area("Chunk Content", st.session_state.chunks[i].page_content, height=150, key=f"chunk_{i}")
                
        st.subheader("🤖 Step 3: AI Analysis Pipeline")
        if st.button("Generate AI Content Summary"):
            if not os.getenv("OPENROUTER_API_KEY"):
                st.warning("API Key missing in .env")
            else:
                with st.spinner("Executing AI Model Chain..."):
                    summary, ai_traces = agent.ai.summarize_chunks(st.session_state.chunks)
                    st.session_state.ai_logs = ai_traces
                    st.session_state.summary = summary
                    st.rerun()
        
        if st.session_state.summary:
            st.markdown("---")
            st.markdown("### 📝 AI Generated Intelligence")
            st.success(st.session_state.summary)
            st.info("💡 **Lesson:** You've just seen the full lifecycle from raw file to AI-digested summary. Click any box on the right to study the code again!")

with col_logs:
    st.subheader("🕵️ Developer Console: Trace Explorer")
    
    if st.session_state.logs:
        st.info("Click on a step below to reveal the actual Python code running that step.")
        for trace in st.session_state.logs:
            with st.expander(f"🚀 {trace['message']}"):
                st.code(trace["code"], language="python")
        
        if st.session_state.ai_logs:
            st.markdown("---")
            st.subheader("🤖 AI Process Tracing")
            for trace in st.session_state.ai_logs:
                with st.expander(f"🤖 {trace['message']}"):
                    st.code(trace['code'], language="python")
    else:
        st.info("Upload a file to start the Trace Explorer.")
        st.markdown("""
        ### 👨‍🏫 What happens here?
        When you run the pipeline, this console will record every 'Trace'. 
        
        **Open any trace box** to see details.
        """)
