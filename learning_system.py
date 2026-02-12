"""
🎓 LangChain, LangGraph & DeepAgents - Master Learning System

A comprehensive, interactive learning platform for Agentic AI and RAG systems.
Designed for Python beginners to become AI engineering masters.

Structure:
- 3 Main Tabs: LangChain | LangGraph | DeepAgent
- Step-by-step chapters within each tab
- Right-side code explorer with detailed explanations
- Interactive code examples
"""

import streamlit as st
import os
import sys
from typing import Dict, List, Any, Callable

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.curriculum.langchain_chapters import LANGCHAIN_CURRICULUM
from src.curriculum.langgraph_chapters import LANGGRAPH_CURRICULUM
from src.curriculum.deepagent_chapters import DEEPAGENT_CURRICULUM
from src.curriculum.code_explorer import render_code_explorer, render_concept_card

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🎓 Agentic AI Mastery",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Premium Learning Interface
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1e2130;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        color: #a0aec0;
        border-radius: 8px;
        padding: 15px 30px;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4a5568;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Chapter cards */
    .chapter-card {
        background: #1e2130;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .chapter-card:hover {
        border-color: #667eea;
        transform: translateX(5px);
    }
    
    /* Code explorer panel */
    .code-panel {
        background: #0d1117;
        border-left: 3px solid #667eea;
        border-radius: 0 10px 10px 0;
    }
    
    /* Concept tags */
    .concept-tag {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 2px;
    }
    
    /* Progress indicator */
    .progress-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .progress-completed { background: #48bb78; }
    .progress-inprogress { background: #ed8936; }
    .progress-pending { background: #718096; }
    
    /* Info boxes */
    .info-box {
        background: #2d3748;
        border-left: 4px solid #4299e1;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #2d3748;
        border-left: 4px solid #ed8936;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    .success-box {
        background: #2d3748;
        border-left: 4px solid #48bb78;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Code annotation */
    .code-annotation {
        background: #1a202c;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Monaco', 'Menlo', monospace;
    }
    
    .annotation-line {
        color: #a0aec0;
        font-size: 13px;
        line-height: 1.8;
    }
    
    .annotation-highlight {
        color: #fbd38d;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .sidebar-chapter {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .sidebar-chapter:hover {
        background: #2d3748;
    }
    
    .sidebar-chapter.active {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'current_tab': 'langchain',
        'langchain_chapter': 0,
        'langgraph_chapter': 0,
        'deepagent_chapter': 0,
        'completed_chapters': set(),
        'code_explorer_open': True,
        'selected_code_example': None,
        'notes': {},
        'practice_code': {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
def render_sidebar():
    """Render the sidebar with progress tracking and chapter navigation."""
    with st.sidebar:
        st.image("https://python.langchain.com/img/brand/wordmark.png", width=200)
        st.markdown("---")
        
        # Progress Overview
        st.subheader("📊 Your Progress")
        total_chapters = (
            len(LANGCHAIN_CURRICULUM) + 
            len(LANGGRAPH_CURRICULUM) + 
            len(DEEPAGENT_CURRICULUM)
        )
        completed = len(st.session_state.completed_chapters)
        progress = completed / total_chapters if total_chapters > 0 else 0
        
        st.progress(progress)
        st.caption(f"{completed}/{total_chapters} chapters completed")
        
        st.markdown("---")
        
        # Quick Stats
        st.subheader("🎯 Learning Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LangChain", f"{min(st.session_state.langchain_chapter + 1, len(LANGCHAIN_CURRICULUM))}/{len(LANGCHAIN_CURRICULUM)}")
        with col2:
            st.metric("LangGraph", f"{min(st.session_state.langgraph_chapter + 1, len(LANGGRAPH_CURRICULUM))}/{len(LANGGRAPH_CURRICULUM)}")
        with col3:
            st.metric("DeepAgent", f"{min(st.session_state.deepagent_chapter + 1, len(DEEPAGENT_CURRICULUM))}/{len(DEEPAGENT_CURRICULUM)}")
        
        st.markdown("---")
        
        # Current Chapter Navigation
        st.subheader("📚 Current Track")
        
        current_tab = st.session_state.current_tab
        if current_tab == 'langchain':
            current_chapter = st.session_state.langchain_chapter
            curriculum = LANGCHAIN_CURRICULUM
            emoji = "🔗"
        elif current_tab == 'langgraph':
            current_chapter = st.session_state.langgraph_chapter
            curriculum = LANGGRAPH_CURRICULUM
            emoji = "🕸️"
        else:
            current_chapter = st.session_state.deepagent_chapter
            curriculum = DEEPAGENT_CURRICULUM
            emoji = "🤖"
        
        for i, chapter in enumerate(curriculum):
            status = "completed" if f"{current_tab}_{i}" in st.session_state.completed_chapters else \
                     "inprogress" if i == current_chapter else "pending"
            
            status_emoji = {"completed": "✅", "inprogress": "▶️", "pending": "⭕"}[status]
            
            if st.button(f"{status_emoji} {chapter['title']}", key=f"nav_{current_tab}_{i}"):
                if current_tab == 'langchain':
                    st.session_state.langchain_chapter = i
                elif current_tab == 'langgraph':
                    st.session_state.langgraph_chapter = i
                else:
                    st.session_state.deepagent_chapter = i
                st.rerun()
        
        st.markdown("---")
        
        # Learning Resources
        with st.expander("📖 Additional Resources"):
            st.markdown("""
            - [LangChain Docs](https://python.langchain.com/)
            - [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
            - [DeepAgents Docs](https://docs.langchain.com/oss/python/deepagents/overview)
            - [OpenAI API](https://platform.openai.com/)
            """)


# ============================================================================
# MAIN CONTENT RENDERERS
# ============================================================================
def render_welcome():
    """Render the welcome/landing page."""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Welcome to Agentic AI Mastery</h1>
        <p>From Python Basics to AI Engineering Expert</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔗 LangChain
        **The Foundation of Agentic AI**
        
        Learn to build applications with LLMs through:
        - Document parsing & chunking
        - Vector embeddings & semantic search
        - Chains and retrieval systems
        - Tool integration
        
        *8 Chapters • Beginner Friendly*
        """)
        if st.button("Start LangChain Track", type="primary", use_container_width=True):
            st.session_state.current_tab = 'langchain'
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 🕸️ LangGraph
        **Stateful Agent Workflows**
        
        Master complex agent architectures:
        - Nodes, edges & state management
        - Multi-agent systems
        - Human-in-the-loop
        - Cyclical reasoning patterns
        
        *6 Chapters • Intermediate*
        """)
        if st.button("Start LangGraph Track", type="primary", use_container_width=True):
            st.session_state.current_tab = 'langgraph'
            st.rerun()
    
    with col3:
        st.markdown("""
        ### 🤖 DeepAgents
        **Production-Ready Agents**
        
        Build autonomous AI systems:
        - Task planning & decomposition
        - Subagent spawning
        - Long-term memory
        - MCP server integration
        
        *5 Chapters • Advanced*
        """)
        if st.button("Start DeepAgent Track", type="primary", use_container_width=True):
            st.session_state.current_tab = 'deepagent'
            st.rerun()
    
    st.markdown("---")
    
    # Learning Path Visualization
    st.subheader("🗺️ Your Learning Journey")
    
    journey_col1, journey_col2, journey_col3 = st.columns([1, 2, 1])
    
    with journey_col2:
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │  PHASE 1: FOUNDATIONS                                           │
        │  ├── LangChain Chapters 1-4 (Parsing → Embeddings → Chains)    │
        │  └── 🎯 Goal: Build basic RAG systems                          │
        ├─────────────────────────────────────────────────────────────────┤
        │  PHASE 2: AGENT ARCHITECTURE                                    │
        │  ├── LangChain Chapters 5-8 (Tools → Agents → Memory)          │
        │  └── LangGraph Chapters 1-3 (Basics → Workflows)               │
        │  └── 🎯 Goal: Build autonomous agents                          │
        ├─────────────────────────────────────────────────────────────────┤
        │  PHASE 3: ADVANCED SYSTEMS                                      │
        │  ├── LangGraph Chapters 4-6 (Multi-agent → Persistence)        │
        │  └── DeepAgent Chapters 1-5 (Planning → MCP → Production)      │
        │  └── 🎯 Goal: Production-grade AI systems                      │
        └─────────────────────────────────────────────────────────────────┘
        ```
        """)


def render_chapter_content(chapter: Dict[str, Any], tab_name: str, chapter_idx: int):
    """Render a single chapter with code explorer on the right."""
    
    # Main content area (left 65%)
    content_col, explorer_col = st.columns([0.65, 0.35])
    
    with content_col:
        # Chapter Header
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2>{chapter['emoji']} {chapter['title']}</h2>
            <p>{chapter.get('subtitle', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Learning Objectives
        with st.expander("🎯 Learning Objectives", expanded=True):
            for objective in chapter.get('objectives', []):
                st.markdown(f"- {objective}")
        
        # Main Content
        st.markdown(chapter.get('content', ''))
        
        # Interactive Demo if available
        if 'demo' in chapter and chapter['demo']:
            st.markdown("---")
            st.subheader("🧪 Interactive Practice")
            chapter['demo']()
        
        # Quiz if available
        if 'quiz' in chapter and chapter['quiz']:
            st.markdown("---")
            st.subheader("📝 Knowledge Check")
            render_quiz(chapter['quiz'], tab_name, chapter_idx)
        
        # Navigation buttons
        st.markdown("---")
        nav_cols = st.columns([1, 1, 1])
        
        with nav_cols[0]:
            if chapter_idx > 0:
                if st.button("⬅️ Previous Chapter", use_container_width=True):
                    if tab_name == 'langchain':
                        st.session_state.langchain_chapter = chapter_idx - 1
                    elif tab_name == 'langgraph':
                        st.session_state.langgraph_chapter = chapter_idx - 1
                    else:
                        st.session_state.deepagent_chapter = chapter_idx - 1
                    st.rerun()
        
        with nav_cols[1]:
            chapter_key = f"{tab_name}_{chapter_idx}"
            is_completed = chapter_key in st.session_state.completed_chapters
            
            if is_completed:
                st.success("✅ Completed!")
            else:
                if st.button("✓ Mark as Complete", type="primary", use_container_width=True):
                    st.session_state.completed_chapters.add(chapter_key)
                    st.rerun()
        
        with nav_cols[2]:
            curriculum = {
                'langchain': LANGCHAIN_CURRICULUM,
                'langgraph': LANGGRAPH_CURRICULUM,
                'deepagent': DEEPAGENT_CURRICULUM
            }[tab_name]
            
            if chapter_idx < len(curriculum) - 1:
                if st.button("Next Chapter ➡️", use_container_width=True):
                    if tab_name == 'langchain':
                        st.session_state.langchain_chapter = chapter_idx + 1
                    elif tab_name == 'langgraph':
                        st.session_state.langgraph_chapter = chapter_idx + 1
                    else:
                        st.session_state.deepagent_chapter = chapter_idx + 1
                    st.rerun()
    
    # Code Explorer (right 35%)
    with explorer_col:
        render_code_explorer(chapter.get('code_examples', []))


def render_quiz(quiz_data: List[Dict], tab_name: str, chapter_idx: int):
    """Render interactive quiz questions."""
    quiz_key = f"quiz_{tab_name}_{chapter_idx}"
    
    if quiz_key not in st.session_state:
        st.session_state[quiz_key] = {'answered': [], 'score': 0}
    
    for i, question in enumerate(quiz_data):
        q_key = f"{quiz_key}_q{i}"
        
        st.markdown(f"**Q{i+1}. {question['question']}**")
        
        if i in st.session_state[quiz_key]['answered']:
            # Show answer
            st.info(f"✓ {question['correct']}")
            st.caption(f"💡 {question.get('explanation', '')}")
        else:
            # Show options
            options = question['options']
            selected = st.radio(f"Select answer:", options, key=q_key, label_visibility="collapsed")
            
            if st.button("Submit Answer", key=f"submit_{q_key}"):
                if selected == question['correct']:
                    st.success("✅ Correct!")
                    st.session_state[quiz_key]['score'] += 1
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {question['correct']}")
                st.session_state[quiz_key]['answered'].append(i)
                st.rerun()
        
        st.markdown("---")


def render_track_tab(tab_name: str, curriculum: List[Dict], current_chapter_idx: int):
    """Render a track tab (LangChain, LangGraph, or DeepAgent)."""
    
    # Overview or specific chapter
    view_mode = st.radio(
        "View:",
        ["📋 Chapter Overview", "📖 Current Chapter"],
        horizontal=True,
        key=f"{tab_name}_view"
    )
    
    if view_mode == "📋 Chapter Overview":
        render_chapter_overview(tab_name, curriculum, current_chapter_idx)
    else:
        chapter = curriculum[current_chapter_idx]
        render_chapter_content(chapter, tab_name, current_chapter_idx)


def render_chapter_overview(tab_name: str, curriculum: List[Dict], current_chapter_idx: int):
    """Render the chapter overview grid."""
    
    # Track description
    descriptions = {
        'langchain': {
            'title': '🔗 LangChain Track',
            'desc': 'Master the foundation of LLM applications. Learn document processing, embeddings, chains, and agent tools.',
            'color': '#667eea'
        },
        'langgraph': {
            'title': '🕸️ LangGraph Track',
            'desc': 'Build stateful, multi-agent workflows. Understand cycles, persistence, and human-in-the-loop patterns.',
            'color': '#f093fb'
        },
        'deepagent': {
            'title': '🤖 DeepAgent Track',
            'desc': 'Create production-ready autonomous agents with planning, subagents, memory, and MCP integration.',
            'color': '#4ade80'
        }
    }
    
    track_info = descriptions[tab_name]
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {track_info['color']}22 0%, transparent 100%); 
                padding: 20px; border-radius: 10px; border-left: 5px solid {track_info['color']}; margin-bottom: 20px;">
        <h2>{track_info['title']}</h2>
        <p>{track_info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chapter grid
    cols = st.columns(2)
    
    for i, chapter in enumerate(curriculum):
        with cols[i % 2]:
            is_completed = f"{tab_name}_{i}" in st.session_state.completed_chapters
            is_current = i == current_chapter_idx
            
            status_badge = ""
            if is_completed:
                status_badge = "<span style='color: #48bb78;'>✅ Completed</span>"
            elif is_current:
                status_badge = "<span style='color: #ed8936;'>▶️ In Progress</span>"
            else:
                status_badge = "<span style='color: #718096;'>⭕ Pending</span>"
            
            with st.container():
                st.markdown(f"""
                <div style="background: #1e2130; border: 1px solid {'#667eea' if is_current else '#2d3748'}; 
                            border-radius: 12px; padding: 20px; margin: 10px 0;">
                    <h4>{chapter['emoji']} Chapter {i+1}: {chapter['title']}</h4>
                    <p style="color: #a0aec0; font-size: 14px;">{chapter.get('subtitle', '')}</p>
                    <p>{status_badge}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Start Chapter", key=f"start_{tab_name}_{i}", use_container_width=True):
                    if tab_name == 'langchain':
                        st.session_state.langchain_chapter = i
                    elif tab_name == 'langgraph':
                        st.session_state.langgraph_chapter = i
                    else:
                        st.session_state.deepagent_chapter = i
                    st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    
    # Main header
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h1>🎓 Agentic AI Mastery</h1>
        <p style="color: #a0aec0;">Master LangChain • LangGraph • DeepAgents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab_langchain, tab_langgraph, tab_deepagent = st.tabs([
        "🔗 LangChain",
        "🕸️ LangGraph", 
        "🤖 DeepAgent"
    ])
    
    with tab_langchain:
        st.session_state.current_tab = 'langchain'
        render_track_tab('langchain', LANGCHAIN_CURRICULUM, st.session_state.langchain_chapter)
    
    with tab_langgraph:
        st.session_state.current_tab = 'langgraph'
        render_track_tab('langgraph', LANGGRAPH_CURRICULUM, st.session_state.langgraph_chapter)
    
    with tab_deepagent:
        st.session_state.current_tab = 'deepagent'
        render_track_tab('deepagent', DEEPAGENT_CURRICULUM, st.session_state.deepagent_chapter)


if __name__ == "__main__":
    main()
