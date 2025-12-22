"""
Streamlit UI for LLM Router.

Interactive web interface for testing and visualizing the cost-aware LLM routing system.
"""

import streamlit as st
import sys
from pathlib import Path
import os
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from routing.router import LLMRouter
from routing.difficulty import QueryDifficultyEstimator
from llm.local import LocalLLM
from llm.openai_llm import OpenAILLM
from utils.metrics import MetricsLogger


# Page configuration
st.set_page_config(
    page_title="LLM Router - Cost-Aware Routing System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_router():
    """Initialize router components (cached for performance)."""
    try:
        # Initialize difficulty estimator
        difficulty_estimator = QueryDifficultyEstimator()
        
        # Load local LLM
        model_path = Path(__file__).parent / "models" / "phi-2.Q4_K_M.gguf"
        if not model_path.exists():
            return None, None, "Model not found. Please download phi-2.Q4_K_M.gguf to the models/ directory."
        
        local_llm = LocalLLM(str(model_path))
        
        # Initialize remote LLM (if API key is available)
        api_key = os.getenv("OPENAI_API_KEY")
        remote_llm = None
        if api_key:
            try:
                remote_llm = OpenAILLM(api_key=api_key, model="gpt-4o")
            except Exception as e:
                pass  # Will show warning in sidebar
        
        # Initialize router
        router = LLMRouter(
            difficulty_estimator=difficulty_estimator,
            local_llm=local_llm,
            remote_llm=remote_llm
        )
        
        return router, difficulty_estimator, None
    except Exception as e:
        return None, None, str(e)


def format_difficulty_score(score: float) -> tuple:
    """Format difficulty score with color and emoji."""
    if score < 0.3:
        return "🟢 Easy", "#28a745", score
    elif score < 0.6:
        return "🟡 Medium", "#ffc107", score
    else:
        return "🔴 Hard", "#dc3545", score


def format_routing_decision(decision: str) -> tuple:
    """Format routing decision with color and icon."""
    if decision == "local":
        return "🟢 Local (GGUF)", "#28a745"
    elif decision == "escalated":
        return "🟡 Escalated (Local → GPT-4o)", "#ffc107"
    else:
        return "🔴 Remote (GPT-4o)", "#dc3545"


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🚀 LLM Router</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        Cost-Aware LLM Routing with Local Inference & GPT-4o Escalation
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize router (cached)
    router, difficulty_estimator, error = initialize_router()
    
    if router is None:
        st.error(f"❌ Initialization Error: {error}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System status
        st.subheader("System Status")
        st.success("✅ Router Initialized")
        st.info("✅ Local Model Loaded")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and router.remote_llm is not None:
            st.success("✅ OpenAI API Connected")
        else:
            st.warning("⚠️ OpenAI API Key not set")
            st.info("Set OPENAI_API_KEY environment variable to enable GPT-4o")
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        show_details = st.checkbox("Show Detailed Analysis", value=True)
        show_response = st.checkbox("Show Full Response", value=True)
        
        st.divider()
        
        # Info
        st.subheader("ℹ️ About")
        st.info("""
        This system routes queries based on difficulty:
        - **Easy** (< 0.3) → Local GGUF
        - **Medium** (0.3-0.6) → Local + Escalate if needed
        - **Hard** (≥ 0.6) → GPT-4o
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Your Query")
        query = st.text_area(
            "Query",
            height=100,
            placeholder="Enter your question or prompt here...",
            help="The system will automatically route this to the appropriate model based on difficulty."
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            submit_button = st.button("🚀 Route Query", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        st.header("📊 Quick Stats")
        if 'total_queries' not in st.session_state:
            st.session_state.total_queries = 0
            st.session_state.total_cost = 0.0
            st.session_state.total_saved = 0.0
        
        st.metric("Total Queries", st.session_state.total_queries)
        st.metric("Total Cost", f"${st.session_state.total_cost:.6f}")
        st.metric("Total Saved", f"${st.session_state.total_saved:.6f}")
    
    # Process query
    if submit_button and query:
        with st.spinner("Routing query and generating response..."):
            try:
                # Route query
                start_time = time.time()
                result = router.route(query)
                end_time = time.time()
                
                # Update session state
                st.session_state.total_queries += 1
                st.session_state.total_cost += result.get("cost_usd", 0.0)
                st.session_state.total_saved += result.get("cost_saved_usd", 0.0)
                
                # Display results
                st.divider()
                st.header("📊 Routing Analysis")
                
                # Difficulty and Routing Decision
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    difficulty = result.get("difficulty", 0.0)
                    label, color, score = format_difficulty_score(difficulty)
                    st.metric("Difficulty Score", f"{score:.3f}", label)
                
                with col2:
                    decision = result.get("routing_decision", "unknown")
                    decision_label, decision_color = format_routing_decision(decision)
                    st.metric("Routing Decision", decision_label)
                
                with col3:
                    latency = result.get("latency_ms", 0.0)
                    st.metric("Latency", f"{latency:.2f} ms", f"{latency/1000:.2f}s")
                
                # Detailed Analysis
                if show_details:
                    st.subheader("🔍 Detailed Analysis")
                    
                    # Difficulty breakdown
                    with st.expander("📈 Difficulty Breakdown", expanded=True):
                        if difficulty_estimator:
                            length_score = difficulty_estimator._length_score(query)
                            keyword_score = difficulty_estimator._keyword_score(query)
                            structure_score = difficulty_estimator._structure_score(query)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("**Length Score**")
                                st.metric("", f"{length_score:.3f}")
                                st.progress(min(length_score, 1.0))
                            with col2:
                                st.write("**Keyword Score**")
                                st.metric("", f"{keyword_score:.3f}")
                                st.progress(min(keyword_score, 1.0))
                            with col3:
                                st.write("**Structure Score**")
                                st.metric("", f"{structure_score:.3f}")
                                st.progress(min(structure_score, 1.0))
                            
                            # Show weight explanation
                            st.caption("Final score: 25% length + 50% keyword + 25% structure")
                        else:
                            st.info("Difficulty breakdown not available")
                    
                    # Cost Analysis
                    with st.expander("💰 Cost Analysis", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Cost", f"${result.get('cost_usd', 0.0):.6f}")
                        with col2:
                            st.metric("Cost Saved", f"${result.get('cost_saved_usd', 0.0):.6f}")
                        with col3:
                            st.metric("Input Tokens", result.get("input_tokens", 0))
                        with col4:
                            st.metric("Output Tokens", result.get("output_tokens", 0))
                    
                    # Model Information
                    with st.expander("🤖 Model Information"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Model:** {result.get('model', 'unknown')}")
                            st.write(f"**Device:** {result.get('device', 'unknown')}")
                        with col2:
                            total_tokens = result.get("input_tokens", 0) + result.get("output_tokens", 0)
                            st.write(f"**Total Tokens:** {total_tokens}")
                            st.write(f"**Processing Time:** {end_time - start_time:.2f}s")
                
                # Response
                st.divider()
                st.header("💬 Response")
                
                if show_response:
                    st.text_area(
                        "Generated Response",
                        value=result.get("text", ""),
                        height=200,
                        disabled=True
                    )
                else:
                    st.text_area(
                        "Generated Response (truncated)",
                        value=result.get("text", "")[:200] + "...",
                        height=100,
                        disabled=True
                    )
                
                # Visual indicators
                st.divider()
                st.subheader("📈 Performance Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if result.get("routing_decision") == "local":
                        st.success("✅ Local Model")
                        st.caption("Zero Cost")
                    elif result.get("routing_decision") == "escalated":
                        st.warning("⚠️ Escalated")
                        st.caption("Low Confidence")
                    else:
                        st.info("🔴 GPT-4o")
                        st.caption("Hard Query")
                
                with col2:
                    cost = result.get("cost_usd", 0.0)
                    st.metric("Cost", f"${cost:.6f}")
                
                with col3:
                    saved = result.get("cost_saved_usd", 0.0)
                    if saved > 0:
                        st.metric("Saved", f"${saved:.6f}", delta=f"${saved:.6f}")
                    else:
                        st.metric("Saved", "$0.000000")
                
                with col4:
                    if latency < 2000:
                        st.success("⚡ Fast")
                        st.caption(f"{latency/1000:.2f}s")
                    elif latency < 5000:
                        st.warning("⏱️ Moderate")
                        st.caption(f"{latency/1000:.2f}s")
                    else:
                        st.info("🐌 Slower")
                        st.caption(f"{latency/1000:.2f}s")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    elif submit_button and not query:
        st.warning("⚠️ Please enter a query first!")
    
    if clear_button:
        st.session_state.total_queries = 0
        st.session_state.total_cost = 0.0
        st.session_state.total_saved = 0.0
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using Streamlit | Cost-Aware LLM Routing System</p>
        <p>Reduces expensive LLM calls by ~65–70% while preserving answer quality</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

