"""
Streamlit app for comparing RAG vs CRAG systems.
"""

import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    pass

import streamlit as st
import time
from typing import Dict, Any
import config
from rag_system import BasicRAGSystem
from crag_system import CRAGSystem


def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'crag_system' not in st.session_state:
        st.session_state.crag_system = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'doc_count' not in st.session_state:
        st.session_state.doc_count = 0


def check_api_keys():
    """Check if required API keys are configured."""
    missing_keys = []
    if not config.OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not config.OPENROUTER_API_KEY:
        missing_keys.append("OPENROUTER_API_KEY")
    if not config.TAVILY_API_KEY:
        missing_keys.append("TAVILY_API_KEY")
    
    return missing_keys


def load_documents(urls):
    """Load documents into both RAG and CRAG systems."""
    try:
        with st.spinner("Loading documents into RAG system..."):
            rag_system = BasicRAGSystem()
            rag_count = rag_system.load_documents(urls)
            st.session_state.rag_system = rag_system
        
        with st.spinner("Loading documents into CRAG system..."):
            crag_system = CRAGSystem()
            crag_count = crag_system.load_documents(urls)
            st.session_state.crag_system = crag_system
        
        st.session_state.documents_loaded = True
        st.session_state.doc_count = rag_count
        
        chunk_limit_note = f" (limited to {config.MAX_CHUNKS} max)" if rag_count >= config.MAX_CHUNKS else ""
        return True, f"Successfully loaded {rag_count} document chunks into both systems{chunk_limit_note}."
        
    except Exception as e:
        return False, f"Error loading documents: {e}"


def display_system_response(response: Dict[str, Any], system_name: str):
    """Display the response from a RAG/CRAG system."""
    st.subheader(f"{system_name} Response")
    
    # Display answer
    st.markdown("**Answer:**")
    st.write(response["answer"])
    
    # Display system information
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Documents Used", response["num_documents"])
    
    with col2:
        if system_name == "CRAG":
            if "initial_documents_count" in response:
                st.metric("Initial Documents", response["initial_documents_count"])
    
    # CRAG specific information
    if system_name == "CRAG" and "relevance_ratio" in response:
        st.markdown("**CRAG Analysis:**")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Relevance Ratio", f"{response['relevance_ratio']:.2f}")
        with col4:
            st.metric("Web Search", "Yes" if response.get("web_search_performed", False) else "No")
        
        if response.get("improved_question"):
            st.markdown("**Improved Question:**")
            st.info(response["improved_question"])
        
        if response.get("evaluation_details"):
            with st.expander("Document Evaluation Details"):
                for i, detail in enumerate(response["evaluation_details"]):
                    relevance_icon = "✅" if detail["relevant"] else "❌"
                    st.write(f"{relevance_icon} **Document {i+1}:** {detail['document_preview']}")
    
    # Retrieved documents
    if response["retrieved_documents"]:
        with st.expander(f"Retrieved Documents ({len(response['retrieved_documents'])})"):
            for i, doc in enumerate(response["retrieved_documents"]):
                st.markdown(f"**Document {i+1}:**")
                preview_length = config.Constants.ANSWER_PREVIEW_LENGTH
                st.text(doc.page_content[:preview_length] + "..." if len(doc.page_content) > preview_length else doc.page_content)
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                st.divider()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG vs CRAG Comparison",
        page_icon="🔍",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🔍 RAG vs CRAG Comparison")
    st.markdown("""
    This application demonstrates the differences between traditional **RAG (Retrieval-Augmented Generation)** 
    and **CRAG (Corrective Retrieval-Augmented Generation)** systems.
    
    - **RAG**: Retrieves documents and generates answers directly
    - **CRAG**: Evaluates document relevance, filters irrelevant content, and performs web search when needed
    """)
    
    # Demo limits and configuration info
    with st.expander("ℹ️ Demo Configuration", expanded=False):
        st.markdown(f"""
        **Resource limits for optimal demo performance:**
        - 📄 **Maximum URLs**: {config.MAX_URLS} 
        - 🔢 **Maximum chunks**: {config.MAX_CHUNKS}
        - ⚡ **Purpose**: Ensures fast loading and response times
        
        **Models used:**
        - 🤖 **Chat Model**: {config.CHAT_MODEL} (via OpenRouter)
        - 📊 **Embeddings**: {config.EMBEDDING_MODEL} (via OpenAI)
        - 🔍 **Web Search**: Tavily API
        """)
        
    
    # Check API keys
    missing_keys = check_api_keys()
    if missing_keys:
        st.error(f"Please configure the following API keys: {', '.join(missing_keys)}")
        st.info("Add your API keys to a `.env` file or set them as environment variables.")
        st.stop()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Document loading section
    st.sidebar.subheader("Knowledge Base")
    
    use_default_urls = st.sidebar.checkbox("Use default sample URLs", value=True)
    
    if use_default_urls:
        urls = config.SAMPLE_URLS
        st.sidebar.write("Using sample URLs:")
        for url in urls:
            st.sidebar.write(f"- {url}")
    else:
        urls_text = st.sidebar.text_area(
            f"Enter URLs (one per line, max {config.MAX_URLS}):",
            height=100,
            placeholder="https://example.com/page1\nhttps://example.com/page2"
        )
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        # Show URL count and limit
        if urls:
            url_count = len(urls)
            if url_count > config.MAX_URLS:
                st.sidebar.warning(f"⚠️ Only first {config.MAX_URLS} URLs will be used (you entered {url_count})")
            else:
                st.sidebar.info(f"📄 {url_count} URL{'s' if url_count != 1 else ''} entered")
    
    if st.sidebar.button("Load Documents", type="primary"):
        if not urls:
            st.sidebar.error("Please provide at least one URL.")
        else:
            success, message = load_documents(urls)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
    
    if st.session_state.documents_loaded:
        st.sidebar.success(f"✅ {st.session_state.doc_count} documents loaded")
    
    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; color: #999; font-size: 12px; margin-top: 20px;'>"
        "Author: <a href='https://www.linkedin.com/in/bart%C5%82omiej-m%C4%85kina' target='_blank' style='color: #999; text-decoration: none;'>Bartlomiej Makina</a><br>"
        "<a href='https://www.inteligems.io/' target='_blank' style='color: #999; text-decoration: none;'>InteliGems®Labs</a>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Main content area
    if not st.session_state.documents_loaded:
        st.info("👆 Please load documents using the sidebar to begin comparison.")
        return
    
    # Query section
    st.header("Ask a Question")
    
    # Sample questions from config
    sample_questions = config.SAMPLE_QUESTIONS[:8]  # Limit to first 8 for UI
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="Ask anything about the loaded documents..."
        )
    
    with col2:
        selected_sample = st.selectbox(
            "Or select a sample:",
            [""] + sample_questions
        )
    
    if selected_sample:
        question = selected_sample
    
    if st.button("Compare Systems", type="primary", disabled=not question):
        if question:
            col1, col2 = st.columns(2)
            
            # RAG System
            with col1:
                with st.spinner("Querying RAG system..."):
                    start_time = time.time()
                    rag_response = st.session_state.rag_system.query(question)
                    rag_time = time.time() - start_time
                
                display_system_response(rag_response, "RAG")
                st.caption(f"⏱️ Response time: {rag_time:.2f} seconds")
            
            # CRAG System
            with col2:
                with st.spinner("Querying CRAG system..."):
                    start_time = time.time()
                    crag_response = st.session_state.crag_system.query(question)
                    crag_time = time.time() - start_time
                
                display_system_response(crag_response, "CRAG")
                st.caption(f"⏱️ Response time: {crag_time:.2f} seconds")
            
            # Comparison summary
            st.header("📊 Comparison Summary")
            
            # Summary metrics in a clean layout
            st.subheader("📈 Key Metrics")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric(
                    "RAG Documents", 
                    rag_response["num_documents"],
                    help="Number of documents used by RAG system"
                )
            
            with summary_col2:
                st.metric(
                    "CRAG Documents", 
                    crag_response["num_documents"],
                    delta=crag_response["num_documents"] - rag_response["num_documents"],
                    help="Number of documents used by CRAG system"
                )
            
            with summary_col3:
                relevance_pct = f"{crag_response.get('relevance_ratio', 0)*100:.0f}%"
                st.metric(
                    "Document Relevance", 
                    relevance_pct,
                    help="Percentage of documents deemed relevant by CRAG"
                )
            
            with summary_col4:
                web_search_status = "✅ Yes" if crag_response.get("web_search_performed", False) else "❌ No"
                st.metric(
                    "Web Search", 
                    web_search_status,
                    help="Whether CRAG performed web search"
                )
            
            # Key insights section
            st.subheader("🔍 Key Insights")
            insights = []
            
            # Performance insight
            time_diff = crag_time - rag_time
            if time_diff > 0:
                insights.append(f"🐌 CRAG took {time_diff:.1f}s longer but provided {'more accurate' if crag_response.get('web_search_performed') else 'similar'} results")
            else:
                insights.append(f"⚡ CRAG was {abs(time_diff):.1f}s faster than RAG")
            
            # Document filtering insight
            if "relevance_ratio" in crag_response:
                if crag_response['relevance_ratio'] < 0.5:
                    insights.append(f"🎯 CRAG detected low relevance ({crag_response['relevance_ratio']*100:.0f}%) and applied corrective measures")
                else:
                    insights.append(f"✅ CRAG found high relevance ({crag_response['relevance_ratio']*100:.0f}%) in retrieved documents")
            
            # Web search insight
            if crag_response.get("web_search_performed", False):
                insights.append("🌐 CRAG enhanced results with web search when local knowledge was insufficient")
                if crag_response.get("improved_question"):
                    insights.append(f"🔄 CRAG refined the query for better search results")
            
            # Document filtering insight (when no web search but docs were filtered)
            elif "initial_documents_count" in crag_response and crag_response["initial_documents_count"] > crag_response["num_documents"]:
                filtered_count = crag_response["initial_documents_count"] - crag_response["num_documents"]
                insights.append(f"📋 CRAG filtered out {filtered_count} irrelevant document{'s' if filtered_count != 1 else ''} for higher quality context")
            
            # Answer quality insight
            rag_answered = "don't know" not in rag_response["answer"].lower() and "error" not in rag_response["answer"].lower()
            crag_answered = "don't know" not in crag_response["answer"].lower() and "error" not in crag_response["answer"].lower()
            
            if crag_answered and not rag_answered:
                insights.append("🎯 CRAG provided a complete answer while RAG could not")
            elif rag_answered and not crag_answered:
                insights.append("⚡ RAG provided a sufficient answer from local knowledge")
            elif both_answered := (rag_answered and crag_answered):
                insights.append("✅ Both systems provided answers, showing knowledge base adequacy")
            
            for insight in insights:
                st.info(insight)


if __name__ == "__main__":
    main()
