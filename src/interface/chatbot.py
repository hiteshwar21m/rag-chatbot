"""
MANIT Chatbot - Final Version
Uses hybrid retrieval + OpenRouter LLM
"""
import streamlit as st
import sys
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

# Add path for hybrid retriever and logger
sys.path.append(str(Path(__file__).parent.parent / "retrieval"))
from hybrid_retriever import HybridRetriever

sys.path.append(str(Path(__file__).parent.parent / "monitoring"))
from query_logger import get_logger

sys.path.append(str(Path(__file__).parent.parent / "config"))
from settings import get_config

# Load API key
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / "config/.env")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'retriever' not in st.session_state:
    st.session_state.retriever = HybridRetriever()

if 'logger' not in st.session_state:
    st.session_state.logger = get_logger()

def generate_answer(query, chunks):
    """Generate answer using LLM"""
    import time
    
    # Build context from chunks
    context = "\n\n".join([
        f"[{i+1}] {chunk['document_title']} - {chunk.get('section', 'N/A')}\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    ])
    
    prompt = f"""You are a helpful assistant for MANIT Bhopal. Answer questions directly and comprehensively using the information provided.

Context:
{context}

Question: {query}

Instructions:
- Answer the question DIRECTLY - don't say "based on the context" or similar phrases
- If asked for specific information (like syllabus, requirements, fees), provide it in detail
- Use clear formatting with paragraphs and bullet points
- Be thorough and include all relevant details from the context
- Write naturally as if speaking to a student
- If information is not available, simply say "I don't have that information"

Answer:"""
    
    # Get config
    config = get_config()
    api_key = config.openrouter_api_key
    
    if not api_key:
        return "Error: API key not found. Please set OPENROUTER_API_KEY in config/.env", 0
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    model = config.llm_model
    temperature = config.llm_temperature
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "MANIT Chatbot"
    }
    
    try:
        start = time.time()
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        llm_time = time.time() - start
        
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            return answer, llm_time, model, temperature
        else:
            return f"Error: {response.status_code} - {response.text[:200]}", 0, model, temperature
    except Exception as e:
        return f"Error calling LLM: {str(e)}", 0, model, temperature

# Page config
st.set_page_config(
    page_title="MANIT Q&A", 
    page_icon="🎓", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎓 MANIT Q&A Chatbot")
st.caption("Ask questions about MANIT Bhopal")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.container():
            st.markdown("**👤 You:**")
            st.write(message["content"])
    else:
        with st.container():
            st.markdown("**🤖 Assistant:**")
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about MANIT..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show user message
    with st.container():
        st.markdown("**👤 You:**")
        st.write(prompt)
    
    # Generate response
    with st.spinner("🔍 Searching and generating answer..."):
        # Start logging
        logger = st.session_state.logger
        logger.start_query(prompt)
        
        # Retrieve chunks (with logging)
        chunks = st.session_state.retriever.retrieve(prompt, top_k=5, logger=logger)
        
        # Generate answer (with logging)
        answer, llm_time, model, temperature = generate_answer(prompt, chunks)
        logger.log_llm(llm_time, model, temperature, len(answer))
        
        # End logging
        logger.end_query()
        
        # Show bot message
        with st.container():
            st.markdown("**🤖 Assistant:**")
            st.write(answer)
        
        # Add to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer
        })
    
    # Auto-rerun to show message properly
    st.rerun()

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This chatbot uses:
    - **Gemini 2.0 Flash** LLM
    """)
    
    st.write(f"**Total messages:** {len(st.session_state.messages)}")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
