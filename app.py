import streamlit as st
import time
from generator import classify_riddle, get_llm_reasoning # New functions
from retriever import get_similar_riddles

# 1. Page Configuration
st.set_page_config(page_title="Riddle Classifier AI", page_icon="🧩", layout="wide")

# 2. Enhanced CSS
st.markdown("""
    <style>
    .stApp { background-color: transparent; }
    
    .category-bubble {
        padding: 12px 24px;
        border-radius: 12px;
        display: inline-block;
        font-weight: 700;
        margin-top: 10px;
        border: 1px solid rgba(0,0,0,0.1);
        color: #1E1E1E;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .reasoning-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-left: 5px solid #9C27B0;
        margin-top: 15px;
        font-style: italic;
        border-radius: 0 8px 8px 0;
    }

    .layer-badge {
        font-size: 0.75rem;
        padding: 4px 8px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        margin-bottom: 5px;
        display: inline-block;
    }
    .layer-1 { background-color: #4CAF50; } 
    .layer-2 { background-color: #2196F3; } 
    .layer-3 { background-color: #9C27B0; } 

    .main-title { color: var(--text-color); font-weight: 800; font-size: 3rem; margin-bottom: 0px; }
    .metric-text { font-size: 0.85rem; color: #888; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.title("🛠️ Hybrid Pipeline")
    st.success("**Classification:** Layer 1 & 2 (Deterministic)")
    st.warning("**Reasoning:** Layer 3 (GPT/Llama)")
    st.divider()
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Header
st.markdown('<h1 class="main-title">🧩 Riddle AI</h1>', unsafe_allow_html=True)
st.caption("Deterministic Hybrid Pipeline: Local Classification + LLM Reasoning")

# 5. Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "category" in message:
            layer_class = f"layer-{message.get('layer_num', 2)}"
            st.markdown(f'<div class="layer-badge {layer_class}">Verified via {message.get("method", "Local Logic")}</div>', unsafe_allow_html=True)
            
            bg_colors = {"Logic": "#D1E8FF", "Mathematical": "#D1FFD7", "Wordplay": "#FFF4D1", "Cultural": "#FFD1D1", "Error": "#FFCCCC"}
            st.markdown(f'<div class="category-bubble" style="background-color: {bg_colors.get(message["category"], "#f5f5f5")};">🎯 Category: {message["category"]}</div>', unsafe_allow_html=True)
            
            if "reasoning" in message:
                st.markdown(f'<div class="reasoning-box">💡 <b>Expert Reasoning:</b><br>{message["reasoning"]}</div>', unsafe_allow_html=True)

# 6. Chat Input
if prompt := st.chat_input("Enter a riddle..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- PHASE 1: LOCAL CLASSIFICATION (INSTANT) ---
        start_time = time.time()
        
        # This now returns (label, method_name)
        prediction, method_name = classify_riddle(prompt)
        layer_num = 1 if "Layer 1" in method_name else 2
        
        duration_local = time.time() - start_time
        
        # Display Result Immediately
        st.markdown(f'<div class="layer-badge layer-{layer_num}">Solved via {method_name}</div>', unsafe_allow_html=True)
        bg_colors = {"Logic": "#D1E8FF", "Mathematical": "#D1FFD7", "Wordplay": "#FFF4D1", "Cultural": "#FFD1D1"}
        st.markdown(f'<div class="category-bubble" style="background-color: {bg_colors.get(prediction, "#f5f5f5")};">🎯 Category: {prediction}</div>', unsafe_allow_html=True)
        
        # --- PHASE 2: LLM REASONING (BACKGROUND) ---
        with st.spinner("Generating expert reasoning..."):
            reasoning = get_llm_reasoning(prompt, prediction)
            st.markdown(f'<div class="reasoning-box">💡 <b>Expert Reasoning:</b><br>{reasoning}</div>', unsafe_allow_html=True)
        
        total_duration = time.time() - start_time
        st.markdown(f'<p class="metric-text">Local Latency: {duration_local:.4f}s | Total: {total_duration:.2f}s</p>', unsafe_allow_html=True)

        # RAG Context for transparency
        matches = get_similar_riddles(prompt, k=3)
        if matches:
            with st.expander("📚 View Semantic Neighbors"):
                for m in matches:
                    st.markdown(f"**Similar:** {m['Question']} | **Label:** `{m['Category']}`")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Classification complete.", 
        "category": prediction,
        "method": method_name,
        "layer_num": layer_num,
        "reasoning": reasoning,
        "duration": total_duration
    })