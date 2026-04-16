import streamlit as st
import time
from generator import classify_riddle

# 1. Page Configuration - No 'max-width' to kill negative space
st.set_page_config(page_title="Riddle Classifier AI", page_icon="🧩", layout="wide")

# 2. Modern, Theme-Aware CSS
# We use 'var(--text-color)' so it automatically flips between black and white
st.markdown("""
    <style>
    /* Remove the forced white background to allow Dark Mode to work */
    .stApp { background-color: transparent; }
    
    /* Category result box style - Theme Aware */
    .category-bubble {
        padding: 10px 20px;
        border-radius: 15px;
        display: inline-block;
        font-weight: 600;
        margin-top: 8px;
        border: 1px solid var(--secondary-background-color);
        color: #333333; /* Keep text dark for the colored bubbles */
    }

    /* Target the Title specifically to ensure visibility */
    .main-title {
        color: var(--text-color);
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Initialization - Persistent History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Header - Using a class for theme-aware coloring
st.markdown('<h1 class="main-title">🧩 Riddle AI</h1>', unsafe_allow_html=True)
st.caption("RAG-Enhanced Multilingual Classification")

# 5. Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "category" in message:
            # Theme-friendly pastel colors
            bg_colors = {
                "Logic": "#D1E8FF",
                "Mathematical": "#D1FFD7",
                "Wordplay": "#FFF4D1",
                "Cultural": "#FFD1D1",
                "Error": "#FFCCCC"
            }
            color = bg_colors.get(message["category"], "#f5f5f5")
            st.markdown(f'<div class="category-bubble" style="background-color: {color};">Category: {message["category"]}</div>', unsafe_allow_html=True)
            
            # Persistent "Reasoned in" caption
            if "duration" in message:
                st.caption(f"Reasoned in {message['duration']:.2f}s")

# 6. Chat Input Logic
if prompt := st.chat_input("Ask a riddle..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            start_time = time.time()
            prediction = classify_riddle(prompt)
            end_time = time.time()
            duration = end_time - start_time
            
            response_text = f"Classification complete for the provided riddle."
            st.markdown(response_text)
            
            # Display current category bubble
            bg_colors = {
                "Logic": "#D1E8FF",
                "Mathematical": "#D1FFD7",
                "Wordplay": "#FFF4D1",
                "Cultural": "#FFD1D1",
                "Error": "#FFCCCC"
            }
            color = bg_colors.get(prediction, "#f5f5f5")
            st.markdown(f'<div class="category-bubble" style="background-color: {color};">Category: {prediction}</div>', unsafe_allow_html=True)
            st.caption(f"Reasoned in {duration:.2f}s")

    # CRITICAL: Save duration to history so it stays when the app reruns
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "category": prediction,
        "duration": duration
    })