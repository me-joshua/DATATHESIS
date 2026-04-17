# 🧩 Hybrid Riddle Classifier AI (Tamil & Malayalam)
### A 3-Layer Deterministic RAG Architecture for Dravidian Folk Literature

This project implements a high-performance, cost-effective hybrid pipeline to classify Tamil and Malayalam riddles into four distinct categories: Logic, Mathematical, Wordplay, and Cultural. The system prioritizes local deterministic logic to minimize LLM latency and API costs, using the LLM exclusively for linguistic reasoning.

---

## 🏗️ Architecture: The 3-Layer Pipeline

1. Layer 1: Weighted Rule-Based (Keywords)
A local Python-based regex engine that calculates scores based on "Hard" and "Soft" linguistic signals.

2. Layer 2: Semantic RAG (Retrieval-Augmented Generation)
Uses 'paraphrase-multilingual-mpnet-base-v2' and 'FAISS' to perform a weighted k-Nearest Neighbor search against a curated dataset.

3. Layer 3: LLM Reasoning (Linguistic Expert)
If classification is successful in local layers, an LLM (GPT-4o/Llama-3.1) provides a two-sentence linguistic justification grounded in the retrieved context.

---

## 🚀 Phase-by-Phase Implementation

### Phase 1: Data Engineering & Pre-processing
* Dataset: A bilingual corpus of Tamil and Malayalam riddles.
* Cleaning: Normalization of Dravidian characters, removal of whitespace, and Title Case standardization for categories.
* Splitting: Stratified split into test_dataset.csv to ensure category balance.

### Phase 2: Semantic Indexing (The Vector Brain)
* Script: create_index.py
* Embedding Model: HuggingFace sentence-transformers.
* Index: FAISS IndexFlatIP (Inner Product) for cosine similarity.
* Storage: Saves riddles.index and riddle_labels.json.

### Phase 3: Deterministic Logic Development
* Script: hybrid_classifier.py
* Mechanism: Implements a "Tiered Confidence" scoring system. 
* Innovation: Includes a "Body Part" keyword expansion and a "Phonetic Wordplay" detector that ignores common Dravidian stop-words to prevent false positives in rhythmic riddles.

### Phase 4: The Hybrid Orchestrator
* Script: generator.py
* Logic: 
    - classify_riddle(): Routes the input through Layer 1 and Layer 2.
    - get_llm_reasoning(): Triggered only after a label is found. Uses a "Model Pool" (GPT-4o -> Llama-405b -> GPT-4o-mini) to handle 429 Rate Limit errors.

### Phase 5: UI Integration
* Script: app.py
* Framework: Streamlit.
* Features: Real-time processing metrics, "Layer Badges" indicating how each riddle was solved, and expanders to view the underlying RAG semantic neighbors.

---

## 📊 Evaluation & Analytics Phase

| Script | Purpose |
| :--- | :--- |
| baseline_eval.py | Measures raw LLM performance (No RAG/Keywords). |
| layer_1_test.py | Validates the Precision/Recall of the keyword engine. |
| layer_2_test.py | Evaluates the semantic engine's standalone accuracy. |
| rag_eval.py | The final test of the integrated Hybrid system. |
| visualize_final.py | Generates Accuracy Bar Charts and Workload Distribution Pie Charts. |
| final_analytics.py | Generates Confusion Matrices (PNG) and detailed Recall/Precision reports. |

---

## 📂 Project Structure

DATATHESIS/
├── .env                # GitHub/OpenAI API Tokens
├── app.py              # Streamlit UI
├── create_index.py     # FAISS Index Generator
├── generator.py        # Logic Orchestrator & LLM Manager
├── retriever.py        # RAG Search & Weighted Voting Logic
├── hybrid_classifier.py # Layer 1 Keyword Logic
├── visualize_final.py  # Thesis Visualizations
├── data/
│   ├── train_dataset.csv
│   └── test_dataset.csv
└── models/
    ├── riddles.index
    └── riddle_labels.json

---

## 🛠️ Setup & Usage

1. Install Dependencies:
pip install streamlit pandas faiss-cpu sentence-transformers openai python-dotenv matplotlib seaborn scikit-learn

2. Generate the Semantic Index:
python create_index.py

3. Run the Evaluation Suite:
python rag_eval.py
python final_analytics.py

4. Launch the Interface:
streamlit run app.py

---

## 📈 Key Results
* Baseline Accuracy: 46%
* Hybrid Accuracy: 84.75%
* Technical Advantage: The Hybrid system significantly reduces LLM tokens by resolving ~30-50% of requests locally via Layer 1 and Layer 2, while maintaining >80% accuracy on complex regional folk metaphors.
