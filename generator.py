import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Import our custom logic layers
from hybrid_classifier import RuleBasedClassifier
from retriever import get_similar_riddles, get_rag_prediction, get_best_rag_match

load_dotenv()

# 1. Setup GitHub Models Client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ GITHUB_TOKEN not found in .env file.")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
    timeout=15.0 
)

# Optimized Model Pool for Reasoning
MODEL_POOL = [
    "gpt-4o", 
    "meta-llama-3.1-405b-instruct", 
    "gpt-4o-mini"
]

current_model_idx = 0
rule_clf = RuleBasedClassifier()

def standardize_label(raw_label: str) -> str:
    """Helper to ensure all layers return the exact Title Case label."""
    mapping = {
        "logic": "Logic",
        "mathematical": "Mathematical",
        "wordplay": "Wordplay",
        "cultural": "Cultural"
    }
    clean = str(raw_label).lower().strip()
    for key, value in mapping.items():
        if key in clean:
            return value
    return "Logic" # Default fallback for local logic

def classify_riddle(user_input):
    """
    STRICT LOCAL CLASSIFICATION:
    Bypasses LLM for classification to ensure deterministic, database-grounded results.
    Returns: (label, method_description)
    """
    # 1. Layer 1: Weighted Rule-Based (Keywords)
    rule_result = rule_clf.classify(user_input)
    if rule_result:
        return standardize_label(rule_result.label), "Layer 1 (Weighted Keywords)"

    # 2. Layer 2: High-Confidence RAG (Strict Semantic Match)
    rag_prediction = get_rag_prediction(user_input)
    if rag_prediction:
        return standardize_label(rag_prediction['label']), rag_prediction['method']

    # 3. Layer 2 Fallback: Weighted Semantic Voter (The Final Judge)
    # This ensures a local decision is made based on the top 5 nearest neighbors.
    best_match_label = get_best_rag_match(user_input)
    return standardize_label(best_match_label), "Layer 2 (Weighted Semantic Vote)"

def get_llm_reasoning(user_input, final_label):
    """
    LLM REASONING (CONTEXT-AWARE):
    Uses the LLM to explain the logic behind the local classification choice.
    """
    global current_model_idx
    
    # 1. Fetch context (semantic neighbors) to help the LLM provide accurate reasoning
    matches = get_similar_riddles(user_input, k=3)
    
    context_str = ""
    if matches:
        context_str = "\n".join([
            f"Reference {i+1}: {m['Question']} (Category: {m['Category']})" 
            for i, m in enumerate(matches)
        ])
    else:
        context_str = "No specific database matches found."

    # 2. Build the context-grounded reasoning prompt
    reasoning_prompt = f"""
You are an expert in South Indian (Tamil/Malayalam) folk literature and riddles. 
The following riddle has been classified as '{final_label}' based on our database.

RIDDLE TO EXPLAIN: "{user_input}"
ASSIGNED CATEGORY: {final_label}

SIMILAR EXAMPLES FROM DATABASE:
{context_str}

TASK:
Provide a professional 2-sentence linguistic explanation of why this riddle fits the '{final_label}' category.
- If Cultural: Mention the personification or regional metaphors.
- If Logic: Mention the functional deduction or physical properties.
- If Wordplay: Mention the linguistic pun or sound pattern.
- If Mathematical: Mention the quantitative nature.

EXPLANATION:"""
    
    temp_idx = current_model_idx 
    while temp_idx < len(MODEL_POOL):
        model_name = MODEL_POOL[temp_idx]
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a riddle analyst providing short, expert reasoning."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                model=model_name,
                temperature=0.5,
                max_tokens=120 
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                print(f"🚨 {model_name} rate limited. Switching pointer...")
                current_model_idx += 1
                temp_idx = current_model_idx
            else:
                temp_idx += 1 
            
            if temp_idx >= len(MODEL_POOL):
                break

    return "Reasoning unavailable due to rate limits, but classification is verified locally."

if __name__ == "__main__":
    # Test a tricky cultural riddle
    test_q = "മൂന്ന് കണ്ണുണ്ട് ശിവനുമല്ല, വാലുണ്ട് പക്ഷിയുമല്ല." # Coconut
    label, method = classify_riddle(test_q)
    print(f"🎯 Category: {label} ({method})")
    
    reasoning = get_llm_reasoning(test_q, label)
    print(f"💡 Reasoning: {reasoning}")