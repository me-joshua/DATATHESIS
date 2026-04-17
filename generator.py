import time
from openai import OpenAI
from retriever import get_similar_riddles
from augmentor import build_prompt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Setup GitHub Models Client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ GITHUB_TOKEN not found in .env file")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

# 2. Define Model Pool
MODEL_POOL = [
    "gpt-4o", 
    "meta-llama-3.1-405b-instruct", 
    "gpt-4o-mini", 
    "meta-llama-3.1-8b-instruct"
]

# Sticky pointer to keep track of the working model across function calls
current_model_idx = 0

def classify_riddle(user_input):
    """Pipeline with persistent (sticky) Model Rotation for Rate Limits."""
    global current_model_idx
    
    # RAG: Retrieve and Augment
    matches = get_similar_riddles(user_input, k=3)
    prompt = build_prompt(user_input, matches)
    
    # Try models starting from the last known working index
    while current_model_idx < len(MODEL_POOL):
        model_name = MODEL_POOL[current_model_idx]
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a specialized South Indian riddle classifier."},
                        {"role": "user", "content": prompt}
                    ],
                    model=model_name,
                    temperature=0,
                    max_tokens=15
                )
                
                result = response.choices[0].message.content.strip().replace(".", "")
                
                # Successful response: Extract category keyword
                for category in ["Logic", "Mathematical", "Wordplay", "Cultural"]:
                    if category.lower() in result.lower():
                        return category
                return result

            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit (429) - Trigger Sticky Switch
                if "429" in error_str or "rate limit" in error_str:
                    print(f"🚨 {model_name} rate limited. Permanently switching to next model...")
                    current_model_idx += 1
                    # Break the retry loop to immediately try the next model in the while loop
                    break 
                
                # Non-rate-limit error (Timeout, 500, etc.) - Quick Retry
                if attempt < max_retries - 1:
                    print(f"⚠️ {model_name} error: {e}. Retrying in 2s...")
                    time.sleep(2)
                else:
                    print(f"❌ {model_name} failed permanently for this request.")
                    # Move to next model if this one is just broken
                    current_model_idx += 1
                    break

        # If we ran out of models in the pool
        if current_model_idx >= len(MODEL_POOL):
            break

    return "Error: All models exhausted"

if __name__ == "__main__":
    # Test cases to verify rotation
    test_q = "അമ്മിമ്മയ്ക്ക് ഒരു പല്ല്, അപ്പൂപ്പന് മുപ്പത്തിരണ്ട് പல்ல്. എന്താണത്?"
    print(f"Testing: {test_q}")
    print(f"Result: {classify_riddle(test_q)}")