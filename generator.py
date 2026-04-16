import time
from openai import OpenAI
from retriever import get_similar_riddles
from augmentor import build_prompt
import os
from dotenv import load_dotenv
# 1. Setup GitHub Models Client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

# 2. Define Model Pool based on your available models
# Prioritizing the 405B monster since your GPT-4o is rate-limited
MODEL_POOL = [
    "gpt-4o", 
    "meta-llama-3.1-405b-instruct", 
    "gpt-4o-mini", 
    "meta-llama-3.1-8b-instruct"
]

def classify_riddle(user_input):
    """Pipeline with automatic Model Rotation for Rate Limits."""
    
    matches = get_similar_riddles(user_input, k=3)
    prompt = build_prompt(user_input, matches)
    
    # Iterate through models in the pool
    for model_name in MODEL_POOL:
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
                
                # Extract category keyword
                for category in ["Logic", "Mathematical", "Wordplay", "Cultural"]:
                    if category.lower() in result.lower():
                        return category
                return result

            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit (429) or capacity issues
                if "429" in error_str or "rate limit" in error_str:
                    print(f"🚨 {model_name} rate limited. Switching to next model in pool...")
                    break # Break inner loop to try the next model name
                
                # If it's a different error, try a quick retry
                if attempt < max_retries - 1:
                    print(f"⚠️ {model_name} error: {e}. Retrying in 2s...")
                    time.sleep(2)
                else:
                    print(f"❌ {model_name} failed permanently.")
                    break 

    return "Error: All models exhausted"

if __name__ == "__main__":
    test_q = "അമ്മിമ്മയ്ക്ക് ഒരു പല്ല്, അപ്പൂപ്പന് മുപ്പത്തിരണ്ട് പല്ല്. എന്താണത്?"
    print(f"Testing: {test_q}")
    print(f"Result: {classify_riddle(test_q)}")