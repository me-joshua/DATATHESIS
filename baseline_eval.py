import pandas as pd
import time
import os
from openai import OpenAI  
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("❌ GITHUB_TOKEN not found! Check your .env file.")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

# 2. Define Model Pool
MODEL_POOL = [
    "meta-llama-3.1-8b-instruct",
    "gpt-4o", 
    "meta-llama-3.1-405b-instruct", 
    "gpt-4o-mini"
]

# This index will "stick" throughout the entire script run
current_model_idx = 0

# 3. Load data
df = pd.read_csv('master_dataset.csv', encoding='utf-8-sig')
test_set = df.sample(n=50, random_state=42) 

results = []

print(f"📉 Starting Optimized Baseline on {len(test_set)} riddles...")

for i, row in test_set.iterrows():
    riddle = row['Question']
    actual = str(row['Category']).strip()
    prompt = f"Classify the following riddle into one of these categories: Logic, Mathematical, Wordplay, or Cultural. Output ONLY the category name.\n\nRiddle: {riddle}\nCategory:"

    predicted = "Error"
    success = False

    # Try models starting from the last known working model
    while current_model_idx < len(MODEL_POOL):
        model_name = MODEL_POOL[current_model_idx]
        
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
            
            raw_output = response.choices[0].message.content.strip().replace(".", "")
            
            # Extract category
            for category in ["Logic", "Mathematical", "Wordplay", "Cultural"]:
                if category.lower() in raw_output.lower():
                    predicted = category
                    break
            
            success = True
            break # Exit the 'while' loop - we found a working model!

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str:
                print(f"🚨 {model_name} is rate limited. Permanent switch to next model...")
                current_model_idx += 1 # Move the "sticky" pointer to the next model
                if current_model_idx >= len(MODEL_POOL):
                    print("❌ ALL MODELS EXHAUSTED.")
                    break
                continue # Re-try the SAME riddle with the new model immediately
            
            else:
                print(f"⚠️ Unexpected error with {model_name}: {e}")
                break # Non-429 error, move to next riddle

    is_correct = 1 if predicted.lower() == actual.lower() else 0
    results.append({
        "Question": riddle,
        "Actual": actual,
        "Baseline_Predicted": predicted,
        "Baseline_Correct": is_correct,
        "Model_Used": MODEL_POOL[current_model_idx] if current_model_idx < len(MODEL_POOL) else "None"
    })
    
    status = "✅" if is_correct else "❌"
    used = MODEL_POOL[current_model_idx] if current_model_idx < len(MODEL_POOL) else "N/A"
    print(f"[{i}] {actual} -> {predicted} ({used}) {status}")
    
    time.sleep(0.5)

# 4. Save
baseline_df = pd.DataFrame(results)
baseline_df.to_csv('baseline_results.csv', index=False, encoding='utf-8-sig')

print(f"\n✅ Baseline complete using sticky model logic.")
print(f"📊 Final Accuracy: {baseline_df['Baseline_Correct'].mean()*100:.2f}%")