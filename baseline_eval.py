import pandas as pd
import time
from openai import OpenAI  # pip install openai
import os
from dotenv import load_dotenv
# 1. Setup GitHub Models Client
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

# Load data
df = pd.read_csv('master_dataset.csv', encoding='utf-8-sig')

# CRITICAL: random_state=42 keeps the test set identical across all runs
test_set = df.sample(n=50, random_state=42) 

results = []

print(f"📉 Starting Zero-Shot Baseline (GPT-4o) on {len(test_set)} riddles...")

for i, row in test_set.iterrows():
    riddle = row['Question']
    actual = str(row['Category']).strip()
    
    # Zero-shot prompt (No examples/RAG)
    prompt = f"Classify the following riddle into one of these categories: Logic, Mathematical, Wordplay, or Cultural. Output ONLY the category name.\n\nRiddle: {riddle}\nCategory:"

    predicted = "Error"
    max_retries = 3
    
    # --- RETRY LOOP FOR RATE LIMITS/ERRORS ---
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a specialized South Indian riddle classifier."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o", 
                temperature=0,
                max_tokens=15
            )
            
            raw_output = response.choices[0].message.content.strip().replace(".", "")
            
            # Keyword extraction
            for category in ["Logic", "Mathematical", "Wordplay", "Cultural"]:
                if category.lower() in raw_output.lower():
                    predicted = category
                    break
            break 

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Attempt {attempt+1} failed at index {i}. Retrying in 5s...")
                time.sleep(5)
            else:
                print(f"❌ Permanent Failure at index {i}: {e}")
                predicted = "Error"

    is_correct = 1 if predicted.lower() == actual.lower() else 0
    
    results.append({
        "Question": riddle,
        "Actual": actual,
        "Baseline_Predicted": predicted,
        "Baseline_Correct": is_correct
    })
    
    status = "✅" if is_correct else "❌"
    print(f"[{i}] {actual} -> {predicted} {status}")
    
    # Respecting GitHub Student Tier rate limits
    time.sleep(0.3)

# 2. Save Baseline
baseline_df = pd.DataFrame(results)
baseline_df.to_csv('baseline_results.csv', index=False, encoding='utf-8-sig')

print(f"\n✅ Baseline complete.")
print(f"📊 Baseline Accuracy: {baseline_df['Baseline_Correct'].mean()*100:.2f}%")