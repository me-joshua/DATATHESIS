import pandas as pd
import time
import os
from openai import OpenAI  
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
    timeout=15.0 
)

# 2. Model Pool
MODEL_POOL = [
    "gpt-4o", 
    "meta-llama-3.1-405b-instruct", 
    "gpt-4o-mini", 
    "meta-llama-3.1-8b-instruct"
]
current_model_idx = 0 

# 3. Load the TEST SET
if os.path.exists('test_dataset.csv'):
    test_set = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    # CLEANUP: Strip hidden spaces from headers
    test_set.columns = test_set.columns.str.strip()
    
    # SELF-HEALING: Create ID if it doesn't exist
    if 'ID' not in test_set.columns:
        print("💡 'ID' column missing in test_set. Generating sequential IDs...")
        test_set['ID'] = range(1, len(test_set) + 1)
        
    print(f"🧪 Loaded test_dataset.csv ({len(test_set)} riddles)")
else:
    print("❌ ERROR: Run your split script first to generate test_dataset.csv")
    exit()

results = []

print(f"📉 Starting Zero-Shot Baseline (The Control Group)...")
print("-" * 75)

for count, (idx, row) in enumerate(test_set.iterrows(), 1):
    riddle = row['Question']
    actual = str(row['Category']).strip()
    riddle_id = row['ID']
    
    prompt = f"""Classify: Logic, Mathematical, Wordplay, or Cultural. Output ONLY the category.
Riddle: {riddle}
Category:"""

    predicted = "Error"
    model_used = "None"
    temp_idx = current_model_idx 

    while temp_idx < len(MODEL_POOL):
        model_name = MODEL_POOL[temp_idx]
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a specialized South Indian riddle classifier."},
                    {"role": "user", "content": prompt}
                ],
                model=model_name, 
                temperature=0,
                max_tokens=20
            )
            raw_output = response.choices[0].message.content.strip().replace(".", "")
            
            for category in ["Logic", "Mathematical", "Wordplay", "Cultural"]:
                if category.lower() in raw_output.lower():
                    predicted = category
                    break
            
            if "Error" in predicted:
                predicted = f"Unmatched ({raw_output[:10]}...)"
            
            model_used = model_name
            break 

        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                current_model_idx += 1 
                temp_idx = current_model_idx
            else:
                temp_idx += 1
            
            if temp_idx >= len(MODEL_POOL):
                break

    is_correct = 1 if predicted.lower() == actual.lower() else 0
    results.append({
        "ID": riddle_id,
        "Question": riddle,
        "Actual": actual,
        "Baseline_Predicted": predicted,
        "Baseline_Correct": is_correct
    })
    
    status = "✅" if is_correct else "❌"
    print(f"[{count}/{len(test_set)}] ID:{riddle_id:<4} | {actual:<12} -> {predicted:<15} {status}")
    time.sleep(0.8)

# 4. Save
baseline_df = pd.DataFrame(results)
baseline_df.to_csv('baseline_results.csv', index=False, encoding='utf-8-sig')

final_acc = baseline_df['Baseline_Correct'].mean() * 100
print("-" * 75)
print(f"📊 Final Baseline Accuracy: {final_acc:.2f}%")