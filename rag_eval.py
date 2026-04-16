import pandas as pd
import time
from generator import classify_riddle

# 1. Setup
df = pd.read_csv('master_dataset.csv', encoding='utf-8-sig')

# CRITICAL: Same random_state as baseline_eval to ensure the test is fair (apples-to-apples)
test_set = df.sample(n=50, random_state=42) 
results = []

print(f"🚀 Starting RAG Evaluation (GitHub Model Pool + MPNet) on {len(test_set)} riddles...")

for i, row in test_set.iterrows():
    riddle = row['Question']
    actual = str(row['Category']).strip()
    
    # Calls the upgraded generator (now with GPT-4o / Llama 405B auto-rotation)
    predicted = classify_riddle(riddle)
    
    # Robust case-insensitive comparison
    is_correct = 1 if predicted.strip().lower() == actual.lower() else 0
    
    results.append({
        "Question": riddle,
        "Actual": actual,
        "RAG_Predicted": predicted,
        "RAG_Correct": is_correct
    })
    
    status = "✅" if is_correct else "❌"
    print(f"[{i}] {actual} -> {predicted} {status}")
    
    # Respecting GitHub Student Tier rate limits. 
    # Even though generator has retries, a 1.5s buffer keeps the 'pool' healthy.
    time.sleep(1.5) 

# 2. Save
rag_df = pd.DataFrame(results)
rag_df.to_csv('rag_results.csv', index=False, encoding='utf-8-sig')

final_acc = rag_df['RAG_Correct'].mean() * 100
print(f"\n✅ RAG Evaluation Done.")
print(f"📊 Final RAG Accuracy: {final_acc:.2f}%")