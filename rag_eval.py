import pandas as pd
import time
from generator import classify_riddle

# 1. Setup
df = pd.read_csv('master_dataset.csv', encoding='utf-8-sig')

# CLEANING: Ensure the test set only contains unique riddles
df_unique = df.drop_duplicates(subset=['Question'])

# Sampling 50 unique riddles for a fair test
test_set = df_unique.sample(n=50, random_state=42) 
results = []

print(f"🚀 Starting RAG Evaluation on {len(test_set)} Unique Riddles...")
print("-" * 50)

# Using enumerate to track progress (1/50, 2/50...)
for count, (i, row) in enumerate(test_set.iterrows(), 1):
    riddle = row['Question']
    actual = str(row['Category']).strip()
    
    # Calls the upgraded generator (with Sticky Model Rotation)
    predicted = classify_riddle(riddle)
    
    # Validation logic
    is_correct = 1 if predicted.strip().lower() == actual.lower() else 0
    
    results.append({
        "Question": riddle,
        "Actual": actual,
        "RAG_Predicted": predicted,
        "RAG_Correct": is_correct
    })
    
    status = "✅" if is_correct else "❌"
    # Log with sequence number instead of DataFrame index to prevent confusion
    print(f"[{count}/50] {actual} -> {predicted} {status}")
    
    # Safety delay for rate limits
    time.sleep(1.2) 

# 2. Final Calculations
rag_df = pd.DataFrame(results)
rag_df.to_csv('rag_results.csv', index=False, encoding='utf-8-sig')

total_correct = rag_df['RAG_Correct'].sum()
total_riddles = len(rag_df)
final_acc = (total_correct / total_riddles) * 100

print("-" * 50)
print(f"✅ RAG Evaluation Done.")
print(f"📊 Total Correct: {total_correct} / {total_riddles}")
print(f"📊 Final RAG Accuracy: {final_acc:.2f}%")
print("-" * 50)