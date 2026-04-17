import pandas as pd
import time
import os
from generator import classify_riddle

def evaluate_pipeline():
    # 1. Setup Data
    if not os.path.exists('test_dataset.csv'):
        print("❌ Error: test_dataset.csv not found. Run your split script first.")
        return

    df = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['Category'] = df['Category'].str.strip().str.capitalize()
    
    # Self-healing ID
    if 'ID' not in df.columns:
        df['ID'] = range(len(df))

    results = []

    print(f"🚀 Starting Deterministic Evaluation (L1 + L2) on {len(df)} riddles...")
    print("-" * 105)
    print(f"{'ID':<5} | {'Actual':<15} | {'Predicted':<15} | {'Method':<35} | {'Status'}")
    print("-" * 105)

    # 2. Evaluation Loop
    for count, (idx, row) in enumerate(df.iterrows(), 1):
        riddle = row['Question']
        actual = row['Category']
        riddle_id = row['ID']
        
        start_time = time.time()
        
        # Now calls the updated generator which returns (label, method)
        # LLM Reasoning is bypassed for evaluation to focus on Classification Accuracy
        predicted, method = classify_riddle(riddle)
        
        duration = time.time() - start_time
        is_correct = 1 if predicted == actual else 0
        status = "✅" if is_correct else "❌"

        print(f"{riddle_id:<5} | {actual:<15} | {predicted:<15} | {method:<35} | {status}")

        results.append({
            "ID": riddle_id,
            "Question": riddle,
            "Actual": actual,
            "Predicted": predicted,
            "Method": method,
            "Latency": round(duration, 4),
            "Correct": is_correct
        })

    # 3. Final Statistics
    res_df = pd.DataFrame(results)
    res_df.to_csv('rag_results.csv', index=False, encoding='utf-8-sig')
    
    accuracy = (res_df['Correct'].sum() / len(res_df)) * 100
    avg_latency = res_df['Latency'].mean()

    print("-" * 105)
    print(f"📊 DETERMINISTIC SYSTEM EVALUATION (No LLM Bias):")
    print(f"🎯 Total Accuracy: {accuracy:.2f}%")
    print(f"⏱️ Avg Latency:    {avg_latency:.4f}s per riddle")
    print(f"📂 Data saved to 'rag_results.csv'")
    print("-" * 105)

if __name__ == "__main__":
    evaluate_pipeline()