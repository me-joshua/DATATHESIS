import pandas as pd
import time
import os
from generator import classify_riddle

def evaluate_main_pipeline():
    # 1. Setup Data
    if not os.path.exists('test_dataset.csv'):
        print("❌ Error: test_dataset.csv not found.")
        return

    df = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['Category'] = df['Category'].str.strip().str.capitalize()
    
    # Ensure ID exists
    if 'ID' not in df.columns:
        df['ID'] = range(len(df))

    results = []

    print(f"🚀 Running FINAL Hybrid Pipeline Evaluation on {len(df)} samples...")
    print("Logic: Layer 1 (Weighted Keywords) -> Layer 2 (Weighted Semantic Engine)")
    print("-" * 110)
    print(f"{'ID':<5} | {'Actual':<15} | {'Predicted':<15} | {'Method Used':<35} | {'Status'}")
    print("-" * 110)

    start_time = time.time()

    # 2. Execution Loop
    for idx, row in df.iterrows():
        riddle = row['Question']
        actual = row['Category']
        riddle_id = row['ID']
        
        row_start = time.time()
        
        # This calls the deterministic L1 + L2 logic in generator.py
        predicted, method = classify_riddle(riddle)
        
        latency = time.time() - row_start
        is_correct = 1 if predicted == actual else 0
        status = "✅" if is_correct else "❌"

        print(f"{riddle_id:<5} | {actual:<15} | {predicted:<15} | {method:<35} | {status}")

        results.append({
            "ID": riddle_id,
            "Question": riddle,
            "Actual": actual,
            "Predicted": predicted,
            "Method": method,
            "Latency": round(latency, 4),
            "Correct": is_correct
        })

    # 3. Final Statistics
    total = len(df)
    total_duration = time.time() - start_time
    res_df = pd.DataFrame(results)
    
    # Save the master results file
    res_df.to_csv('rag_results.csv', index=False, encoding='utf-8-sig')
    
    final_accuracy = (res_df['Correct'].sum() / total) * 100
    avg_latency = res_df['Latency'].mean()

    print("-" * 110)
    print(f"🏆 FINAL PROJECT STATISTICS (DETERMINISTIC PIPELINE):")
    print(f"🎯 Overall Accuracy: {final_accuracy:.2f}%")
    print(f"⏱️ Average Latency:  {avg_latency:.4f}s per riddle")
    print(f"📂 Master data saved to 'rag_results.csv'")
    print("-" * 110)

if __name__ == "__main__":
    evaluate_main_pipeline()