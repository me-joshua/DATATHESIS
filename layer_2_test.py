import pandas as pd
import os
import time
from retriever import get_rag_prediction, get_best_rag_match

def test_layer2_full():
    # 1. Setup Data
    if not os.path.exists('test_dataset.csv'):
        print("❌ Error: test_dataset.csv not found.")
        return

    df = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['Category'] = df['Category'].str.strip().str.capitalize()

    results = []
    confident_hits = 0
    confident_correct = 0
    overall_correct = 0

    print(f"🧠 Full Layer 2 Evaluation (Semantic Engine) on {len(df)} samples...")
    print("-" * 105)
    print(f"{'ID':<5} | {'Actual':<15} | {'Predicted':<15} | {'Method':<30} | {'Status'}")
    print("-" * 105)

    start_time = time.time()

    # 2. Execution Loop
    for idx, row in df.iterrows():
        riddle = row['Question']
        actual = row['Category']
        riddle_id = row.get('ID', idx)
        
        # --- PHASE A: High-Confidence Check ---
        prediction = get_rag_prediction(riddle)
        
        if prediction:
            predicted = prediction['label'].capitalize()
            method = prediction['method']
            is_confident = 1
            confident_hits += 1
            if predicted == actual:
                confident_correct += 1
        else:
            # --- PHASE B: Best-Effort Weighted Vote (The Judge) ---
            predicted = get_best_rag_match(riddle)
            method = "Layer 2 (Weighted Vote)"
            is_confident = 0

        # Overall Accuracy Check
        is_correct = 1 if predicted == actual else 0
        if is_correct:
            overall_correct += 1
            status = "✅"
        else:
            status = "❌"

        # Log to Console (Sampling the output)
        print(f"{riddle_id:<5} | {actual:<15} | {predicted:<15} | {method:<30} | {status}")

        results.append({
            "ID": riddle_id,
            "Actual": actual,
            "Predicted": predicted,
            "Method": method,
            "Confident": is_confident,
            "Correct": is_correct
        })

    # 3. Final Statistics
    total = len(df)
    duration = time.time() - start_time
    
    # Precision of the high-confidence subset
    conf_precision = (confident_correct / confident_hits * 100) if confident_hits > 0 else 0
    # Overall Accuracy of the entire semantic engine
    overall_acc = (overall_correct / total * 100)

    print("-" * 105)
    print(f"📊 LAYER 2 (FULL SEMANTIC ENGINE) STATISTICS:")
    print(f"🎯 Overall RAG Accuracy: {overall_acc:.2f}% (Total Correct: {overall_correct}/{total})")
    print(f"📈 Confidence Rate:     {(confident_hits/total)*100:.2f}% ({confident_hits}/{total} met threshold)")
    print(f"💎 Semantic Precision:  {conf_precision:.2f}% (Accuracy when confident)")
    print(f"⏱️ Total Time:          {duration:.2f}s")
    print("-" * 105)

    # Save results for final visualization
    pd.DataFrame(results).to_csv('layer2_full_results.csv', index=False, encoding='utf-8-sig')
    print("📂 Detailed log saved to 'layer2_full_results.csv' for thesis visualization.")

if __name__ == "__main__":
    test_layer2_full()