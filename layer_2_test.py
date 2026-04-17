import pandas as pd
import os
import time
from retriever import get_rag_prediction

def test_layer2():
    # 1. Setup Data
    if not os.path.exists('test_dataset.csv'):
        print("❌ Error: test_dataset.csv not found.")
        return

    df = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    df['Category'] = df['Category'].str.strip().str.capitalize()

    results = []
    hits = 0
    correct_hits = 0

    print(f"🧠 Testing Layer 2 (Semantic RAG) on {len(df)} samples...")
    print("Confidence Threshold is set in retriever.py (Default: 0.88 or Unanimous Vote)")
    print("-" * 85)
    print(f"{'ID':<5} | {'Actual':<15} | {'RAG Prediction':<20} | {'Status'}")
    print("-" * 85)

    # 2. Execution
    start_time = time.time()

    for idx, row in df.iterrows():
        riddle = row['Question']
        actual = row['Category']
        riddle_id = row.get('ID', idx)
        
        # Call ONLY Layer 2
        # This bypasses keywords and doesn't call any LLM APIs
        prediction = get_rag_prediction(riddle)
        
        if prediction:
            hits += 1
            predicted = prediction['label'].replace("Second Layer", "").strip()
            is_correct = 1 if predicted.lower() == actual.lower() else 0
            if is_correct:
                correct_hits += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{riddle_id:<5} | {actual:<15} | {predicted:<20} | {status}")
        else:
            predicted = "LOW_CONFIDENCE"
            is_correct = 0

        results.append({
            "ID": riddle_id,
            "Actual": actual,
            "Layer2_Predicted": predicted,
            "Correct": is_correct,
            "Confident": 1 if prediction else 0
        })

    # 3. Final Statistics
    total = len(df)
    duration = time.time() - start_time
    hit_rate = (hits / total) * 100
    precision = (correct_hits / hits * 100) if hits > 0 else 0

    print("-" * 85)
    print(f"📊 LAYER 2 (RAG) STATISTICS:")
    print(f"📈 Confidence Rate: {hit_rate:.2f}% ({hits}/{total} met threshold)")
    print(f"🎯 Semantic Precision: {precision:.2f}% (Accuracy when RAG was confident)")
    print(f"⏱️ Total Time: {duration:.2f}s (Avg: {duration/total:.4f}s per riddle)")
    print("-" * 85)

    # Save Results
    pd.DataFrame(results).to_csv('layer2_test_results.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    test_layer2()