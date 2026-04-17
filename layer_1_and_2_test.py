import pandas as pd
import os
import time
from hybrid_classifier import RuleBasedClassifier
from retriever import get_rag_prediction

def test_hybrid_layers():
    # 1. Setup Data
    if not os.path.exists('test_dataset.csv'):
        print("❌ Error: test_dataset.csv not found.")
        return

    df = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    
    # Standardize column headers and actual labels
    df.columns = df.columns.str.strip()
    df['Category'] = df['Category'].str.strip().str.capitalize()

    # Initialize Layer 1
    rule_clf = RuleBasedClassifier()
    
    results = []
    l1_hits = 0
    l2_hits = 0
    total_correct = 0

    print(f"🚀 Testing Combined Pipeline (Layer 1 + Layer 2) on {len(df)} samples...")
    print("-" * 105)
    print(f"{'ID':<5} | {'Actual':<15} | {'Predicted':<15} | {'Method':<25} | {'Status'}")
    print("-" * 105)

    start_time = time.time()

    # 2. Execution Loop
    for idx, row in df.iterrows():
        riddle = row['Question']
        actual = row['Category']
        riddle_id = row.get('ID', idx)
        
        predicted = "Fallback"
        method = "Requires Layer 3 (LLM)"
        is_correct = 0
        status = "⚪"

        # --- STEP 1: LAYER 1 (KEYWORDS) ---
        l1_res = rule_clf.classify(riddle)
        if l1_res:
            # Standardize output to Title Case
            predicted = l1_res.label.split(' ')[0].capitalize() 
            method = "Layer 1 (Keywords)"
            l1_hits += 1
        else:
            # --- STEP 2: LAYER 2 (RAG) ---
            l2_res = get_rag_prediction(riddle)
            if l2_res:
                # Clean up and capitalize
                predicted = l2_res['label'].replace("Second Layer", "").strip().capitalize()
                method = "Layer 2 (Semantic RAG)"
                l2_hits += 1

        # 3. Validation Logic
        if predicted != "Fallback":
            if predicted == actual:
                is_correct = 1
                total_correct += 1
                status = "✅"
            else:
                is_correct = 0
                status = "❌"

        print(f"{riddle_id:<5} | {actual:<15} | {predicted:<15} | {method:<25} | {status}")

        results.append({
            "ID": riddle_id,
            "Question": riddle,
            "Actual": actual,
            "Predicted": predicted,
            "Method": method,
            "Correct": is_correct
        })

    # 4. Final Analysis
    total = len(df)
    duration = time.time() - start_time
    total_offloaded = l1_hits + l2_hits
    coverage_rate = (total_offloaded / total) * 100
    # Precision is accuracy of the local guesses only
    precision = (total_correct / total_offloaded * 100) if total_offloaded > 0 else 0

    print("-" * 105)
    print(f"📊 HYBRID PIPELINE STATISTICS (PRE-LLM):")
    print(f"✅ Layer 1 Hits: {l1_hits} riddles")
    print(f"🧠 Layer 2 Hits: {l2_hits} riddles")
    print(f"📈 Total Coverage: {coverage_rate:.2f}% ({total_offloaded}/{total} offloaded from LLM)")
    print(f"🎯 Local Precision: {precision:.2f}% (Accuracy of local layers)")
    print(f"⏱️ Total Time: {duration:.2f}s")
    print("-" * 105)

    pd.DataFrame(results).to_csv('layer_1_and_2_results.csv', index=False, encoding='utf-8-sig')
    print("📂 Results saved to 'layer_1_and_2_results.csv'")

if __name__ == "__main__":
    test_hybrid_layers()