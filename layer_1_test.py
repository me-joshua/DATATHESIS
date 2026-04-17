import pandas as pd
import os
from hybrid_classifier import RuleBasedClassifier

def test_layer1():
    # 1. Setup Data
    if not os.path.exists('test_dataset.csv'):
        print("❌ Error: test_dataset.csv not found.")
        return

    df = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    
    # Standardize column headers and actual labels for strict comparison
    df.columns = df.columns.str.strip()
    df['Category'] = df['Category'].str.strip().str.capitalize()

    # 2. Initialize Classifier
    rule_clf = RuleBasedClassifier()
    
    results = []
    hits = 0
    correct_hits = 0

    print(f"🔬 Testing Layer 1 (Weighted Keywords) on {len(df)} samples...")
    print("-" * 95)
    print(f"{'ID':<5} | {'Actual':<15} | {'Predicted':<15} | {'Score':<10} | {'Status'}")
    print("-" * 95)

    # 3. Execution
    for idx, row in df.iterrows():
        riddle = row['Question']
        actual = row['Category']
        riddle_id = row.get('ID', idx)
        
        # Call the updated Weighted Classifier
        prediction_obj = rule_clf.classify(riddle)
        
        if prediction_obj:
            hits += 1
            predicted = prediction_obj.label
            # Capture the score from the explanation string if available, or set placeholder
            score_text = prediction_obj.explanation.split(': ')[-1].replace(')', '') if ':' in prediction_obj.explanation else "N/A"
            
            is_correct = 1 if predicted == actual else 0
            if is_correct:
                correct_hits += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{riddle_id:<5} | {actual:<15} | {predicted:<15} | {score_text:<10} | {status}")
        else:
            # For visualization, we need a label. If L1 fails, it 'defaults' to Logic or None
            predicted = "Skipped" 
            score_text = "0.00"
            is_correct = 0
            status = "⚪" # Not triggered

        results.append({
            "ID": riddle_id,
            "Actual": actual,
            "Layer1_Predicted": predicted,
            "Correct": is_correct,
            "Triggered": 1 if prediction_obj else 0,
            "Confidence_Score": score_text
        })

    # 4. Statistical Analysis
    total = len(df)
    hit_rate = (hits / total) * 100
    precision = (correct_hits / hits * 100) if hits > 0 else 0
    recall = (correct_hits / total * 100)

    print("-" * 95)
    print(f"📊 LAYER 1 STATISTICS:")
    print(f"📈 Hit Rate:  {hit_rate:.2f}% ({hits}/{total} riddles triggered Layer 1)")
    print(f"🎯 Precision: {precision:.2f}% (Accuracy of the triggered keywords)")
    print(f"🏆 Coverage:  {recall:.2f}% (Total dataset resolved correctly by Layer 1)")
    print("-" * 95)

    # Save for your thesis charts
    res_df = pd.DataFrame(results)
    res_df.to_csv('layer1_test_results.csv', index=False, encoding='utf-8-sig')
    print("📂 Detailed log saved to 'layer1_test_results.csv' for visualization.")

if __name__ == "__main__":
    test_layer1()