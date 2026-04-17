import pandas as pd
import os
from hybrid_classifier import RuleBasedClassifier

def test_layer1():
    # 1. Setup Data
    if not os.path.exists('test_dataset.csv'):
        print("❌ Error: test_dataset.csv not found. Please run your split script first.")
        return

    df = pd.read_csv('test_dataset.csv', encoding='utf-8-sig')
    
    # Ensure standard capitalization for comparison
    df['Category'] = df['Category'].str.strip().str.capitalize()

    # 2. Initialize Classifier
    rule_clf = RuleBasedClassifier()
    
    results = []
    hits = 0
    correct_hits = 0

    print(f"🔬 Testing Layer 1 (Keyword-Based) on {len(df)} samples...")
    print("-" * 80)
    print(f"{'ID':<5} | {'Actual':<15} | {'Predicted':<15} | {'Status'}")
    print("-" * 80)

    # 3. Execution
    for idx, row in df.iterrows():
        riddle = row['Question']
        actual = row['Category']
        riddle_id = row.get('ID', idx)
        
        # Call ONLY Layer 1
        prediction_obj = rule_clf.classify(riddle)
        
        if prediction_obj:
            hits += 1
            predicted = prediction_obj.label
            is_correct = 1 if predicted == actual else 0
            if is_correct:
                correct_hits += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{riddle_id:<5} | {actual:<15} | {predicted:<15} | {status}")
        else:
            predicted = "None (Skipped)"
            is_correct = 0
            # We don't print skipped ones to keep the console clean, 
            # but we track them in results.

        results.append({
            "ID": riddle_id,
            "Actual": actual,
            "Layer1_Predicted": predicted,
            "Correct": is_correct,
            "Triggered": 1 if prediction_obj else 0
        })

    # 4. Statistical Analysis
    total = len(df)
    hit_rate = (hits / total) * 100
    # Precision: Accuracy specifically when a keyword was found
    precision = (correct_hits / hits * 100) if hits > 0 else 0
    # Recall: How much of the total dataset did Layer 1 solve correctly?
    recall = (correct_hits / total * 100)

    print("-" * 80)
    print(f"📊 LAYER 1 STATISTICS:")
    print(f"📈 Hit Rate: {hit_rate:.2f}% ({hits}/{total} riddles matched keywords)")
    print(f"🎯 Precision: {precision:.2f}% (Accuracy of the hits)")
    print(f"🏆 Total Recall: {recall:.2f}% (Total dataset solved by Layer 1)")
    print("-" * 80)

    # Save for your thesis charts
    res_df = pd.DataFrame(results)
    res_df.to_csv('layer1_test_results.csv', index=False, encoding='utf-8-sig')
    print("📂 Detailed log saved to 'layer1_test_results.csv'")

if __name__ == "__main__":
    test_layer1()