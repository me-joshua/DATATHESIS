import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# Set visual style
sns.set_theme(style="whitegrid")

def load_and_standardize(file_path, pred_col_hint):
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    # Clean column names of spaces but keep casing for a moment
    df.columns = df.columns.str.strip()
    
    # Identify the actual and prediction columns (case-insensitive search)
    actual_col = next((c for c in df.columns if c.lower() == 'actual'), None)
    pred_col = next((c for c in df.columns if c.lower() == pred_col_hint.lower()), None)
    
    if not actual_col or not pred_col:
        print(f"⚠️ Could not find columns in {file_path}. Found: {df.columns.tolist()}")
        return None

    # Standardize values: Title Case and Strip
    df[actual_col] = df[actual_col].astype(str).str.strip().str.capitalize()
    df[pred_col] = df[pred_col].astype(str).str.strip().str.capitalize()
    
    return df, actual_col, pred_col

def run_analytics():
    # 1. Config: Mapping components to their files and column names
    configs = {
        "Baseline": {"file": "baseline_results.csv", "hint": "Baseline_Predicted"},
        "Layer 1": {"file": "layer1_test_results.csv", "hint": "Layer1_Predicted"},
        "Layer 2 Alone": {"file": "layer2_full_results.csv", "hint": "Predicted"},
        "Hybrid (L1+L2)": {"file": "rag_results.csv", "hint": "Predicted"}
    }
    
    classes = ['Logic', 'Mathematical', 'Wordplay', 'Cultural']
    final_stats = []

    print(f"🚀 Starting Final Thesis Analytics Engine...")
    print("-" * 60)

    for name, cfg in configs.items():
        result = load_and_standardize(cfg["file"], cfg["hint"])
        if result is None: continue
        
        df, actual_col, pred_col = result
        
        y_true = df[actual_col]
        y_pred = df[pred_col]

        # Calculate Accuracy and Recall
        # Note: Layer 1 'Skipped' entries are treated as incorrect
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
        weighted_recall = report['weighted avg']['recall']
        
        final_stats.append({
            "Component": name,
            "Accuracy": acc,
            "Recall": weighted_recall
        })

        # --- GENERATE CONFUSION MATRIX ---
        # (We skip Layer 1 CM as it's dominated by 'Skipped' values)
        if name != "Layer 1":
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral_r', xticklabels=classes, yticklabels=classes)
            plt.title(f"Confusion Matrix: {name}", fontsize=14, fontweight='bold')
            plt.ylabel('Actual Category')
            plt.xlabel('Predicted Category')
            plt.tight_layout()
            plt.savefig(f"cm_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
            plt.close()

    # 2. ACCURACY COMPARISON BAR CHART
    stats_df = pd.DataFrame(final_stats)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Component", y="Accuracy", data=stats_df, hue="Component", palette="viridis", legend=False)
    plt.ylim(0, 1.1)
    plt.title("Accuracy Comparison: All Layers vs Baseline", fontsize=15, fontweight='bold')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, fontweight='bold', xytext=(0, 9), textcoords='offset points')
    
    plt.savefig("final_accuracy_comparison.png")
    plt.close()

    # 3. PRINT ANALYTICS SUMMARY
    print(f"{'COMPONENT':<20} | {'ACCURACY':<12} | {'RECALL':<10}")
    print("-" * 60)
    for s in final_stats:
        print(f"{s['Component']:<20} | {s['Accuracy']*100:>10.2f}% | {s['Recall']*100:>8.2f}%")
    print("-" * 60)
    print("✅ Confusion Matrices saved as PNGs.")
    print("✅ Accuracy Comparison chart saved as final_accuracy_comparison.png.")

if __name__ == "__main__":
    generate_advanced_stats = run_analytics()