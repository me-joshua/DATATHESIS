import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Load and Align Data
# We merge on 'ID' to ensure we are comparing the exact same riddles
baseline_df = pd.read_csv('baseline_results.csv')
rag_df = pd.read_csv('rag_results.csv')

# Merge results on ID
combined = pd.merge(
    baseline_df[['ID', 'Actual', 'Baseline_Predicted', 'Baseline_Correct']], 
    rag_df[['ID', 'RAG_Predicted', 'RAG_Correct']], 
    on="ID"
)

# 2. Performance Metrics
base_acc = combined['Baseline_Correct'].mean() * 100
rag_acc = combined['RAG_Correct'].mean() * 100

print("\n" + "="*50)
print(f"📈 PROJECT PERFORMANCE SUMMARY")
print("="*50)
print(f"Zero-Shot Accuracy:  {base_acc:.2f}%")
print(f"RAG-Enhanced Accuracy: {rag_acc:.2f}%")
print(f"Net Improvement:       {rag_acc - base_acc:+.2f}%")
print("="*50)

# 3. Category-Wise Analysis (The "Research Gold")
# This calculates accuracy for each category individually
cat_base = combined.groupby('Actual')['Baseline_Correct'].mean() * 100
cat_rag = combined.groupby('Actual')['RAG_Correct'].mean() * 100
categories = cat_base.index.tolist()

# 4. Generate Category-Wise Comparison Bar Chart
plt.figure(figsize=(12, 6))
x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, cat_base, width, label='Zero-Shot (Baseline)', color='#ff9999', alpha=0.8)
plt.bar(x + width/2, cat_rag, width, label='RAG-Enhanced (Proposed)', color='#66b3ff', alpha=0.8)

plt.ylabel('Accuracy (%)')
plt.title('Accuracy by Riddle Category: Baseline vs. RAG', fontsize=14, fontweight='bold')
plt.xticks(x, categories)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars
for i, v in enumerate(cat_base):
    plt.text(i - width/2, v + 1, f"{v:.0f}%", ha='center', fontsize=9)
for i, v in enumerate(cat_rag):
    plt.text(i + width/2, v + 1, f"{v:.0f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('category_accuracy.png')
print("✅ Saved 'category_accuracy.png'")

# 5. Generate Confusion Matrix (RAG Version)
labels = sorted(combined['Actual'].unique())
cm = confusion_matrix(combined['Actual'], combined['RAG_Predicted'], labels=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', cbar=False)
plt.xlabel('Predicted by AI', fontsize=12)
plt.ylabel('Actual Label (Ground Truth)', fontsize=12)
plt.title('Confusion Matrix: RAG-Enhanced Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✅ Saved 'confusion_matrix.png'")

# 6. Detailed Classification Report
print("\n--- Detailed Classification Metrics (RAG) ---")
print(classification_report(combined['Actual'], combined['RAG_Predicted'], target_names=labels))

plt.show()