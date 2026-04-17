import pandas as pd
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# 1. Load and Clean
if not os.path.exists('train_dataset.csv'):
    raise FileNotFoundError("❌ train_dataset.csv missing. Run your split script first.")

df = pd.read_csv('train_dataset.csv', encoding='utf-8-sig')

# Standardize column headers and Category labels
df.columns = df.columns.str.strip()
df['Category'] = df['Category'].str.strip().str.capitalize()

# 2. Model Initialization
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 3. Contextual Feature Extraction
# Matches the "Question: " prefix used in retriever.py
def create_feature_string(row):
    q = str(row['Question']).strip()
    a = str(row['Answer']).strip()
    return f"Question: {q} | Answer: {a}"

print(f"🚀 Encoding {len(df)} riddles...")
feature_list = df.apply(create_feature_string, axis=1).tolist()
embeddings = model.encode(feature_list, batch_size=32, show_progress_bar=True)

# 4. Normalization for Cosine Similarity
faiss.normalize_L2(embeddings) 

# 5. Build and Save Index
data_to_add = np.ascontiguousarray(embeddings).astype('float32')
index = faiss.IndexFlatIP(data_to_add.shape[1]) 
index.add(data_to_add)

faiss.write_index(index, 'riddles.index')
print(f"✅ Successfully created 'riddles.index'.")