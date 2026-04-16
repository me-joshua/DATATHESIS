import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Must match the upgraded model from create_index.py
df = pd.read_csv('master_dataset.csv', encoding='utf-8-sig')
index = faiss.read_index('riddles.index')
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def get_similar_riddles(user_query, k=3):
    # 2. Encode and Normalize the query vector
    # This is critical for Cosine Similarity to work
    query_embedding = model.encode([str(user_query)]).astype('float32')
    faiss.normalize_L2(query_embedding) 
    
    # 3. Search the index (IndexFlatIP returns Inner Product, which is Cosine now)
    distances, indices = index.search(query_embedding, k=10)
    
    context_list = []
    seen_questions = set()

    for idx in indices[0]:
        # FAISS returns -1 if no match is found
        if idx == -1: continue
        
        match = df.iloc[idx]
        question = match['Question']
        
        # Self-Exclusion logic
        if question.strip().lower() == user_query.strip().lower():
            continue
            
        if question not in seen_questions:
            context_list.append({
                "Question": question,
                "Category": match['Category']
            })
            seen_questions.add(question)
        
        if len(context_list) >= k:
            break
    
    return context_list