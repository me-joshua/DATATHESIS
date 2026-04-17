import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

# 1. Initialization
df = pd.read_csv('train_dataset.csv', encoding='utf-8-sig')
df.columns = df.columns.str.strip()
df['Category'] = df['Category'].str.strip().str.capitalize()

index = faiss.read_index('riddles.index')
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def get_similar_riddles(user_query, k=3):
    """Retrieves context for reasoning and UI display."""
    refined_query = f"Question: {str(user_query).strip()}"
    query_embedding = model.encode([refined_query]).astype('float32')
    faiss.normalize_L2(query_embedding) 
    
    distances, indices = index.search(query_embedding, k=15)
    
    context_list = []
    seen_categories = {} 
    seen_questions = set()

    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: continue
        if dist < 0.35: continue 
        
        match = df.iloc[idx]
        question = str(match['Question']).strip()
        category = str(match['Category']).strip().capitalize()
        answer = str(match['Answer']).strip()
        
        if question.lower() == str(user_query).strip().lower():
            continue
            
        if question not in seen_questions:
            if seen_categories.get(category, 0) < 2:
                context_list.append({
                    "Question": question,
                    "Answer": answer,
                    "Category": category,
                    "Score": float(dist)
                })
                seen_questions.add(question)
                seen_categories[category] = seen_categories.get(category, 0) + 1
        
        if len(context_list) >= k:
            break
    
    return context_list

def get_rag_prediction(user_query, threshold=0.90):
    """Layer 2: High-Confidence Semantic Filter."""
    matches = get_similar_riddles(user_query, k=3)
    if not matches:
        return None

    best_match = matches[0]
    
    # 1. Direct High-Similarity Hit
    if best_match['Score'] >= threshold:
        return {
            "label": best_match['Category'],
            "method": "Layer 2 (Confident Match)",
            "explanation": f"Strong semantic match found (Score: {best_match['Score']:.2f})"
        }

    # 2. Unanimous Cluster Vote
    categories = [m['Category'] for m in matches]
    if len(categories) == 3 and len(set(categories)) == 1:
        return {
            "label": categories[0],
            "method": "Layer 2 (Unanimous Vote)",
            "explanation": "Top 3 semantic neighbors are identical."
        }

    return None

def get_best_rag_match(user_query, top_k=5):
    """
    NEW: Weighted Semantic Voter (The Final Judge).
    Uses Cosine Similarity as a weight to decide between competing neighbors.
    This logic is what boosted the Tamil script to 90% accuracy.
    """
    refined_query = f"Question: {str(user_query).strip()}"
    query_embedding = model.encode([refined_query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Fetch more neighbors for a broader vote
    distances, indices = index.search(query_embedding, top_k)
    
    score_map = {'Logic': 0.0, 'Mathematical': 0.0, 'Wordplay': 0.0, 'Cultural': 0.0}
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: continue
        label = df.iloc[idx]['Category'].strip().capitalize()
        # The weight is the similarity score (closer neighbors count more)
        score_map[label] += float(dist)

    # Return the label with the highest weighted sum
    final_label = max(score_map, key=score_map.get)
    return final_label