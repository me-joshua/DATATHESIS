import pandas as pd
import os

# Define folders and files
data_dir = 'data' # Change this if your CSVs are in the root
files = {
    'Malayalam': ['mal_logic.csv', 'mal_cultural.csv', 'mal_math.csv', 'mal_wordplay.csv'],
    'Tamil': ['tam_logic.csv', 'tam_cultural.csv', 'tam_math.csv', 'tam_wordplay.csv']
}

all_dfs = []

for lang, file_list in files.items():
    for file in file_list:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            temp_df = pd.read_csv(file_path, encoding='utf-8-sig')
            temp_df['Language'] = lang # Add metadata for RAG retrieval
            all_dfs.append(temp_df)

# Merge and save
master_df = pd.concat(all_dfs, ignore_index=True)
master_df.to_csv('master_dataset.csv', index=False, encoding='utf-8-sig')

print(f"✅ Master dataset created with {len(master_df)} total riddles.")