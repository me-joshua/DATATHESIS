import pandas as pd
import os

def fix_csv_formatting(file_path):
    if not os.path.exists(file_path):
        return
    
    # Load with utf-8-sig to handle Tamil/Malayalam characters
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 1. Fix Column Names (Ensure 'Category', 'Question', 'Answer', 'ID')
    df.columns = df.columns.str.strip().str.capitalize()
    
    # 2. Fix Category Values (wordplay -> Wordplay, logical -> Logic, etc.)
    # We map them to the exact 4 labels used in your generator/eval scripts
    category_map = {
        'wordplay': 'Wordplay',
        'logical': 'Logic',
        'logic': 'Logic',
        'cultural': 'Cultural',
        'mathematical': 'Mathematical',
        'math': 'Mathematical'
    }
    
    if 'Category' in df.columns:
        # Convert to lowercase first to ensure match, then map to Title Case
        df['Category'] = df['Category'].astype(str).str.strip().str.lower().map(category_map).fillna(df['Category'])
        
    # 3. Save it back
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"✅ Fixed formatting for {file_path}")

# Run for all your core files
files_to_fix = ['master_dataset.csv', 'train_dataset.csv', 'test_dataset.csv']
for f in files_to_fix:
    fix_csv_formatting(f)