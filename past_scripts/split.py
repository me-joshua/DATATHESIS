import pandas as pd
from sklearn.model_selection import train_test_split
import os

def perform_split(input_file, train_size=0.8):
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    # 1. Load your dataset
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    # 2. Perform the split (80% Train, 20% Test)
    # random_state=42 ensures you get the same split every time you run it
    # stratify=df['Category'] keeps the proportion of categories identical in both sets
    train_df, test_df = train_test_split(
        df, 
        train_size=train_size, 
        random_state=42, 
        stratify=df['Category']
    )

    # 3. Save the resulting files
    train_df.to_csv('train_dataset.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test_dataset.csv', index=False, encoding='utf-8-sig')

    # 4. Print Summary
    print(f"✅ Split Complete!")
    print(f"📂 Total Samples: {len(df)}")
    print(f"📝 Training Set:  {len(train_df)} rows (Saved to 'train_dataset.csv')")
    print(f"🧪 Testing Set:   {len(test_df)} rows (Saved to 'test_dataset.csv')")
    
    # Check category distribution
    print("\n📊 Category Distribution in Test Set:")
    print(test_df['Category'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    # Replace with your master filename
    perform_split('master_dataset.csv')