import pandas as pd
import ast
from datetime import datetime

# --- Load JSONL files ---
def load_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

if __name__ == "__main__":
    reviews_df = load_jsonl("data/Appliances.jsonl")
    meta_df = load_jsonl("data/meta_Appliances.jsonl")

    # --- Quick overview ---
    print("Reviews shape:", reviews_df.shape)
    print("Reviews columns:", reviews_df.columns.tolist())
    print("Metadata shape:", meta_df.shape)
    print("Metadata columns:", meta_df.columns.tolist())

    # --- Check timestamp span ---
    reviews_df['review_date'] = pd.to_datetime(reviews_df['timestamp'], unit='ms')
    print("Review date range:", reviews_df['review_date'].min(), "to", reviews_df['review_date'].max())

    # --- Check rating distribution ---
    print("Rating distribution:\n", reviews_df['rating'].value_counts())

    # --- Check main categories ---
    print("Metadata main categories:\n", meta_df['main_category'].value_counts())

    # --- Create representative samples ---
    
    # Create year and category columns for stratified sampling
    reviews_df['year'] = reviews_df['review_date'].dt.year
    
    # Store original column names before merging
    original_reviews_cols = reviews_df.columns.tolist()
    
    # Merge with metadata to get categories for stratified sampling
    temp_merged = pd.merge(reviews_df, meta_df, left_on="asin", right_on="parent_asin", how="left", suffixes=('', '_meta'))
    
    # --- 1. TINY TEST SAMPLE (50 records) - For quick testing ---
    tiny_sample = temp_merged.sample(n=50, random_state=42)
    tiny_reviews = tiny_sample[original_reviews_cols]
    tiny_meta = meta_df[meta_df['parent_asin'].isin(tiny_reviews['asin'])].drop_duplicates()
    
    tiny_reviews.to_csv("data/reviews_tiny_sample.csv", index=False)
    tiny_meta.to_csv("data/meta_tiny_sample.csv", index=False)
    print(f"Tiny test sample saved: {len(tiny_reviews)} reviews")
    
    # --- 2. REPRESENTATIVE 10K SAMPLE (Stratified by year and category) ---
    # Get proportional samples from each year and category combination
    sample_size = 10000
    
    # Calculate proportions
    year_category_counts = temp_merged.groupby(['year', 'main_category']).size().reset_index(name='count')
    year_category_counts['proportion'] = year_category_counts['count'] / len(temp_merged)
    year_category_counts['target_sample'] = (year_category_counts['proportion'] * sample_size).round().astype(int)
    
    # Ensure we don't exceed available data
    year_category_counts['target_sample'] = year_category_counts.apply(
        lambda row: min(row['target_sample'], row['count']), axis=1
    )
    
    # Sample from each group
    sampled_groups = []
    for _, group_info in year_category_counts.iterrows():
        year, category, target = group_info['year'], group_info['main_category'], group_info['target_sample']
        if target > 0:
            group_data = temp_merged[
                (temp_merged['year'] == year) & 
                (temp_merged['main_category'] == category)
            ]
            if len(group_data) > 0:
                sample = group_data.sample(n=min(target, len(group_data)), random_state=42)
                sampled_groups.append(sample)
    
    # Combine all samples
    representative_sample = pd.concat(sampled_groups, ignore_index=True)
    
    # If we're under 10k, fill with random samples
    if len(representative_sample) < sample_size:
        remaining = sample_size - len(representative_sample)
        excluded_data = temp_merged[~temp_merged.index.isin(representative_sample.index)]
        if len(excluded_data) > 0:
            additional_sample = excluded_data.sample(n=min(remaining, len(excluded_data)), random_state=42)
            representative_sample = pd.concat([representative_sample, additional_sample], ignore_index=True)
    
    # Extract reviews and metadata
    rep_reviews = representative_sample[original_reviews_cols]
    rep_meta = meta_df[meta_df['parent_asin'].isin(rep_reviews['asin'])].drop_duplicates()
    
    rep_reviews.to_csv("data/reviews_10k_sample.csv", index=False)
    rep_meta.to_csv("data/meta_10k_sample.csv", index=False)
    
    print(f"Representative 10k sample saved: {len(rep_reviews)} reviews")
    print("Year distribution in 10k sample:")
    print(rep_reviews['year'].value_counts().sort_index())
    
    # --- 3. LEGACY 500 SAMPLE (for backward compatibility) ---
    reviews_sample = reviews_df.sample(n=500, random_state=42)
    meta_sample = meta_df[meta_df['parent_asin'].isin(reviews_sample['asin'])]
    reviews_sample.to_csv("data/reviews_sample.csv", index=False)
    meta_sample.to_csv("data/meta_sample.csv", index=False)
    
    print("All sample files saved successfully!")
