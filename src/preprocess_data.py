# [file name]: src/preprocess_data.py
"""
TRUE Full Dataset Processing - Creates ONLY 10K representative sample
"""
import pandas as pd
import json
import ast
from datetime import datetime
import numpy as np

def load_jsonl_efficiently(file_path, max_records=None):
    """Load JSONL file efficiently with optional record limit"""
    records = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)

def extract_image_url(images_data):
    """Extract the first valid image URL from images data"""
    if images_data is None or (isinstance(images_data, float) and pd.isna(images_data)):
        return None
    
    if hasattr(images_data, '__len__') and not isinstance(images_data, (str, dict, list)):
        if len(images_data) == 0:
            return None
        images_data = images_data[0] if len(images_data) > 0 else None
    
    if not images_data:
        return None
    
    try:
        if isinstance(images_data, str):
            if images_data.startswith('http'):
                return images_data
            elif images_data.startswith('['):
                images = ast.literal_eval(images_data)
            else:
                return None
        elif isinstance(images_data, list):
            images = images_data
        else:
            return None
            
        if isinstance(images, list) and len(images) > 0:
            first_image = images[0]
            if isinstance(first_image, dict):
                for key in ['large', 'hi_res', 'thumb']:
                    if key in first_image and first_image[key]:
                        url = first_image[key]
                        if isinstance(url, str) and url.startswith(('http://', 'https://')):
                            return url
            elif isinstance(first_image, str) and first_image.startswith(('http://', 'https://')):
                return first_image
    except Exception:
        pass
    
    return None

def clean_text_field(text):
    """Clean text fields"""
    if pd.isna(text) or not str(text).strip() or str(text).lower() in ['nan', 'none', 'null', '']:
        return ""
    return str(text).strip()

def validate_rating(rating):
    """Ensure rating is valid (1-5)"""
    try:
        rating_val = float(rating)
        if 1 <= rating_val <= 5:
            return rating_val
    except (ValueError, TypeError):
        pass
    return None

def extract_price(price_val):
    """Extract clean price value"""
    if pd.isna(price_val) or str(price_val).strip() == '':
        return None
    
    try:
        price_str = str(price_val).replace(',', '').replace('$', '').strip()
        if price_str == '' or price_str == 'nan':
            return None
        price_float = float(price_str)
        if 0.01 <= price_float <= 100000:
            return f"${price_float:.2f}"
    except (ValueError, TypeError):
        pass
    
    return None

def assign_sentiment(rating):
    """Assign sentiment based on rating"""
    if pd.isna(rating):
        return "Unknown"
    
    rating = float(rating)
    if rating >= 4:
        return "Positive"
    elif rating >= 3:
        return "Neutral"
    else:
        return "Negative"

def add_time_features(df, date_column='timestamp'):
    """Add time-based features from timestamp"""
    df = df.copy()
    
    try:
        if date_column not in df.columns:
            print(f"   Warning: {date_column} column not found, using current date")
            df['review_date'] = datetime.now()
        else:
            df[date_column] = pd.to_numeric(df[date_column], errors='coerce')
            valid_timestamps = df[date_column].notna()
            if valid_timestamps.sum() == 0:
                print(f"   Warning: No valid timestamps found, using current date")
                df['review_date'] = datetime.now()
            else:
                df['review_date'] = pd.to_datetime(df[date_column] / 1000, unit='s', errors='coerce')
                df['review_date'] = df['review_date'].fillna(datetime.now())
    except Exception as e:
        print(f"   Warning: Date conversion failed ({e}), using current date")
        df['review_date'] = datetime.now()
    
    # Add time features
    df['year'] = df['review_date'].dt.year
    df['month'] = df['review_date'].dt.month
    df['quarter'] = df['review_date'].dt.quarter
    df['weekday'] = df['review_date'].dt.dayofweek
    df['is_weekend'] = df['weekday'].isin([5, 6])
    df['review_age_days'] = (datetime.now() - df['review_date']).dt.days
    
    return df

def create_stratified_sample(df, sample_size=10000, random_state=42):
    """
    Create stratified sample maintaining distribution by year and category
    Sample ~10 records from each year-category combination
    """
    
    if len(df) <= sample_size:
        return df
    
    print(f"   Creating stratified sample of {sample_size} records...")
    
    try:
        # Group by year and category
        samples = []
        
        # Calculate target per group (aim for balanced representation)
        year_category_combinations = df.groupby(['year', 'main_category']).size()
        total_combinations = len(year_category_combinations)
        
        # Target ~10 records per combination, but adjust for sample size
        target_per_group = max(5, min(20, sample_size // total_combinations))
        
        print(f"   Sampling ~{target_per_group} records from each of {total_combinations} year-category combinations")
        
        for (year, category), group_df in df.groupby(['year', 'main_category']):
            group_size = min(target_per_group, len(group_df))
            if group_size > 0:
                sample = group_df.sample(n=group_size, random_state=random_state)
                samples.append(sample)
        
        # Combine all samples
        result = pd.concat(samples, ignore_index=True)
        
        # If we're under sample_size, fill with random samples
        if len(result) < sample_size:
            remaining = sample_size - len(result)
            excluded_data = df[~df.index.isin(result.index)]
            if len(excluded_data) > 0:
                additional_sample = excluded_data.sample(n=min(remaining, len(excluded_data)), random_state=random_state)
                result = pd.concat([result, additional_sample], ignore_index=True)
        
        # Final adjustment if needed
        if len(result) > sample_size:
            result = result.sample(n=sample_size, random_state=random_state)
        
        print(f"   ✅ Final sample: {len(result):,} records")
        print(f"   ✅ Year distribution: {result['year'].value_counts().to_dict()}")
        print(f"   ✅ Category distribution: {result['main_category'].nunique()} categories")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Stratified sampling failed: {e}, using random sampling")
        return df.sample(n=sample_size, random_state=random_state)

def process_true_full_dataset():
    """Process the actual full JSONL dataset and create 10K sample only"""
    
    print("🚀 Processing Amazon Appliances Dataset - Creating 10K Sample")
    print("=" * 60)
    
    # Step 1: Load datasets
    print("\n📁 Step 1: Loading JSONL datasets...")
    
    try:
        # Load reviews (limit to 100K for efficiency)
        print("   Loading reviews...")
        reviews_df = load_jsonl_efficiently("data/Appliances.jsonl", max_records=100000)
        print(f"   ✅ Reviews loaded: {len(reviews_df):,} records")
        
        # Load metadata  
        print("   Loading product metadata...")
        meta_df = load_jsonl_efficiently("data/meta_Appliances.jsonl")
        print(f"   ✅ Metadata loaded: {len(meta_df):,} records")
        
    except Exception as e:
        print(f"   ❌ Error loading JSONL files: {e}")
        return None
    
    # Step 2: Clean reviews data
    print("\n🧹 Step 2: Cleaning reviews data...")
    reviews_clean = reviews_df.copy()
    original_count = len(reviews_clean)
    
    # Validate ratings
    reviews_clean['rating'] = reviews_clean['rating'].apply(validate_rating)
    reviews_clean = reviews_clean.dropna(subset=['rating'])
    print(f"   ✅ Valid ratings: {len(reviews_clean):,} / {original_count:,}")
    
    # Clean text fields
    reviews_clean['title'] = reviews_clean['title'].apply(clean_text_field)
    reviews_clean['text'] = reviews_clean['text'].apply(clean_text_field)
    
    # Remove empty reviews
    reviews_clean = reviews_clean[
        (reviews_clean['title'] != "") | (reviews_clean['text'] != "")
    ]
    print(f"   ✅ With content: {len(reviews_clean):,}")
    
    # Add sentiment and clean numeric fields
    reviews_clean['sentiment'] = reviews_clean['rating'].apply(assign_sentiment)
    reviews_clean['helpful_vote'] = pd.to_numeric(reviews_clean['helpful_vote'], errors='coerce').fillna(0)
    
    # Add time features
    reviews_clean = add_time_features(reviews_clean, 'timestamp')
    print(f"   ✅ With valid dates: {len(reviews_clean):,}")
    
    # Step 3: Clean metadata
    print("\n🧹 Step 3: Cleaning product metadata...")
    meta_clean = meta_df.copy()
    original_meta_count = len(meta_clean)
    
    # Clean titles and categories
    meta_clean['title'] = meta_clean['title'].apply(clean_text_field)
    meta_clean = meta_clean[meta_clean['title'] != ""]
    print(f"   ✅ With titles: {len(meta_clean):,} / {original_meta_count:,}")
    
    meta_clean['main_category'] = meta_clean['main_category'].apply(clean_text_field)
    meta_clean = meta_clean[meta_clean['main_category'] != ""]
    print(f"   ✅ With categories: {len(meta_clean):,}")
    
    # Extract images and prices
    meta_clean['main_image'] = meta_clean['images'].apply(extract_image_url)
    images_count = meta_clean['main_image'].notna().sum()
    print(f"   ✅ With valid images: {images_count:,}")
    
    meta_clean['price_clean'] = meta_clean['price'].apply(extract_price)
    price_count = meta_clean['price_clean'].notna().sum()
    print(f"   ✅ With valid prices: {price_count:,}")
    
    # Step 4: Merge datasets
    print("\n🔗 Step 4: Merging cleaned datasets...")
    
    merged_df = pd.merge(
        reviews_clean,
        meta_clean,
        left_on="asin",
        right_on="parent_asin",
        how="inner",
        suffixes=('_review', '_product')
    )
    
    print(f"   ✅ Complete merged records: {len(merged_df):,}")
    
    # Create final structure
    final_df = pd.DataFrame({
        # Review information
        'asin': merged_df['asin'],
        'rating': merged_df['rating'],
        'review_title': merged_df['title_review'],
        'review_text': merged_df['text'],
        'review_date': merged_df['review_date'],
        'helpful_vote': merged_df['helpful_vote'],
        'sentiment': merged_df['sentiment'],
        'user_id': merged_df['user_id'],
        'verified_purchase': merged_df.get('verified_purchase', True),
        
        # Time features
        'year': merged_df['year'],
        'month': merged_df['month'],
        'quarter': merged_df['quarter'],
        'weekday': merged_df['weekday'],
        'is_weekend': merged_df['is_weekend'],
        'review_age_days': merged_df['review_age_days'],
        
        # Product information
        'product_title': merged_df['title_product'],
        'main_category': merged_df['main_category'],
        'main_image': merged_df['main_image'],
        'price': merged_df['price_clean'],
        'average_rating': pd.to_numeric(merged_df.get('average_rating', 0), errors='coerce'),
        'rating_number': pd.to_numeric(merged_df.get('rating_number', 0), errors='coerce'),
        'parent_asin': merged_df['parent_asin_product']
    })
    
    # Step 5: Create 10K stratified sample only
    print("\n📊 Step 5: Creating 10K stratified sample...")
    
    sample_10k = create_stratified_sample(final_df, 10000)
    print(f"   ✅ 10K sample created: {len(sample_10k):,} records")
    print(f"   ✅ Categories: {sample_10k['main_category'].nunique()}")
    print(f"   ✅ Years: {sample_10k['year'].min()}-{sample_10k['year'].max()}")
    
    # Step 6: Save ONLY the 10K sample
    print("\n💾 Step 6: Saving 10K sample...")
    
    sample_10k.to_csv("data/appliances_10k_sample.csv", index=False)
    print(f"   ✅ Saved appliances_10k_sample.csv ({len(sample_10k):,} records)")
    
    # Step 7: Final report
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 **10K Representative Sample Created:**")
    print(f"   • Records: {len(sample_10k):,}")
    print(f"   • Date range: {sample_10k['review_date'].min().strftime('%Y-%m-%d')} to {sample_10k['review_date'].max().strftime('%Y-%m-%d')}")
    print(f"   • Categories: {sample_10k['main_category'].nunique()}")
    print(f"   • Positive sentiment: {(sample_10k['sentiment'] == 'Positive').mean()*100:.1f}%")
    
    print(f"\n🚀 **Ready to run: streamlit run app.py**")
    
    return sample_10k

if __name__ == "__main__":
    sample_10k = process_true_full_dataset()