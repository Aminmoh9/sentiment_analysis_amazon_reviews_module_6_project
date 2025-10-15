# [file name]: src/data_utils.py
"""
Consolidated data utilities
"""
import pandas as pd
import ast

def map_rating_to_sentiment(rating):
    """Map numerical ratings to sentiment categories."""
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"
    else:
        return "Unknown"

def extract_image_url(images_data):
    """Extract image URL from images data."""
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

def clean_dataset_for_analysis(df):
    """Quick cleaning function for already processed datasets."""
    df_clean = df.copy()
    
    # Ensure datetime
    if 'review_date' in df_clean.columns:
        df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], errors='coerce')
        df_clean['year'] = df_clean['review_date'].dt.year
    
    # Ensure sentiment is consistent
    if 'rating' in df_clean.columns and 'sentiment' not in df_clean.columns:
        df_clean['sentiment'] = df_clean['rating'].apply(map_rating_to_sentiment)
    
    # Remove records with critical missing data
    critical_columns = ['review_date', 'rating', 'main_category']
    df_clean = df_clean.dropna(subset=[col for col in critical_columns if col in df_clean.columns])
    
    return df_clean