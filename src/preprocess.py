"""
Data preprocessing utilities for Amazon Appliances review dataset.
This module handles data cleaning, sentiment mapping, and metadata processing
for the RAG chatbot and sentiment analysis project.
"""

import pandas as pd
import ast

def map_rating_to_sentiment(rating):
    """
    Map numerical ratings to sentiment categories.
    
    Args:
        rating (int): Numerical rating from 1-5
        
    Returns:
        str: Sentiment category ('Negative', 'Neutral', 'Positive', 'Unknown')
        
    Examples:
        >>> map_rating_to_sentiment(1)
        'Negative'
        >>> map_rating_to_sentiment(4)
        'Positive'
    """
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    elif rating in [4, 5]:
        return "Positive"
    else:
        return "Unknown"

def preprocess_reviews(reviews_df):
    """
    Clean and preprocess reviews dataframe with sentiment mapping and date handling.
    
    Args:
        reviews_df (pd.DataFrame): Raw reviews dataframe with columns like 'rating', 'timestamp'
        
    Returns:
        pd.DataFrame: Processed dataframe with added 'sentiment' and 'review_date' columns
        
    Processing steps:
        1. Maps ratings to sentiment categories
        2. Converts timestamps to datetime objects
        3. Handles missing or malformed date data
    """
    reviews_df = reviews_df.copy()
    reviews_df['sentiment'] = reviews_df['rating'].apply(map_rating_to_sentiment)
    
    # Handle review_date - check if it's already processed or needs conversion
    if 'review_date' not in reviews_df.columns:
        if 'timestamp' in reviews_df.columns:
            # Try to convert timestamp - handle both numeric and string formats
            try:
                reviews_df['review_date'] = pd.to_datetime(reviews_df['timestamp'], unit='ms')
            except (ValueError, TypeError):
                # If timestamp is already a date string, parse it directly
                reviews_df['review_date'] = pd.to_datetime(reviews_df['timestamp'])
        else:
            # Create a dummy date if no timestamp available
            reviews_df['review_date'] = pd.to_datetime('2020-01-01')
    else:
        # review_date already exists, ensure it's datetime
        reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
    
    return reviews_df

def preprocess_meta(meta_df):
    """
    Extract and clean metadata with focus on product images.
    
    Args:
        meta_df (pd.DataFrame): Raw metadata dataframe with 'images' column containing JSON strings
        
    Returns:
        pd.DataFrame: Processed dataframe with added 'main_image' column containing primary image URLs
        
    Processing steps:
        1. Parses JSON image data from strings
        2. Extracts main product images (MAIN variant preferred)
        3. Falls back to first available image if no MAIN variant exists
    """
    meta_df = meta_df.copy()
    main_images = []
    for img_str in meta_df['images']:
        try:
            img_list = ast.literal_eval(img_str)
            main_image = None
            for img in img_list:
                if img.get('variant') == 'MAIN':
                    main_image = img.get('hi_res')
                    break
            if not main_image and img_list:
                main_image = img_list[0].get('hi_res')
            main_images.append(main_image)
        except:
            main_images.append(None)
    meta_df['main_image'] = main_images
    return meta_df

# Script execution part (only runs when file is executed directly)
# Usage: python src/preprocess.py (from project root directory)
if __name__ == "__main__":
    # --- Load CSVs ---
    reviews_df = pd.read_csv("data/reviews_sample.csv")
    meta_df = pd.read_csv("data/meta_sample.csv")

    # --- Preprocess data ---
    reviews_df = preprocess_reviews(reviews_df)
    meta_df = preprocess_meta(meta_df)

    # --- Merge reviews with metadata on ASIN ---
    df = pd.merge(reviews_df, meta_df, left_on="asin", right_on="parent_asin", how="left")

    # --- Save preprocessed CSV ---
    df.to_csv("data/preprocessed_reviews.csv", index=False)
    print("Preprocessed data saved as preprocessed_reviews.csv")
