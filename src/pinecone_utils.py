# [file name]: src/pinecone_utils.py
"""
Pinecone vector store utilities for the RAG system
"""
import pandas as pd
import numpy as np
import os
import time
import json
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def init_pinecone(api_key):
    """Initialize Pinecone client with API key"""
    pc = Pinecone(api_key=api_key)
    return pc

def create_vector_store(df, index_name="amazon-reviews", force_recreate=False):
    """Create Pinecone vector store from DataFrame"""
    # Initialize Pinecone client
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = init_pinecone(api_key)
    
    # Delete existing index if force_recreate is True
    if force_recreate and index_name in pc.list_indexes().names():
        print(f"Deleting existing index '{index_name}' to recreate with fresh data...")
        pc.delete_index(index_name)
        time.sleep(5)  # Wait for deletion to complete
    
    # Check if index exists, create if it doesn't
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Prepare metadata - clean NaN values for Pinecone
    def clean_value(value):
        """Convert NaN, None, and other problematic values to empty string"""
        # Check for various types of NaN/null values
        if (pd.isna(value) or 
            value is None or 
            str(value).lower() in ['nan', 'none', 'null'] or
            (isinstance(value, float) and np.isnan(value))):
            return ""
        
        # Handle numeric values
        if isinstance(value, (int, float)) and pd.notna(value):
            return str(value)
        
        return str(value).strip()
    
    metadata_list = []
    
    # Create texts from review title and review text
    review_title = df["review_title"].fillna("") if "review_title" in df.columns else pd.Series([""] * len(df))
    review_text = df["review_text"].fillna("") if "review_text" in df.columns else pd.Series([""] * len(df))
    product_title = df["product_title"].fillna("") if "product_title" in df.columns else pd.Series([""] * len(df))
    
    # Combine review title + review text for better semantic search
    texts = []
    for title, text, prod_title in zip(review_title, review_text, product_title):
        combined_text = f"Product: {prod_title}. Review: {title}. {text}"
        # Ensure we have some content
        if combined_text.strip() in ["", ".", "Product: . Review: ."]:
            combined_text = "No review content available"
        texts.append(combined_text)
        
        # Prepare metadata for this row
        price_value = df.loc[df.index[texts.index(combined_text)], "price"] if len(texts) <= len(df) else ""
        if pd.notna(price_value) and str(price_value).strip() and str(price_value).lower() not in ['nan', 'none', 'null']:
            price_clean = str(price_value).strip()
        else:
            price_clean = ""
        
        # Use main_image if available, otherwise empty string
        image_url = df.loc[df.index[texts.index(combined_text)], "main_image"] if "main_image" in df.columns and len(texts) <= len(df) else ""
        
        metadata_list.append({
            "asin": clean_value(df.loc[df.index[texts.index(combined_text)], "asin"]),
            "product_title": clean_value(prod_title),
            "review_title": clean_value(title),
            "main_category": clean_value(df.loc[df.index[texts.index(combined_text)], "main_category"]),
            "image": clean_value(image_url),
            "review_date": df.loc[df.index[texts.index(combined_text)], "review_date"].strftime("%Y-%m-%d") if pd.notna(df.loc[df.index[texts.index(combined_text)], "review_date"]) else "",
            "sentiment": clean_value(df.loc[df.index[texts.index(combined_text)], "sentiment"]),
            "rating": clean_value(df.loc[df.index[texts.index(combined_text)], "rating"]),
            "price": price_clean
        })
    
    # Debug: Check data quality
    print(f"Creating vector store with {len(texts)} texts")
    print(f"Sample text: {texts[0][:100]}...")
    print(f"Sample metadata keys: {list(metadata_list[0].keys())}")
    
    # Additional check for NaN values in metadata
    try:
        # Test serialize first few metadata entries to check for NaN
        for i, metadata in enumerate(metadata_list[:3]):  # Check first 3 entries
            json.dumps(metadata)
        print("✅ Metadata serialization test passed")
        
        # Show sample metadata for debugging
        print(f"Sample metadata: {metadata_list[0]}")
    except (TypeError, ValueError) as e:
        print(f"❌ Metadata serialization error: {e}")
        print(f"Problematic metadata: {metadata_list[0] if metadata_list else 'No metadata'}")
        
        # Try to identify the problematic field
        if metadata_list:
            for key, value in metadata_list[0].items():
                try:
                    json.dumps(value)
                except:
                    print(f"❌ Problem with field '{key}': {value} (type: {type(value)})")
        
        raise ValueError(f"Invalid metadata detected: {e}")
    
    # Ensure we have valid data
    if not texts:
        raise ValueError("No texts to embed")
    
    # Create vector store
    vectorstore = PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=index_name,
        metadatas=metadata_list
    )
    
    return vectorstore

def load_vector_store(index_name="amazon-reviews"):
    """Load existing Pinecone vector store"""
    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    # Initialize Pinecone client
    pc = init_pinecone(api_key)
    
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' does not exist in Pinecone")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store from existing index
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    return vectorstore

def get_vector_store():
    """Get Pinecone vector store for the app"""
    try:
        # Use the correct name directly
        return load_vector_store("amazon-reviews")
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return None