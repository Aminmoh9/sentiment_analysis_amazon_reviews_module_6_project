import pandas as pd
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        # Fallback imports
        import pinecone
        from langchain.vectorstores import Pinecone as PineconeVectorStore
        from langchain.embeddings import OpenAIEmbeddings
        Pinecone = pinecone.Pinecone
        ServerlessSpec = None
    except ImportError:
        Pinecone = None
        PineconeVectorStore = None
        OpenAIEmbeddings = None
        ServerlessSpec = None

def init_pinecone(api_key):
    """Initialize Pinecone client with API key"""
    pc = Pinecone(api_key=api_key)
    return pc

def create_vector_store(df, index_name="amazon-reviews", force_recreate=False):
    """Create Pinecone vector store from DataFrame"""
    # Initialize Pinecone client
    import os
    api_key = os.getenv("PINECONE_API_KEY")
    pc = init_pinecone(api_key)
    
    # Delete existing index if force_recreate is True
    if force_recreate and index_name in pc.list_indexes().names():
        print(f"Deleting existing index '{index_name}' to recreate with fresh data...")
        pc.delete_index(index_name)
        import time
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
        import numpy as np
        
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
    for _, row in df.iterrows():
        # Handle price with extra care
        price_value = row.get("price", "")
        if pd.notna(price_value) and str(price_value).strip() and str(price_value).lower() not in ['nan', 'none', 'null']:
            price_clean = str(price_value).strip()
        else:
            price_clean = ""
        
        # Image URL - not available in this CSV format
        image_url = ""
        
        metadata_list.append({
            "asin": clean_value(row.get("asin", "")),
            "product_title": clean_value(row.get("product_title", "")),  # Product title from CSV
            "review_title": clean_value(row.get("review_title", "")),   # Review title from CSV
            "main_category": clean_value(row.get("main_category", "")),
            "image": clean_value(image_url),
            "review_date": row["review_date"].strftime("%Y-%m-%d") if pd.notna(row["review_date"]) else "",
            "sentiment": clean_value(row.get("sentiment", "")),
            "rating": clean_value(row.get("rating", "")),
            "price": price_clean
        })
    
    # Create texts from review title and review text (both are valuable for embeddings)
    # Use the correct column names from the CSV
    review_title = df["review_title"].fillna("") if "review_title" in df.columns else pd.Series([""] * len(df))
    review_text = df["review_text"].fillna("") if "review_text" in df.columns else pd.Series([""] * len(df))
    
    # Combine review title + review text for better semantic search
    # Clean and ensure no NaN values
    texts = []
    for title, text in zip(review_title, review_text):
        combined_text = f"{clean_value(title)}. {clean_value(text)}"
        # Ensure we have some content (not just empty strings and dots)
        if combined_text.strip() in ["", ".", "nan.", ".nan"]:
            combined_text = "No review content available"
        texts.append(combined_text)
    
    # Debug: Check data quality
    print(f"Creating vector store with {len(texts)} texts")
    print(f"Sample text: {texts[0][:100]}...")
    print(f"Sample metadata keys: {list(metadata_list[0].keys())}")
    
    # Additional check for NaN values in metadata
    import json
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
    import os
    
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
    if OpenAIEmbeddings is None:
        raise ImportError("OpenAIEmbeddings not available")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vector store from existing index
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    return vectorstore
