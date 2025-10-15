# [file name]: utils/data_loader.py
"""
Data loading and caching utilities - Simplified for 10K dataset only
"""
import streamlit as st
import pandas as pd
from config.settings import DATA_PATH
from src.data_utils import clean_dataset_for_analysis

@st.cache_data
def load_dataset():
    """Load and cache the 10K dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        st.success(f"✅ Loaded 10K representative dataset: {len(df):,} records")
        return df
    except FileNotFoundError:
        st.error(f"❌ Dataset file not found: {DATA_PATH}")
        st.info("Please run the preprocessing script first: python src/preprocess_data.py")
        return pd.DataFrame()

def prepare_dataset(df):
    """Prepare dataset using consolidated utilities"""
    if df.empty:
        return df, {"before": 0, "after": 0, "dropped": 0}
    
    # Use the consolidated cleaning function
    before_cleaning = len(df)
    df_clean = clean_dataset_for_analysis(df)
    after_cleaning = len(df_clean)
    
    cleaning_info = {
        "before": before_cleaning,
        "after": after_cleaning,
        "dropped": before_cleaning - after_cleaning
    }
    
    if cleaning_info["dropped"] > 0:
        st.warning(f"⚠️ Dropped {cleaning_info['dropped']} records with missing critical data")
    
    return df_clean, cleaning_info