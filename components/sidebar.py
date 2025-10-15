# [file name]: components/sidebar.py
"""
Sidebar component - Simplified for single dataset
"""
import streamlit as st
from config.settings import DATASET_NAME
from utils.data_loader import load_dataset, prepare_dataset

def render_sidebar():
    """Render the sidebar with filters for 10K dataset"""
    
    st.sidebar.header("📂 Dataset Information")
    st.sidebar.success(f"✅ Using {DATASET_NAME}")
    
    # Load the dataset
    df = load_dataset()
    if df.empty:
        return None
    
    df, cleaning_info = prepare_dataset(df)
    
    # Show dataset info
    st.sidebar.info(f"📋 Loaded: {len(df):,} reviews")
    st.sidebar.info(f"🏷️ Categories: {df['main_category'].nunique()}")
    st.sidebar.info(f"📅 Years: {df['year'].min()}-{df['year'].max()}")
    
    # Filters section
    st.sidebar.header("🔍 Filters")
    
    # Year range filter
    available_years = sorted(df['year'].unique())
    min_year = int(available_years[0])
    max_year = int(available_years[-1])
    
    year_range = st.sidebar.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
        help=f"Available years: {min_year} to {max_year}"
    )
    
    # Category filter
    try:
        unique_categories = sorted(df['main_category'].unique())
    except TypeError:
        unique_categories = sorted([str(cat) for cat in df['main_category'].unique()])
    
    categories = st.sidebar.multiselect(
        "Select Categories:",
        options=unique_categories,
        default=unique_categories
    )
    
    # Rating filter
    rating_range = st.sidebar.slider(
        "Rating Range:",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.1
    )
    
    # Apply filters
    year_filter = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
    category_filter = df['main_category'].isin(categories)
    rating_filter = (df['rating'] >= rating_range[0]) & (df['rating'] <= rating_range[1])
    
    df_filtered = df[year_filter & category_filter & rating_filter]
    
    # Show filter impact
    if len(df_filtered) != len(df):
        excluded = len(df) - len(df_filtered)
        st.sidebar.info(f"🔍 Filters excluded {excluded} records")
    
    # Check if filtered data is empty
    if df_filtered.empty:
        st.warning("🚫 No data matches your filters. Please adjust your selection.")
        return None
    
    return df_filtered