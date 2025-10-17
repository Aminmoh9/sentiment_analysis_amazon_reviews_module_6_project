# [file name]: config/settings.py
"""
Configuration settings - Streamlined for 10K dataset only
"""
import streamlit as st

# App Configuration
APP_TITLE = "Amazon Product Reviews Analytics Dashboard"
PAGE_TITLE = "🛍️ Amazon Product Reviews Analytics"
LAYOUT = "wide"

# Dataset Configuration - ONLY 10K dataset
DATASET_NAME = "10K Representative Sample"
DATA_PATH = "data/appliances_10k_sample.csv"

# Sentiment Analysis Configuration
SENTIMENT_COLORS = {
    'Positive': '#2E8B57', 
    'Neutral': '#FFD700', 
    'Negative': '#DC143C'
}

# Vector Store Configuration
VECTOR_STORE_NAME = "amazon-reviews"

# Tab Configuration
TAB_NAMES = [
    "📊 Overview", 
    "📅 Time Analytics", 
    "🏷️ Categories & Ratings", 
    "🔍 Advanced Analytics", 
    "📋 Data Explorer",
    "🤖 Product Assistant",
    "🔍 Data Analyst"
]

# Default columns for data explorer
DEFAULT_DISPLAY_COLUMNS = [
    'product_title', 
    'main_category', 
    'rating', 
    'sentiment', 
    'review_date', 
    'year'
]

# OpenAI Configuration
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.1
OPENAI_MAX_TOKENS = 150

# Chat Configuration
MAX_CHAT_HISTORY = 10
CHAT_MODELS = {
    "gpt-3.5-turbo": {
        "description": "Fast and cost-effective, good for general queries",
        "cost_per_1k_tokens": 0.0015
    },
    "gpt-4": {
        "description": "Most accurate, best for complex analysis",
        "cost_per_1k_tokens": 0.03
    }
}
# Price ranges for analysis
PRICE_RANGES = [0, 50, 100, 200, 500, float('inf')]
PRICE_LABELS = ['<$50', '$50-100', '$100-200', '$200-500', '>$500']

# Review length categories
REVIEW_LENGTH_BINS = [0, 100, 300, 1000, float('inf')]
REVIEW_LENGTH_LABELS = ['Short (<100)', 'Medium (100-300)', 'Long (300-1000)', 'Very Long (>1000)']