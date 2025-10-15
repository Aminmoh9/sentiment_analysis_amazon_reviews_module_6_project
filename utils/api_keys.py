# [file name]: utils/api_keys.py
"""
API key management utilities
"""
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_api_keys():
    """Check if required API keys are available from .env file"""
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    # Return status for both keys (no sidebar display here)
    status = {
        'openai': bool(openai_key and openai_key.strip()),
        'pinecone': bool(pinecone_key and pinecone_key.strip())
    }
    
    return status

def display_api_status():
    """Display API key status in sidebar (call this once from main app)"""
    status = check_api_keys()
    
    st.sidebar.header("🔑 API Status")
    
    if status['openai'] and status['pinecone']:
        st.sidebar.success("✅ All API keys loaded")
        st.sidebar.info(f"🤖 OpenAI: Ready")
        st.sidebar.info(f"🗂️ Pinecone: Ready") 
    else:
        missing_keys = []
        if not status['openai']:
            missing_keys.append("OPENAI_API_KEY")
            st.sidebar.error("❌ OpenAI API key missing")
        else:
            st.sidebar.success("✅ OpenAI API key loaded")
            
        if not status['pinecone']:
            missing_keys.append("PINECONE_API_KEY")
            st.sidebar.error("❌ Pinecone API key missing")
        else:
            st.sidebar.success("✅ Pinecone API key loaded")
        
        if missing_keys:
            st.sidebar.warning(f"Missing: {', '.join(missing_keys)}")
            st.sidebar.info("Please check your .env file")
    
    return status

def get_openai_key():
    """Get OpenAI API key from environment"""
    return os.getenv("OPENAI_API_KEY")

def get_pinecone_key():
    """Get Pinecone API key from environment"""
    return os.getenv("PINECONE_API_KEY")