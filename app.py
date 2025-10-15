# [file name]: app.py
"""
Amazon Product Reviews Analytics Dashboard - Final Integrated Version
"""

import streamlit as st
from config.settings import (
    APP_TITLE, PAGE_TITLE, LAYOUT, TAB_NAMES
)
from components.sidebar import render_sidebar
from utils.api_keys import display_api_status
from components.overview_tab import render_overview_tab
from components.time_analytics_tab import render_time_analytics_tab
from components.categories_tab import render_categories_tab
from components.advanced_tab import render_advanced_tab
from components.data_explorer_tab import render_data_explorer_tab
from components.interactive_chatbot_tab import render_interactive_chatbot_tab
from components.interactive_analyst_tab import render_interactive_analyst_tab
from components.interactive_sentiment_tab import render_interactive_sentiment_tab

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛍️",
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )

def main():
    """Main application function"""
    
    # Configure page
    configure_page()
    
    # App header
    st.title(PAGE_TITLE)
    st.markdown("### 🔍 Comprehensive Analytics Dashboard for Amazon Product Reviews")
    
    # Sidebar for data loading and filtering
    with st.sidebar:
        df_filtered = render_sidebar()
        
        # Display API status once here
        st.sidebar.markdown("---")
        display_api_status()
    
    # Check if data is loaded
    if df_filtered is None or len(df_filtered) == 0:
        st.error("❌ No data available. Please check your data files or adjust filters.")
        st.info("""
        **Troubleshooting:**
        1. Ensure data files exist in the `data/` directory
        2. Check filter settings in the sidebar
        3. Verify file formats and column names
        4. Run preprocessing: `python src/preprocess_data.py`
        """)
        return
    
    # Create tabs for different analysis views
    tabs = st.tabs(TAB_NAMES)
    
    # Tab 1: Overview
    with tabs[0]:
        render_overview_tab(df_filtered)
    
    # Tab 2: Time Analytics
    with tabs[1]:
        render_time_analytics_tab(df_filtered)
    
    # Tab 3: Categories & Ratings
    with tabs[2]:
        render_categories_tab(df_filtered)
    
    # Tab 4: Advanced Analytics
    with tabs[3]:
        render_advanced_tab(df_filtered)
    
    # Tab 5: Data Explorer
    with tabs[4]:
        render_data_explorer_tab(df_filtered)
    
    # Tab 6: Product Assistant (Interactive Chatbot)
    with tabs[5]:
        render_interactive_chatbot_tab(df_filtered)
    
    # Tab 7: Data Analyst (Interactive Analyst)
    with tabs[6]:
        render_interactive_analyst_tab(df_filtered)
    
    # Tab 8: Sentiment Detective (Interactive Sentiment)
    with tabs[7]:
        render_interactive_sentiment_tab(df_filtered)
    
    # Footer
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Dashboard Info")
        if df_filtered is not None:
            st.success(f"✅ **{len(df_filtered):,}** reviews loaded")
            st.info(f"📅 **{df_filtered['main_category'].nunique()}** categories")
            
            # Dataset statistics
            if len(df_filtered) > 0:
                date_range = f"{df_filtered['year'].min()}-{df_filtered['year'].max()}"
                st.info(f"📆 **{date_range}** date range")
                
                positive_pct = (df_filtered['sentiment'] == 'Positive').mean() * 100
                st.info(f"😊 **{positive_pct:.1f}%** positive sentiment")
        
        st.markdown("---")
        st.markdown("""
        **🚀 Features:**
        - 📊 Interactive analytics
        - 🤖 AI-powered chatbot
        - 🎭 Advanced sentiment analysis
        - 📱 Responsive design
        - 💾 Data export capabilities
        """)

if __name__ == "__main__":
    main()