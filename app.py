"""
Amazon Appliances Analytics Dashboard
Uses clean samples extracted from 1.39M processed records
Only loads efficient sample datasets (50 or 10K records) for optimal performance
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*pydantic.*", category=Warning)

# Load environment variables from .env file
load_dotenv()

# Add current directory to Python path to ensure src imports work
if '.' not in sys.path:
    sys.path.append('.')
    
RAG_AVAILABLE = False
RAG_ERROR = None

try:
    # Test individual imports to get specific error info
    import langchain
    from langchain_openai import ChatOpenAI
    from src.chatbot import create_qa_chain, ask_question_with_sources
    from src.pinecone_utils import create_vector_store, load_vector_store
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_ERROR = f"Import error: {str(e)}"
except Exception as e:
    RAG_ERROR = f"Unexpected error: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="🏠 Amazon Appliances' Reviews Analytics",
    #page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_dataset(file_path):
    """Load and cache dataset"""
    return pd.read_csv(file_path)

def main():
    # Title with Amazon logo
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
    with col2:
        st.title("Appliances' Reviews Analytics Dashboard")
    #st.markdown("### 📊 Amazon Appliances Analytics - Clean Samples from 1.39M Dataset")
    
    # Dataset selector
    st.sidebar.header("📂 Dataset Selection")
    dataset_option = st.sidebar.selectbox(
        "Choose dataset size:",
        [
            "50 Records (Quick Preview)",
            "10K Records (Full Analytics)"
        ]
    )
    
    # Load appropriate dataset
    if "50 Records" in dataset_option:
        df = load_dataset("data/appliances_50_sample.csv")
        st.sidebar.success("✅ Using 50-record sample")
    else:
        df = load_dataset("data/appliances_10k_sample.csv") 
        st.sidebar.success("✅ Using 10K representative sample")
    
    # Debug: Show initial record count
    initial_count = len(df)
    st.sidebar.info(f"📋 Loaded: {initial_count:,} records")
    
    # Convert review_date to datetime - handle any parsing errors
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['year'] = df['review_date'].dt.year
    
    # Clean data - remove records with critical missing values
    before_cleaning = len(df)
    df = df.dropna(subset=['review_date', 'rating', 'main_category'])
    after_cleaning = len(df)
    
    if before_cleaning != after_cleaning:
        dropped = before_cleaning - after_cleaning
        st.sidebar.warning(f"⚠️ Dropped {dropped} records with missing critical data")
    
    st.sidebar.success(f"✅ Clean dataset: {len(df):,} records ready")
    
    # Sidebar filters
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
    
    # Category filter - preserve all data, just handle the sorting issue
    try:
        # Try normal sorting first
        unique_categories = sorted(df['main_category'].unique())
    except TypeError:
        # If sorting fails due to mixed types, convert to strings first then sort
        unique_categories = sorted([str(cat) for cat in df['main_category'].unique()])
    
    categories = st.sidebar.multiselect(
        "Select Categories:",
        options=unique_categories,
        default=unique_categories  # Show ALL categories by default
    )
    
    # Rating filter
    rating_range = st.sidebar.slider(
        "Rating Range:",
        min_value=1.0,
        max_value=5.0,
        value=(1.0, 5.0),
        step=0.1
    )
    
    # Apply filters with debugging
    year_filter = (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
    category_filter = df['main_category'].isin(categories)
    rating_filter = (df['rating'] >= rating_range[0]) & (df['rating'] <= rating_range[1])
    
    df_filtered = df[year_filter & category_filter & rating_filter]
    
    # Debug filter impact
    if len(df_filtered) != len(df):
        excluded = len(df) - len(df_filtered)
        st.sidebar.info(f"🔍 Filters excluded {excluded} records")
        
        # Show what each filter removes
        year_excluded = len(df) - year_filter.sum()
        cat_excluded = len(df) - category_filter.sum()
        rating_excluded = len(df) - rating_filter.sum()
        
        if year_excluded > 0:
            st.sidebar.text(f"   Year filter: -{year_excluded}")
        if cat_excluded > 0:
            st.sidebar.text(f"   Category filter: -{cat_excluded}")
        if rating_excluded > 0:
            st.sidebar.text(f"   Rating filter: -{rating_excluded}")
    
    # Check if filtered data is empty
    if df_filtered.empty:
        st.warning("🚫 No data matches your filters. Please adjust your selection.")
        return

    # Create tabs for organized content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "📅 Time Analytics", 
        "🏷️ Categories & Ratings", 
        "🔍 Advanced Analytics", 
        "📋 Data Explorer",
        "🤖 AI Chatbot"
      
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("📊 Key Metrics Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Total Reviews",
                value=f"{len(df_filtered):,}",
                delta=f"{len(df_filtered) - len(df):,} vs total"
            )
        
        with col2:
            avg_rating = df_filtered['rating'].mean()
            st.metric(
                label="⭐ Average Rating",
                value=f"{avg_rating:.2f}",
                delta=f"{avg_rating - df['rating'].mean():.2f} vs total"
            )
        
        with col3:
            st.metric(
                label="🏷️ Categories",
                value=df_filtered['main_category'].nunique(),
                delta=f"{df_filtered['main_category'].nunique() - df['main_category'].nunique()} vs total"
            )
        
        with col4:
            years_span = df_filtered['year'].max() - df_filtered['year'].min() + 1
            st.metric(
                label="📅 Time Span",
                value=f"{years_span} years",
                delta=f"{df_filtered['year'].min()}-{df_filtered['year'].max()}"
            )
        
        # Quick insights
        st.subheader("🎯 Key Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            top_category = df_filtered['main_category'].value_counts().index[0]
            top_category_count = df_filtered['main_category'].value_counts().iloc[0]
            st.info(f"🏆 **Top Category**\n{top_category}\n({top_category_count:,} reviews)")
            
        with insight_col2:
            positive_pct = (df_filtered['sentiment'] == 'Positive').sum() / len(df_filtered) * 100
            st.success(f"😊 **Positive Reviews**\n{positive_pct:.1f}%\n({(df_filtered['sentiment'] == 'Positive').sum():,} reviews)")
            
        with insight_col3:
            recent_reviews = df_filtered[df_filtered['year'] >= df_filtered['year'].max() - 1]
            recent_pct = len(recent_reviews) / len(df_filtered) * 100
            st.warning(f"🕐 **Recent Reviews**\nLast 2 Years: {recent_pct:.1f}%\n({len(recent_reviews):,} reviews)")

        # Overview charts
        st.subheader("📈 Overview Charts")
        
        overview_col1, overview_col2 = st.columns(2)
        
        with overview_col1:
            # Sentiment pie chart
            sentiment_counts = df_filtered['sentiment'].value_counts()
            colors = {'Positive': '#2E8B57', 'Neutral': '#FFD700', 'Negative': '#DC143C'}
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig_sentiment, config={'displayModeBar': False})
        
        with overview_col2:
            # Rating distribution
            rating_dist = df_filtered['rating'].value_counts().sort_index()
            fig_ratings = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Rating Distribution",
                labels={'x': 'Rating Stars', 'y': 'Number of Reviews'},
                color=rating_dist.values,
                color_continuous_scale='viridis'
            )
            fig_ratings.update_xaxes(title="Rating Stars")
            fig_ratings.update_yaxes(title="Number of Reviews")
            st.plotly_chart(fig_ratings, config={'displayModeBar': False})

    # TAB 2: TIME ANALYTICS
    with tab2:
        #st.header("📅 Time Analytics")
        
        # Monthly review trends
        st.subheader("📈 Review Volume Trends")
        monthly_reviews = df_filtered.groupby([df_filtered['review_date'].dt.to_period('M')])['rating'].agg(['count', 'mean']).reset_index()
        monthly_reviews['month'] = monthly_reviews['review_date'].dt.to_timestamp()
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=monthly_reviews['month'],
            y=monthly_reviews['count'],
            mode='lines+markers',
            name='Review Count',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig_timeline.update_layout(
            title="Monthly Review Volume Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Reviews",
            height=500
        )
        st.plotly_chart(fig_timeline, config={'displayModeBar': False})
        
        # Yearly analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Reviews by Year")
            yearly_counts = df_filtered['year'].value_counts().sort_index()
            fig_yearly = px.bar(
                x=yearly_counts.index,
                y=yearly_counts.values,
                title="Review Count by Year",
                labels={'x': 'Year', 'y': 'Number of Reviews'},
                color=yearly_counts.values,
                color_continuous_scale='Blues'
            )
            fig_yearly.update_xaxes(title="Year")
            fig_yearly.update_yaxes(title="Number of Reviews")
            st.plotly_chart(fig_yearly, config={'displayModeBar': False})
        
        with col2:
            st.subheader("⭐ Average Rating by Year")
            yearly_ratings = df_filtered.groupby('year')['rating'].mean()
            fig_rating_year = px.line(
                x=yearly_ratings.index,
                y=yearly_ratings.values,
                title="Average Rating Trends Over Time",
                labels={'x': 'Year', 'y': 'Average Rating'},
                markers=True
            )
            fig_rating_year.update_traces(line=dict(width=3))
            fig_rating_year.update_xaxes(title="Year")
            fig_rating_year.update_yaxes(title="Average Rating (1-5 Stars)")
            st.plotly_chart(fig_rating_year, config={'displayModeBar': False})

    # TAB 3: CATEGORIES & RATINGS  
    with tab3:
        #st.header("🏷️ Categories & Rating Analysis")
        
        # Top categories
        st.subheader("📊 Category Performance")
        category_stats = df_filtered.groupby('main_category').agg({
            'rating': ['count', 'mean']
        }).round(2)
        category_stats.columns = ['Review Count', 'Avg Rating']
        category_stats = category_stats.sort_values('Review Count', ascending=False).head(10)
        
        fig_categories = px.bar(
            category_stats.reset_index(),
            x='main_category',
            y='Review Count',
            title="Top 10 Categories by Review Count",
            color='Avg Rating',
            color_continuous_scale='RdYlGn',
            text='Avg Rating',
            labels={'main_category': 'Category', 'Review Count': 'Number of Reviews'}
        )
        fig_categories.update_xaxes(tickangle=45, title="Product Category")
        fig_categories.update_yaxes(title="Number of Reviews")
        fig_categories.update_layout(height=500)
        st.plotly_chart(fig_categories, config={'displayModeBar': False})
        
        # Category details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Category Ratings Comparison")
            category_avg_ratings = df_filtered.groupby('main_category')['rating'].mean().sort_values(ascending=False).head(10)
            
            # Create DataFrame with proper descending order (highest rating at top)
            rating_df = category_avg_ratings.reset_index()
            rating_df.columns = ['Category', 'Avg_Rating']
            
            fig_cat_ratings = px.bar(
                rating_df,
                x='Avg_Rating',
                y='Category',
                orientation='h',
                title="Average Rating by Category (Top 10)",
                color='Avg_Rating',
                color_continuous_scale='RdYlGn',
                labels={'Avg_Rating': 'Average Rating (Stars)', 'Category': 'Product Category'}
            )
            
            # Force descending order: highest ratings at TOP of chart
            fig_cat_ratings.update_layout(
                yaxis={'categoryorder': 'total ascending'},  # This puts highest values at top
                coloraxis_colorbar=dict(title="Rating")
            )
            fig_cat_ratings.update_xaxes(title="Average Rating (Stars)")
            fig_cat_ratings.update_yaxes(title="Product Category")
            st.plotly_chart(fig_cat_ratings, config={'displayModeBar': False})
        
        with col2:
            st.subheader("📈 Rating Trends by Categories")
            
            # Get top categories by volume for default selection
            top_categories = df_filtered['main_category'].value_counts().head(10).index.tolist()
            
            # Category filter for trends chart
            selected_trend_categories = st.multiselect(
                "Select categories to display in trends:",
                options=top_categories,
                default=top_categories[:3],  # Default to top 3 for readability
                help="Select 1-3 categories for clearest visualization"
            )
            
            if selected_trend_categories:
                df_trend_cats = df_filtered[df_filtered['main_category'].isin(selected_trend_categories)]
                
                if len(df_trend_cats) > 0:
                    monthly_cat_rating = df_trend_cats.groupby([
                        df_trend_cats['review_date'].dt.to_period('M'), 
                        'main_category'
                    ])['rating'].mean().reset_index()
                    monthly_cat_rating['month'] = monthly_cat_rating['review_date'].dt.to_timestamp()
                    
                    fig_trends = px.line(
                        monthly_cat_rating,
                        x='month',
                        y='rating',
                        color='main_category',
                        title=f"Rating Trends ({len(selected_trend_categories)} Selected Categories)",
                        labels={'month': 'Month', 'rating': 'Average Rating', 'main_category': 'Category'}
                    )
                    fig_trends.update_xaxes(title="Time (Months)")
                    fig_trends.update_yaxes(title="Average Rating (Stars)", range=[1, 5])
                    fig_trends.update_traces(line=dict(width=3))
                    st.plotly_chart(fig_trends, config={'displayModeBar': False})
                else:
                    st.info("No data available for selected categories")
            else:
                st.info("👆 Select categories above to view rating trends")

    # TAB 4: ADVANCED ANALYTICS
    with tab4:
        #st.header("🔍 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Review Length Analysis")
            # Create a copy to avoid SettingWithCopyWarning
            df_analysis = df_filtered.copy()
            df_analysis['review_length'] = df_analysis['review_text'].str.len()
            df_analysis['length_category'] = pd.cut(
                df_analysis['review_length'], 
                bins=[0, 100, 300, 1000, float('inf')],
                labels=['Short (<100)', 'Medium (100-300)', 'Long (300-1000)', 'Very Long (>1000)']
            )
            
            length_rating = df_analysis.groupby('length_category', observed=False)['rating'].mean()
            fig_length = px.bar(
                x=length_rating.index,
                y=length_rating.values,
                title="Average Rating by Review Length",
                color=length_rating.values,
                color_continuous_scale='viridis',
                labels={'x': 'Review Length Category', 'y': 'Average Rating'}
            )
            fig_length.update_xaxes(title="Review Length Category")
            fig_length.update_yaxes(title="Average Rating (Stars)")
            st.plotly_chart(fig_length, config={'displayModeBar': False})
        
        with col2:
            st.subheader("💰 Price Analysis")
            if 'price' in df_filtered.columns and df_filtered['price'].notna().sum() > 0:
                # Create a copy to avoid SettingWithCopyWarning
                price_data = df_filtered[df_filtered['price'].notna()].copy()
                if len(price_data) > 10:
                    # Extract numeric price
                    price_data['price_numeric'] = price_data['price'].str.replace('$', '').str.replace(',', '').astype(float)
                    price_data['price_range'] = pd.cut(
                        price_data['price_numeric'],
                        bins=[0, 50, 100, 200, 500, float('inf')],
                        labels=['<$50', '$50-100', '$100-200', '$200-500', '>$500']
                    )
                    
                    price_rating = price_data.groupby('price_range', observed=False)['rating'].mean()
                    fig_price = px.bar(
                        x=price_rating.index,
                        y=price_rating.values,
                        title="Average Rating by Price Range",
                        color=price_rating.values,
                        color_continuous_scale='Oranges',
                        labels={'x': 'Price Range', 'y': 'Average Rating'}
                    )
                    fig_price.update_xaxes(title="Price Range")
                    fig_price.update_yaxes(title="Average Rating (Stars)")
                    st.plotly_chart(fig_price, config={'displayModeBar': False})
                else:
                    st.info("Insufficient price data for analysis")
            else:
                st.info("Price data not available in this sample")
        
        # Sentiment deep dive
        st.subheader("😊 Sentiment Deep Dive")
        
        sentiment_col1, sentiment_col2 = st.columns(2)
        
        with sentiment_col1:
            # Sentiment by category
            sentiment_category = pd.crosstab(df_filtered['main_category'], df_filtered['sentiment'], normalize='index') * 100
            fig_sent_cat = px.bar(
                sentiment_category.head(8),
                title="Sentiment Distribution by Category (%)",
                color_discrete_map={'Positive': '#2E8B57', 'Neutral': '#FFD700', 'Negative': '#DC143C'},
                labels={'index': 'Category', 'value': 'Percentage (%)', 'variable': 'Sentiment'}
            )
            fig_sent_cat.update_xaxes(title="Product Category")
            fig_sent_cat.update_yaxes(title="Percentage (%)")
            st.plotly_chart(fig_sent_cat, config={'displayModeBar': False})
        
        with sentiment_col2:
            # Sentiment trends over time
            sentiment_time = df_filtered.groupby([df_filtered['year'], 'sentiment']).size().unstack(fill_value=0)
            sentiment_time_pct = sentiment_time.div(sentiment_time.sum(axis=1), axis=0) * 100
            
            fig_sent_time = px.line(
                sentiment_time_pct,
                title="Sentiment Trends Over Time (%)",
                color_discrete_map={'Positive': '#2E8B57', 'Neutral': '#FFD700', 'Negative': '#DC143C'},
                labels={'index': 'Year', 'value': 'Percentage (%)', 'variable': 'Sentiment'}
            )
            fig_sent_time.update_xaxes(title="Year")
            fig_sent_time.update_yaxes(title="Percentage (%)")
            st.plotly_chart(fig_sent_time, config={'displayModeBar': False})

    # TAB 5: DATA EXPLORER
    with tab5:
        #st.header("📋 Data Explorer")
        
        # Dataset statistics
        st.subheader("📊 Dataset Statistics")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Total Records", f"{len(df_filtered):,}")
            st.metric("Date Range", f"{df_filtered['year'].min()}-{df_filtered['year'].max()}")
        
        with stats_col2:
            st.metric("Categories", df_filtered['main_category'].nunique())
            st.metric("Avg Rating", f"{df_filtered['rating'].mean():.2f}")
        
        with stats_col3:
            st.metric("Positive Reviews", f"{(df_filtered['sentiment'] == 'Positive').sum():,}")
            st.metric("Recent Reviews (Last 2Y)", f"{len(df_filtered[df_filtered['year'] >= df_filtered['year'].max()-1]):,}")
        
        # Sample data view
        st.subheader(f"🔍 Sample Records ({len(df_filtered):,} total)")
        
        # Column selector
        all_columns = df_filtered.columns.tolist()
        default_columns = ['product_title', 'main_category', 'rating', 'sentiment', 'review_date', 'year']
        display_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=[col for col in default_columns if col in all_columns]
        )
        
        if display_columns:
            # Number of records to show
            num_records = st.slider("Number of records to display:", 10, min(100, len(df_filtered)), 20)
            
            sample_df = df_filtered[display_columns].head(num_records)
            st.dataframe(sample_df, width='stretch')
        
        # Download section
        st.subheader("💾 Data Export")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            csv_data = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"amazon_appliances_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            st.info(f"💡 **Dataset Info:**\n- Source: Stratified samples from 1.39M records\n- Processing: Amazon JSONL files\n- Quality: 100% complete metadata\n- Sampling: Category + Rating stratified")

    # TAB 6: AI CHATBOT (RAG-POWERED)
    with tab6:
        st.header("🤖 LangChain RAG Assistant for Amazon Reviews")
        
        # Check if RAG is available
        if not RAG_AVAILABLE:
            st.error(f"🚫 **LangChain RAG Not Available**\n\nError: {RAG_ERROR}")
            
            # Show debug info
            with st.expander("🔧 Debug Information"):
                st.code(f"""
Import Error Details: {RAG_ERROR}

Try these fixes:
1. Restart the Streamlit app
2. Check if all files exist in src/ folder
3. Verify Python environment has all packages

Files needed:
- src/chatbot.py
- src/pinecone_utils.py
- src/embeddings.py
""")
                
            # Fallback to basic analytics mode
            st.subheader("� Basic Analytics Mode")
            st.info("The advanced RAG features are not available, but you can still use the statistical analysis in other tabs.")
            
        # Check for API keys
        elif not check_api_keys():
            st.error("🔑 Please configure your API keys in the sidebar to use the AI assistant.")
            
        else:
            st.info("""
            🚀 **LangChain RAG + Metadata Features:**
            - **Semantic Search** through actual review content + metadata
            - **Product Intelligence** analyzing ASINs, categories, prices, dates
            - **Cross-Reference Analysis** linking reviews to product metadata  
            - **Temporal Insights** understanding trends over time
            - **Price-Performance Analysis** from review + pricing data
            - **Category-Specific Insights** from structured metadata
            """)
            
            # Initialize or load vector store (using current dataset choice, not filtered data)
        if initialize_vector_store_properly(dataset_option):
            st.subheader("🤖 AI Product Assistant")
            
            # Initialize chat history
            if "rag_messages" not in st.session_state:
                # Get the actual full dataset count for the welcome message
                full_dataset = st.session_state.get('full_dataset', df_filtered)
                total_reviews = len(full_dataset)
                
                st.session_state.rag_messages = [
                    {"role": "assistant", "content": f"""🤖 **Welcome to your LangChain RAG Assistant!**

I've indexed **{total_reviews:,} reviews** with full product metadata. I can analyze both review content and product information.

**💬 Try asking me:**
• "What is the top rated product?"
• "Show me 5-star appliances" 
• "What do customers complain about?"
• "Find products under $50"
• "Compare different categories"
• "Which ASINs have delivery issues?"

**🔍 I can search through:**
✓ Review text and titles
✓ Product ASINs and categories  
✓ Ratings and sentiment
✓ Prices and dates
✓ Customer feedback patterns

Ask me anything about your Amazon appliance reviews!"""}
                ]
            
            # Show available metadata fields and features
            with st.expander("📋 Available Features & Query Examples"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**� Example Queries:**")
                    st.code('"What is the top rated product?"')
                    st.code('"Show me 5-star appliances"')
                    st.code('"Find products under $50"')
                    st.code('"What do customers complain about?"')
                    st.write("**Available Data:**")
                    st.write("• ASIN (Product ID)")
                    st.write("• Product Title & Category")
                    st.write("• Price Information")
                with col2:
                    st.write("**🎯 Query Types Supported:**")
                    st.code('"Compare different categories"')
                    st.code('"Which products have delivery issues?"')
                    st.code('"Recommend kitchen appliances"')
                    st.code('"What happened in 2023?"')
                    st.write("**Review Analysis:**")
                    st.write("• Rating (1-5 stars)")
                    st.write("• Review Date & Sentiment")
                    st.write("• Full Text Content")
                    
            # Quick dataset preview
            with st.expander("🔍 Quick Look at Your Actual Dataset"):
                st.subheader("Top-Rated Products (5 Stars) in Your Data")
                top_rated = df_filtered[df_filtered['rating'] == 5.0]
                if len(top_rated) > 0:
                    for i, (_, row) in enumerate(top_rated.head(3).iterrows()):
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            asin = row.get('asin', 'N/A')
                            if asin and asin != 'N/A':
                                st.info(f"📦\n{asin}")
                            else:
                                st.info("📦\nNo ASIN")
                        
                        with col2:
                            st.write(f"**{row.get('product_title', 'Unknown Product')}**")
                            st.write(f"🏷️ {row.get('main_category', 'Unknown')} | ⭐ {row.get('rating', 'N/A')} | 😊 {row.get('sentiment', 'Unknown')}")
                            st.write(f"📝 {row.get('review_title', 'No review title')}")
                        st.write("---")
                else:
                    st.info("No 5-star products found in current filtered data")
                
                # Dataset summary
                st.write(f"**Dataset Summary:** {len(df_filtered)} reviews | {df_filtered['rating'].mean():.2f}⭐ avg | {df_filtered['asin'].nunique() if 'asin' in df_filtered.columns else '?'} unique products")

            # Display chat messages
            for message in st.session_state.rag_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about reviews content + metadata..."):
                # Add user message to chat history
                st.session_state.rag_messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                response_content = ""
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Analyzing reviews + metadata and generating response..."):
                        # Check if this needs dataset analysis (categories are in metadata, not review docs)
                        prompt_lower = prompt.lower()
                        needs_dataset_analysis = any(word in prompt_lower for word in ['compare', 'categories', 'category', 'different categories'])
                        
                        if needs_dataset_analysis:
                            # Use dataset analysis for category comparisons since categories are in metadata
                            st.info("🔄 Using comprehensive dataset analysis (categories are in metadata, not individual reviews)...")
                            fallback_response = analyze_dataset_directly(prompt, df_filtered)
                            response_content = fallback_response['text']
                            st.write(fallback_response['text'])
                            if fallback_response.get('products'):
                                display_product_cards(fallback_response['products'])
                        else:
                            # Use intelligent RAG for other queries
                            try:
                                response = generate_rag_response_with_images(prompt, st.session_state.vector_store)
                                
                                # Always use RAG results if we get them - trust the AI
                                if isinstance(response, dict):
                                    response_content = response['text']
                                    st.write(response['text'])
                                    if response.get('products'):
                                        display_product_cards(response['products'])
                                else:
                                    # RAG returned string response
                                    response_content = str(response)
                                    st.write(response_content)
                            except Exception as e:
                                # Only fallback on actual errors
                                st.error(f"RAG error: {str(e)}")
                                st.info("🔄 Using direct dataset analysis instead...")
                                fallback_response = analyze_dataset_directly(prompt, df_filtered)
                                response_content = fallback_response['text']
                                st.write(fallback_response['text'])
                                if fallback_response.get('products'):
                                    display_product_cards(fallback_response['products'])
                    
                # Add assistant response to chat history with actual content
                st.session_state.rag_messages.append({"role": "assistant", "content": response_content})
            
            # Quick Dataset Analysis Tools
            st.markdown("---")
            st.subheader("🔍 Quick Dataset Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🏆 Top Rated Products"):
                    st.write("**5-Star Rated Products:**")
                    top_products = df_filtered[df_filtered['rating'] == 5.0][['product_title', 'asin', 'price']].drop_duplicates()
                    for i, (_, product) in enumerate(top_products.head(5).iterrows(), 1):
                        price_text = f" - {product['price']}" if product['price'] and str(product['price']).strip() else " - Price not listed"
                        st.write(f"{i}. **{product['product_title'][:40]}{'...' if len(product['product_title']) > 40 else ''}**")
                        st.write(f"   • ASIN: {product['asin']}{price_text}")
            
            with col2:
                if st.button("📈 Rating Distribution"):
                    rating_counts = df_filtered['rating'].value_counts().sort_index()
                    st.write("**Rating Breakdown:**")
                    for rating, count in rating_counts.items():
                        pct = count/len(df_filtered)*100
                        st.write(f"• {rating}⭐: {count} reviews ({pct:.1f}%)")
            
            with col3:
                if st.button("🏷️ Category Comparison"):
                    st.write("**Comprehensive Category Analysis:**")
                    
                    # Get category statistics
                    category_analysis = df_filtered.groupby('main_category').agg({
                        'rating': ['count', 'mean'],
                        'asin': 'nunique'
                    }).round(2)
                    category_analysis.columns = ['Review_Count', 'Avg_Rating', 'Products']
                    
                    # Add sentiment analysis per category
                    for category in category_analysis.index:
                        cat_df = df_filtered[df_filtered['main_category'] == category]
                        total_reviews = len(cat_df)
                        positive_pct = (cat_df['sentiment'] == 'Positive').sum() / total_reviews * 100
                        negative_pct = (cat_df['sentiment'] == 'Negative').sum() / total_reviews * 100
                        
                        # Get top product in this category
                        top_product = cat_df.nlargest(1, 'rating').iloc[0] if len(cat_df) > 0 else None
                        
                        st.write(f"**📂 {category}**")
                        st.write(f"   • Reviews: {total_reviews} | Products: {int(category_analysis.loc[category, 'Products'])}")
                        st.write(f"   • Avg Rating: {category_analysis.loc[category, 'Avg_Rating']}⭐")
                        st.write(f"   • Sentiment: {positive_pct:.0f}% Pos, {negative_pct:.0f}% Neg")
                        
                        if top_product is not None:
                            st.write(f"   • Top: {top_product.get('product_title', 'Unknown')[:30]}... ({top_product.get('rating', 'N/A')}⭐)")
                        st.write("")
        else:
            st.warning("⚠️ Unable to initialize vector store. Please check your API keys and data.")

def check_api_keys():
    """Check if required API keys are available from .env file"""
    # Load API keys from environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    st.sidebar.header("🔑 API Status")
    
    if openai_key and pinecone_key:
        st.sidebar.success("✅ API keys loaded from .env file")
        st.sidebar.info(f"🤖 OpenAI: Ready")
        st.sidebar.info(f"🗂️ Pinecone: Ready") 
        
        return True
    else:
        st.sidebar.error("❌ API keys missing from .env file")
        missing_keys = []
        if not openai_key:
            missing_keys.append("OPENAI_API_KEY")
        if not pinecone_key:
            missing_keys.append("PINECONE_API_KEY")
        
        st.sidebar.warning(f"Missing: {', '.join(missing_keys)}")
        st.sidebar.info("Please add the required API keys to your .env file")
        return False

def initialize_vector_store_properly(current_dataset_option):
    """Initialize vector store with proper caching - use current dataset choice"""
    
    # Check if vector store needs to be created or if dataset changed
    dataset_key = f"vector_store_{current_dataset_option}"
    
    if dataset_key not in st.session_state:
        try:
            # Load the dataset that user actually selected
            if "50 Records" in current_dataset_option:
                full_df = load_dataset("data/appliances_50_sample.csv")
                records_text = "50 records"
            else:
                full_df = load_dataset("data/appliances_10k_sample.csv")
                records_text = "10K records"
            
            # Prepare full dataset
            full_df['review_date'] = pd.to_datetime(full_df['review_date'])
            full_df['year'] = full_df['review_date'].dt.year
            
            st.info(f"🔧 Building vector store from {records_text} dataset ({len(full_df)} reviews)...")
            st.info("📌 This gets cached per dataset choice - filters won't affect the RAG knowledge base")
            
            # Try to load existing, otherwise create new
            # Map to existing Pinecone indexes
            if "50 Records" in current_dataset_option:
                index_name = "amazon-reviews-50-records"
            else:
                index_name = "amazon-reviews"  # For 10K records, use the main index
                
            try:
                vector_store = load_vector_store(index_name)
                st.success(f"✅ Loaded cached vector store ({records_text}) - using '{index_name}'")
            except:
                with st.spinner("🔍 Creating embeddings (one-time process)..."):
                    vector_store = create_vector_store(full_df, index_name=index_name, force_recreate=True)
                st.success(f"✅ Built new vector store ({records_text}) - created '{index_name}'")
            
            # Cache both the vector store and dataset
            st.session_state[dataset_key] = vector_store
            st.session_state.vector_store = vector_store  # For compatibility
            st.session_state.full_dataset = full_df
            
            # Test functionality
            test_results = st.session_state.vector_store.similarity_search("appliance product", k=3)
            st.info(f"🔍 Vector store test: Found {len(test_results)} relevant documents")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Vector store initialization failed: {str(e)}")
            return False
    else:
        # Already initialized for this dataset choice
        st.session_state.vector_store = st.session_state[dataset_key]
        full_df = st.session_state.get('full_dataset')
        if full_df is not None:
            records_text = "50 records" if "50 Records" in current_dataset_option else "10K records"
            st.success(f"✅ Using cached vector store ({records_text})")
        else:
            st.success("✅ Using cached vector store")
        return True

def generate_rag_response(prompt, vector_store):
    """Generate response using LangChain RAG with metadata analysis"""
    try:
        # Enhanced prompt to ensure it analyzes only the provided dataset
        enhanced_prompt = f"""
        You are analyzing a specific Amazon appliances review dataset. Answer this question ONLY based on the actual review data provided to you: {prompt}
        
        IMPORTANT CONSTRAINTS:
        - Only reference products, ASINs, ratings, and reviews that are actually present in the provided dataset
        - Do not make up product names, ratings, or details not found in the actual data
        - Use specific ASINs, product titles, ratings, and review content from the retrieved documents
        - Reference actual customer quotes and specific metadata when possible
        
        The dataset includes appliance reviews with metadata such as:
        - ASIN (product ID)
        - Product titles and categories  
        - Customer ratings (1-5 stars)
        - Review text and sentiment
        - Prices and dates
        
        Base your response entirely on the retrieved review documents and their metadata.
        """
        
        # Use the enhanced function that returns sources
        response = ask_question_with_sources(vector_store, enhanced_prompt)
        
        # Format the response with rich metadata sources
        if isinstance(response, dict) and 'answer' in response:
            formatted_response = f"{response['answer']}\n\n"
            
            if 'sources' in response and response['sources']:
                formatted_response += "📚 **Sources & Metadata:**\n"
                for i, source in enumerate(response['sources'][:3], 1):
                    # Extract rich metadata
                    category = source.get('main_category', 'Unknown')
                    rating = source.get('rating', 'N/A')
                    product = source.get('product_title', 'Unknown Product')
                    asin = source.get('asin', 'N/A')
                    price = source.get('price', 'N/A')
                    date = source.get('review_date', 'Unknown')
                    sentiment = source.get('sentiment', 'Unknown')
                    
                    formatted_response += f"{i}. **{product[:40]}{'...' if len(product) > 40 else ''}**\n"
                    formatted_response += f"   • Category: {category} | Rating: {rating}⭐ | Sentiment: {sentiment}\n"
                    if price and price != 'N/A' and str(price).strip():
                        formatted_response += f"   • Price: {price} | ASIN: {asin}\n"
                    else:
                        formatted_response += f"   • ASIN: {asin}\n"
                    formatted_response += f"   • Date: {str(date)[:10]}\n\n"
            
            return formatted_response
        else:
            return str(response)
            
    except Exception as e:
        return f"⚠️ Sorry, I encountered an error: {str(e)}\n\nPlease check your API keys and try again."

def generate_rag_response_with_images(prompt, vector_store):
    """Generate RAG response with product image support - query-specific analysis"""
    try:
        # Create query-specific prompt that enforces relevance
        # Detect if this is a price-based query
        is_price_query = any(word in prompt.lower() for word in ['under $', 'below $', 'less than $', 'cheaper than', 'budget', 'affordable', 'price'])
        
        # Create enhanced search query for better semantic matching
        enhanced_search_query = prompt
        
        # Enhance search terms for better semantic matching and AI analysis
        if any(word in prompt.lower() for word in ['delivery', 'shipping', 'arrived', 'delayed', 'late', 'package']):
            enhanced_search_query = f"{prompt} shipping delivery package arrived delayed late lost damaged logistics fulfillment"
        elif any(word in prompt.lower() for word in ['complain', 'complaint', 'problem', 'issue', 'bad', 'worst', 'hate', 'terrible']):
            enhanced_search_query = f"{prompt} complaint problem issue bad terrible awful disappointed negative feedback criticism"
        elif any(word in prompt.lower() for word in ['compare', 'category', 'categories', 'different']):
            enhanced_search_query = f"{prompt} category comparison analysis different types performance quality"
        elif any(word in prompt.lower() for word in ['top', 'best', 'highest', 'recommend', 'good']):
            enhanced_search_query = f"{prompt} best top excellent recommend quality satisfied happy positive"
        elif any(word in prompt.lower() for word in ['price', 'cheap', 'expensive', 'cost', '$', 'budget']):
            enhanced_search_query = f"{prompt} price cost money dollar value budget affordable expensive cheap worth"
        else:
            enhanced_search_query = prompt
        
        # General intelligent analysis prompt for all RAG queries
        constrained_prompt = f"""
            You are an expert AI analyst specializing in Amazon product reviews and customer sentiment analysis. Your role is to provide intelligent, comprehensive analysis based on the review documents provided.

            USER QUERY: "{prompt}"

            ANALYSIS FRAMEWORK:
            1. **Semantic Understanding**: Understand the intent behind the query, not just keywords
            2. **Pattern Recognition**: Identify trends, correlations, and insights across reviews
            3. **Context Synthesis**: Connect information across multiple products and reviews
            4. **Intelligent Filtering**: Focus on the most relevant information for the specific query

            QUERY-SPECIFIC INTELLIGENCE:
            - **Complaint Analysis**: When asked about complaints/problems, analyze negative sentiment patterns, identify common issues across products, categorize problem types, and provide actionable insights
            - **Product Recommendations**: When asked for top/best products, consider ratings, sentiment, review content quality, and customer satisfaction patterns
            - **Delivery Issues**: Focus on logistics-related complaints, delivery timeline problems, and shipping quality issues
            - **Price Analysis**: Consider value for money, price-performance relationships, and customer satisfaction at different price points

            RESPONSE STRUCTURE:
            - Start with a clear executive summary
            - Provide specific examples from the review data
            - Include quantitative insights where relevant (percentages, averages, trends)
            - Reference actual customer quotes that illustrate key points
            - Always include product metadata (ASIN, category, rating, price) for recommendations
            - End with actionable insights or recommendations

            INTELLIGENCE REQUIREMENTS:
            - Use semantic analysis to understand customer intent and emotions
            - Identify patterns that might not be obvious from simple keyword matching
            - Provide context and explanation, not just lists
            - Connect related information across different reviews and products
            - Offer insights that go beyond what a simple database query would provide

            Base your analysis entirely on the retrieved review documents while applying advanced analytical thinking to extract meaningful insights.
            """
        
        # For price queries, pre-filter the source documents
        if is_price_query:
            # Extract price threshold from query
            import re
            price_match = re.search(r'\$(\d+)', prompt)
            price_threshold = int(price_match.group(1)) if price_match else None
            
            if price_threshold:
                # Get raw documents from vector store
                raw_docs = vector_store.similarity_search(prompt, k=10)
                filtered_docs = []
                
                # Filter documents by price
                for doc in raw_docs:
                    metadata = doc.metadata
                    if 'price' in metadata and metadata['price']:
                        try:
                            price_clean = re.sub(r'[^\d.]', '', str(metadata['price']))
                            if price_clean:
                                price_value = float(price_clean)
                                if price_value <= price_threshold:
                                    filtered_docs.append(doc)
                        except:
                            continue
                
                if filtered_docs:
                    # Create a focused response with filtered documents
                    products_info = []
                    for doc in filtered_docs:
                        metadata = doc.metadata
                        product_info = f"**{metadata.get('product_title', metadata.get('title', 'Unknown Product'))}** (${metadata.get('price', 'N/A')})"
                        product_info += f"\n   • ASIN: {metadata.get('asin', 'N/A')} • Rating: {metadata.get('overall', metadata.get('rating', 'N/A'))}/5 stars"
                        product_info += f"\n   • Review: {doc.page_content[:150]}..."
                        products_info.append(product_info)
                    
                    answer_text = f"Based on your search for products under ${price_threshold}, I found {len(filtered_docs)} product(s) that meet your criteria:\n\n"
                    answer_text += "\n\n".join(products_info)
                    
                    # Create a response structure that matches normal RAG output
                    response = {
                        'answer': answer_text,
                        'source_documents': filtered_docs
                    }
                else:
                    response = {
                        'answer': f"I couldn't find any products under ${price_threshold} with clear pricing in the current dataset.",
                        'source_documents': []
                    }
            else:
                # Fallback to normal RAG if no price found
                response = ask_question_with_sources(vector_store, constrained_prompt, search_query=enhanced_search_query)
        else:
            # Get the RAG response with sources for non-price queries using enhanced search
            response = ask_question_with_sources(vector_store, constrained_prompt, search_query=enhanced_search_query)
        
        # Extract product information for image display
        products = []
        text_response = ""
        
        if isinstance(response, dict) and 'answer' in response:
            # Clean the response text to avoid formatting issues
            text_response = response['answer'].replace('$', '\\$')
            
            # Handle source documents from both normal RAG and custom price filtering
            source_docs = response.get('source_documents', [])
            sources_list = response.get('sources', [])
            
            # For price queries with pre-filtered documents, extract products directly
            if is_price_query and source_docs and not sources_list:
                for doc in source_docs:
                    metadata = doc.metadata
                    asin = metadata.get('asin', 'N/A')
                    if asin != 'N/A':
                        product_info = {
                            'asin': asin,
                            'title': metadata.get('product_title', metadata.get('title', 'Unknown Product')),
                            'category': metadata.get('main_category', 'Unknown'),
                            'rating': metadata.get('overall', metadata.get('rating', 'N/A')),
                            'price': metadata.get('price', 'N/A'),
                            'sentiment': metadata.get('sentiment', 'Unknown'),
                            'date': str(metadata.get('review_date', 'Unknown'))[:10]
                        }
                        products.append(product_info)
            
            # Only show source data if it adds value to the specific query
            elif sources_list:
                # Get unique products to avoid duplicates
                unique_products = {}
                for source in sources_list:
                    asin = source.get('asin', 'N/A')
                    if asin != 'N/A' and asin not in unique_products:
                        unique_products[asin] = source
                
                # Process products for card display only (no text listing anymore)
                if unique_products:  # Always show products, limit display to 5
                    # text_response += "\\n\\n**Related Products Found:**\\n"  # Removed
                    
                    for i, (asin, source) in enumerate(unique_products.items(), 1):
                        if i > 5:  # Hard limit
                            break
                            
                        product_title = source.get('product_title', source.get('title', 'Unknown Product'))
                        rating = source.get('rating', 'N/A')
                        category = source.get('main_category', 'Unknown')
                        sentiment = source.get('sentiment', 'Unknown')
                        
                        # Advanced filtering based on user query context and content relevance
                        should_include = True
                        
                        # If user asks about delivery/shipping issues, only include products with ACTUAL delivery problems
                        if any(word in prompt.lower() for word in ['delivery', 'shipping', 'arrived', 'delayed', 'late', 'package']):
                            # Check if the source document mentions actual delivery problems
                            source_text = source.get('review_text', '')
                            if not source_text:
                                # Try to find the review text from source_documents
                                source_docs = response.get('source_documents', [])
                                if i-1 < len(source_docs):
                                    source_text = source_docs[i-1].page_content
                            
                            delivery_problem_keywords = [
                                'delayed', 'late delivery', 'slow shipping', 'never arrived', 'lost package', 
                                'damaged package', 'wrong item', 'delivery problem', 'shipping issue', 
                                'took too long', 'still waiting', 'package never', 'delivery failed',
                                'poor packaging', 'broken during shipping'
                            ]
                            has_delivery_problems = any(problem in source_text.lower() for problem in delivery_problem_keywords)
                            if not has_delivery_problems:
                                should_include = False
                        
                        # If user specifically asks for 5-star products, only show 5-star
                        elif any(word in prompt.lower() for word in ['5-star', 'five star', '5 star', 'show me 5', 'perfect rating']):
                            try:
                                if float(rating) < 5.0:
                                    should_include = False
                            except (ValueError, TypeError):
                                should_include = False
                        
                        # If user asks for top/best/highest rated, show 4+ star products
                        elif any(word in prompt.lower() for word in ['top rated', 'best rated', 'highest rated', 'top product', 'best product']):
                            try:
                                if float(rating) < 4.0:
                                    should_include = False
                            except (ValueError, TypeError):
                                should_include = False
                        
                        # If user asks about problems/complaints, ONLY show negative products
                        elif any(word in prompt.lower() for word in ['problem', 'issue', 'complaint', 'complain', 'bad', 'worst', 'terrible', 'hate', 'disappointed', 'awful']):
                            try:
                                # Only include products with low ratings (3 or below) AND negative sentiment
                                if float(rating) > 3.0 or sentiment != 'Negative':
                                    should_include = False
                            except (ValueError, TypeError):
                                # If we can't parse rating, check sentiment only
                                if sentiment != 'Negative':
                                    should_include = False
                        
                        # For other queries, include products but limit to most relevant
                        
                        if should_include:
                            # Add product info to the list for card display
                            product_info = {
                                'asin': asin,
                                'title': product_title,
                                'category': category,
                                'rating': rating,
                                'price': source.get('price', 'N/A'),
                                'sentiment': sentiment,
                                'date': str(source.get('review_date', 'Unknown'))[:10]
                            }
                            products.append(product_info)
                        
                        # Products will only appear as cards below - no text listing
        else:
            text_response = str(response)
        
        # Filter products for price-based queries
        if is_price_query and products:
            import re
            price_match = re.search(r'\$(\d+)', prompt)
            price_threshold = int(price_match.group(1)) if price_match else None
            
            if price_threshold:
                filtered_products = []
                filtered_asins = []
                for product in products:
                    price_str = product.get('price', '')
                    if price_str and price_str != 'N/A':
                        try:
                            # Extract numeric price
                            price_clean = re.sub(r'[^\d.]', '', str(price_str))
                            if price_clean:
                                price_value = float(price_clean)
                                if price_value <= price_threshold:
                                    filtered_products.append(product)
                                    filtered_asins.append(product['asin'])
                        except:
                            continue
                
                products = filtered_products
                
                # Also update the text response to only show filtered products
                if filtered_products:
                    # Replace the "Related Products Found" section with filtered results
                    if "Related Products Found" in text_response:
                        base_text = text_response.split("Related Products Found")[0]
                        text_response = base_text + f"Related Products Under ${price_threshold}:**\n\nFound {len(filtered_products)} product(s) that meet your price criteria:\n"
                        
                        for i, product in enumerate(filtered_products, 1):
                            text_response += f"\n{i}. **{product['title']}** (${product['price']})"
                            text_response += f"\n   • ASIN: {product['asin']} • Rating: {product['rating']}⭐"
        
        return {
            'text': text_response,
            'products': products
        }
        
    except Exception as e:
        return f"⚠️ Sorry, I encountered an error: {str(e)}\n\nPlease check your API keys and try again."

@st.cache_data
def load_product_images_from_metadata():
    """Load product images from metadata file - enhanced with debugging"""
    try:
        import json
        images_dict = {}
        lines_processed = 0
        max_lines = 10000  # Reduced for debugging
        sample_structures = []
        
        with open("data/meta_Appliances.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                if lines_processed >= max_lines:
                    break
                    
                try:
                    item = json.loads(line.strip())
                    asin = item.get('asin')
                    
                    # Collect sample structures for debugging (first 3 items)
                    if lines_processed < 3:
                        sample_structures.append({
                            'asin': asin,
                            'has_images': 'images' in item,
                            'images_type': type(item.get('images', None)).__name__,
                            'keys': list(item.keys())[:10]  # First 10 keys
                        })
                    
                    if asin and 'images' in item and item['images']:
                        images = item['images']
                        
                        # Handle different image structures
                        image_url = None
                        
                        # Structure 1: List of dictionaries
                        if isinstance(images, list) and images:
                            for img in images:
                                if isinstance(img, dict):
                                    # Try different key names
                                    for key in ['large', 'hiRes', 'medium', 'small', 'thumb']:
                                        if key in img and img[key]:
                                            image_url = img[key]
                                            break
                                    if image_url:
                                        break
                                elif isinstance(img, str):  # Direct URL string
                                    image_url = img
                                    break
                        
                        # Structure 2: Direct dictionary
                        elif isinstance(images, dict):
                            for key in ['large', 'hiRes', 'medium', 'small', 'thumb']:
                                if key in images and images[key]:
                                    image_url = images[key]
                                    break
                        
                        # Structure 3: Direct string URL
                        elif isinstance(images, str):
                            image_url = images
                        
                        if image_url:
                            images_dict[asin] = image_url
                            
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    pass
                
                lines_processed += 1
        
        # Return images dictionary without sidebar notifications
        return images_dict
    except Exception as e:
        # Fail silently without sidebar messages
        return {}

def display_product_cards(products):
    """Display product information cards with images"""
    if not products:
        return
        
    st.subheader("🛒 Featured Products")
    
    # Load actual product images from metadata (cached)
    product_images = load_product_images_from_metadata()
    
    # Create columns for product cards
    cols = st.columns(min(len(products), 3))
    
    for i, product in enumerate(products):
        with cols[i % 3]:
            with st.container():
                # Product image from metadata or fallback patterns
                asin = product['asin']
                image_displayed = False
                
                # First try: Use actual image URL from metadata
                if asin in product_images:
                    try:
                        st.image(product_images[asin], width=150, caption=f"ASIN: {asin} (from metadata)")
                        image_displayed = True
                    except Exception as e:
                        st.write(f"⚠️ Could not load metadata image: {str(e)[:50]}")
                        pass
                
                # Fallback: Try Amazon image URL patterns
                if not image_displayed:
                    image_patterns = [
                        f"https://images-na.ssl-images-amazon.com/images/P/{asin}.01.L.jpg",
                        f"https://m.media-amazon.com/images/I/{asin}.jpg", 
                        f"https://images-na.ssl-images-amazon.com/images/P/{asin}.jpg",
                        f"https://images.amazon.com/images/P/{asin}.01.LZZZZZZZ.jpg"
                    ]
                    
                    for pattern in image_patterns:
                        try:
                            st.image(pattern, width=150, caption=f"ASIN: {asin}")
                            image_displayed = True
                            break
                        except:
                            continue
                
                if not image_displayed:
                    # Show a nice placeholder with ASIN
                    st.markdown(f"""
                    <div style='border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 10px;'>
                        <h4>📦 Product Image</h4>
                        <p><strong>ASIN:</strong> {asin}</p>
                        <p><em>Image not available</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Product details
                st.write(f"**{product['title'][:40]}{'...' if len(product['title']) > 40 else ''}**")
                st.write(f"🏷️ {product['category']}")
                st.write(f"⭐ Rating: {product['rating']}")
                st.write(f"😊 Sentiment: {product['sentiment']}")
                
                # Clean price display
                price = product['price']
                if price and str(price).lower() not in ['n/a', 'nan', 'none', '']:
                    st.write(f"💰 {price}")
                else:
                    st.write("💰 Price not listed")
                
                st.write(f"📅 {product['date']}")
                
                # Amazon link
                amazon_url = f"https://www.amazon.com/dp/{asin}"
                st.markdown(f"[🔗 View on Amazon]({amazon_url})")
                
                st.write("---")

    # # Tab 7: Live Sentiment Analysis
    # with tab7:
    #     st.header("🎭 Live Sentiment Analysis Comparison")
    #     st.write("Compare OpenAI GPT vs Transformer-based models for real-time sentiment analysis")
        
    #     # Text input for sentiment analysis
    #     user_text = st.text_area(
    #         "Enter text to analyze sentiment:",
    #         placeholder="Type or paste any text here to analyze its sentiment...",
    #         height=100
    #     )
        
    #     if user_text and st.button("🔍 Analyze Sentiment"):
    #         col1, col2 = st.columns(2)
            
    #         with col1:
    #             st.subheader("🤖 OpenAI GPT Analysis")
    #             with st.spinner("Analyzing with GPT..."):
    #                 try:
    #                     # OpenAI sentiment analysis
    #                     from openai import OpenAI
    #                     client = OpenAI(api_key=st.secrets["openai"]["api_key"])
                        
    #                     gpt_prompt = f"""
    #                     Analyze the sentiment of this text and provide:
    #                     1. Overall sentiment (Positive/Negative/Neutral)
    #                     2. Confidence score (0-1)
    #                     3. Key emotional indicators
    #                     4. Brief explanation
                        
    #                     Text: "{user_text}"
                        
    #                     Respond in JSON format:
    #                     {{
    #                         "sentiment": "positive/negative/neutral",
    #                         "confidence": 0.85,
    #                         "key_emotions": ["joy", "satisfaction"],
    #                         "explanation": "Brief explanation here"
    #                     }}
    #                     """
                        
    #                     response = client.chat.completions.create(
    #                         model="gpt-3.5-turbo",
    #                         messages=[{"role": "user", "content": gpt_prompt}],
    #                         temperature=0.1
    #                     )
                        
    #                     import json
    #                     gpt_result = json.loads(response.choices[0].message.content)
                        
    #                     # Display results
    #                     sentiment_color = {
    #                         "positive": "green",
    #                         "negative": "red", 
    #                         "neutral": "gray"
    #                     }
                        
    #                     st.markdown(f"**Sentiment:** :{sentiment_color[gpt_result['sentiment']]}[{gpt_result['sentiment'].upper()}]")
    #                     st.markdown(f"**Confidence:** {gpt_result['confidence']:.2f}")
    #                     st.markdown(f"**Key Emotions:** {', '.join(gpt_result['key_emotions'])}")
    #                     st.markdown(f"**Explanation:** {gpt_result['explanation']}")
                        
    #                 except Exception as e:
    #                     st.error(f"OpenAI analysis failed: {e}")
            
    #         with col2:
    #             st.subheader("🤗 Transformer Model Analysis")
    #             with st.spinner("Analyzing with Transformers..."):
    #                 try:
    #                     # Transformer sentiment analysis
    #                     from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
                        
    #                     # Load pre-trained sentiment analysis model
    #                     model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    #                     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #                     model = AutoModelForSequenceClassification.from_pretrained(model_name)
                        
    #                     # Create pipeline
    #                     sentiment_pipeline = pipeline(
    #                         "sentiment-analysis",
    #                         model=model,
    #                         tokenizer=tokenizer,
    #                         return_all_scores=True
    #                     )
                        
    #                     # Analyze text
    #                     results = sentiment_pipeline(user_text)
                        
    #                     # Process results
    #                     sentiment_mapping = {
    #                         "LABEL_0": ("negative", "red"),
    #                         "LABEL_1": ("neutral", "gray"), 
    #                         "LABEL_2": ("positive", "green")
    #                     }
                        
    #                     # Find highest confidence prediction
    #                     best_result = max(results[0], key=lambda x: x['score'])
    #                     sentiment_label, color = sentiment_mapping.get(best_result['label'], ("unknown", "gray"))
                        
    #                     st.markdown(f"**Sentiment:** :{color}[{sentiment_label.upper()}]")
    #                     st.markdown(f"**Confidence:** {best_result['score']:.2f}")
                        
    #                     # Show all scores
    #                     st.markdown("**All Scores:**")
    #                     for result in results[0]:
    #                         label, _ = sentiment_mapping.get(result['label'], ("unknown", "gray"))
    #                         st.write(f"- {label.capitalize()}: {result['score']:.3f}")
                            
    #                 except Exception as e:
    #                     st.error(f"Transformer analysis failed: {e}")
    #                     st.info("Installing required packages... Please refresh the page after installation.")
    #                     # Auto-install transformers if needed
    #                     import subprocess
    #                     try:
    #                         subprocess.check_call(["pip", "install", "transformers", "torch", "torchvision", "torchaudio"])
    #                     except:
    #                         st.error("Could not install transformers. Please install manually: pip install transformers torch")
        
    #     st.write("---")
        
        # # Batch analysis section
        # st.subheader("📊 Batch Analysis on Review Dataset")
        
        # if st.button("🔍 Analyze Sample Reviews"):
        #     # Get sample reviews for analysis
        #     sample_reviews = df_filtered.head(5)['reviewText'].dropna().tolist()
            
        #     if sample_reviews:
        #         st.write("**Comparing sentiment analysis on sample reviews:**")
                
        #         for i, review in enumerate(sample_reviews[:3]):  # Limit to 3 for demo
        #             with st.expander(f"Review {i+1}: {review[:100]}..."):
        #                 col1, col2 = st.columns(2)
                        
        #                 with col1:
        #                     st.write("**OpenAI Result:**")
        #                     # Simplified OpenAI analysis for batch
        #                     try:
        #                         from openai import OpenAI
        #                         client = OpenAI(api_key=st.secrets["openai"]["api_key"])
                                
        #                         response = client.chat.completions.create(
        #                             model="gpt-3.5-turbo",
        #                             messages=[{
        #                                 "role": "user", 
        #                                 "content": f"Analyze sentiment (positive/negative/neutral) of: {review[:200]}"
        #                             }],
        #                             temperature=0.1,
        #                             max_tokens=50
        #                         )
                                
        #                         gpt_sentiment = response.choices[0].message.content.strip().lower()
        #                         if "positive" in gpt_sentiment:
        #                             st.markdown(":green[POSITIVE]")
        #                         elif "negative" in gpt_sentiment:
        #                             st.markdown(":red[NEGATIVE]")
        #                         else:
        #                             st.markdown(":gray[NEUTRAL]")
                                    
        #                     except Exception as e:
        #                         st.error(f"GPT failed: {e}")
                        
        #                 with col2:
        #                     st.write("**Transformer Result:**")
        #                     try:
        #                         # Use the transformer pipeline
        #                         results = sentiment_pipeline(review[:512])  # Limit text length
        #                         best_result = max(results[0], key=lambda x: x['score'])
                                
        #                         sentiment_mapping = {
        #                             "LABEL_0": ("NEGATIVE", "red"),
        #                             "LABEL_1": ("NEUTRAL", "gray"), 
        #                             "LABEL_2": ("POSITIVE", "green")
        #                         }
                                
        #                         sentiment_label, color = sentiment_mapping.get(best_result['label'], ("UNKNOWN", "gray"))
        #                         st.markdown(f":{color}[{sentiment_label}] ({best_result['score']:.2f})")
                                
        #                     except Exception as e:
        #                         st.error(f"Transformer failed: {e}")
        #     else:
        #         st.warning("No reviews available for analysis")

def analyze_dataset_directly(prompt, df):
    """Direct analysis of dataset when RAG fails - works with actual data"""
    try:
        prompt_lower = prompt.lower()
        products = []
        
        # Analyze for delivery issues specifically - look for ACTUAL PROBLEMS
        if any(word in prompt_lower for word in ['delivery', 'shipping', 'arrived', 'delayed', 'late']):
            # Search for delivery PROBLEMS, not just mentions
            delivery_problem_keywords = [
                'delayed', 'late delivery', 'slow shipping', 'never arrived', 'lost package', 
                'damaged package', 'wrong item', 'delivery problem', 'shipping issue', 
                'took too long', 'still waiting', 'package never', 'delivery failed',
                'poor packaging', 'broken during shipping', 'delivery disaster'
            ]
            
            # Find reviews that mention actual delivery problems
            delivery_issues = []
            for _, row in df.iterrows():
                review_text = str(row.get('review_text', '')).lower()
                if any(problem in review_text for problem in delivery_problem_keywords):
                    delivery_issues.append(row)
            
            if len(delivery_issues) > 0:
                response_text = f"**📦 Actual Delivery Issues Found in {len(delivery_issues)} Reviews:**\n\n"
                
                for i, row in enumerate(delivery_issues[:5], 1):
                    product_title = row.get('product_title', 'Unknown Product')
                    asin = row.get('asin', 'N/A')
                    review_text = row.get('review_text', '')
                    rating = row.get('rating', 'N/A')
                    
                    # Find the specific delivery problem sentence
                    sentences = review_text.split('.')
                    problem_sentence = ""
                    for sentence in sentences:
                        if any(problem in sentence.lower() for problem in delivery_problem_keywords):
                            problem_sentence = sentence.strip()
                            break
                    
                    response_text += f"**{i}. {product_title}** ({rating}⭐)\n"
                    response_text += f"   • ASIN: {asin}\n"
                    response_text += f"   • Delivery issue: \"{problem_sentence[:100]}{'...' if len(problem_sentence) > 100 else ''}\"\n\n"
                    
                    if asin != 'N/A':
                        products.append({
                            'asin': asin,
                            'title': product_title,
                            'category': row.get('main_category', 'Unknown'),
                            'rating': row.get('rating', 'N/A'),
                            'price': row.get('price', 'N/A'),
                            'sentiment': row.get('sentiment', 'Unknown'),
                            'date': str(row.get('review_date', 'Unknown'))[:10]
                        })
            else:
                response_text = "**📦 No Delivery Issues Found**\n\nI searched through all reviews in the dataset and found no delivery issues or shipping problems reported by customers."
        
        # Analyze for top-rated products
        elif any(word in prompt_lower for word in ['top', 'best', 'highest', 'rated', 'star']):
            # Get highest rated products
            top_products = df.nlargest(5, 'rating')
            
            response_text = f"**🏆 Top-Rated Products in Your Dataset:**\n\n"
            response_text += f"Found {len(top_products)} highly-rated products:\n\n"
            
            for i, (_, row) in enumerate(top_products.iterrows(), 1):
                product_title = row.get('product_title', 'Unknown Product')
                rating = row.get('rating', 'N/A')
                asin = row.get('asin', 'N/A')
                category = row.get('main_category', 'Unknown')
                sentiment = row.get('sentiment', 'Unknown')
                
                response_text += f"**{i}. {product_title}**\n"
                response_text += f"   • Rating: {rating}⭐ | Category: {category}\n"
                response_text += f"   • ASIN: {asin} | Sentiment: {sentiment}\n\n"
                
                if asin != 'N/A':
                    products.append({
                        'asin': asin,
                        'title': product_title,
                        'category': category,
                        'rating': rating,
                        'price': row.get('price', 'N/A'),
                        'sentiment': sentiment,
                        'date': str(row.get('review_date', 'Unknown'))[:10]
                    })
        
        # Analyze for 5-star products specifically
        elif '5' in prompt_lower and 'star' in prompt_lower:
            five_star = df[df['rating'] == 5.0]
            
            if len(five_star) > 0:
                response_text = f"**⭐ 5-Star Products in Your Dataset:**\n\n"
                response_text += f"Found {len(five_star)} products with perfect 5-star ratings:\n\n"
                
                for i, (_, row) in enumerate(five_star.head(5).iterrows(), 1):
                    product_title = row.get('product_title', 'Unknown Product')
                    asin = row.get('asin', 'N/A')
                    category = row.get('main_category', 'Unknown')
                    
                    response_text += f"**{i}. {product_title}**\n"
                    response_text += f"   • Category: {category} | ASIN: {asin}\n"
                    response_text += f"   • Review: \"{row.get('review_title', 'No title')}\"\n\n"
                    
                    if asin != 'N/A':
                        products.append({
                            'asin': asin,
                            'title': product_title,
                            'category': category,
                            'rating': 5.0,
                            'price': row.get('price', 'N/A'),
                            'sentiment': row.get('sentiment', 'Positive'),
                            'date': str(row.get('review_date', 'Unknown'))[:10]
                        })
            else:
                response_text = "No 5-star products found in the current dataset."
        
        # Category comparison analysis - simplified and readable
        elif any(word in prompt_lower for word in ['compare', 'categories', 'category', 'different categories']):
            response_text = f"**📊 Category Comparison Analysis:**\n\n"
            
            # Get category statistics
            category_stats = df.groupby('main_category').agg({
                'rating': ['count', 'mean'],
                'sentiment': lambda x: (x == 'Positive').sum(),
                'asin': 'nunique'
            }).round(2)
            
            # Flatten column names
            category_stats.columns = ['Total_Reviews', 'Avg_Rating', 'Positive_Reviews', 'Unique_Products']
            category_stats = category_stats.sort_values('Total_Reviews', ascending=False)
            
            # Add AI insights and rankings
            total_categories = len(category_stats)
            total_reviews_all = category_stats['Total_Reviews'].sum()
            
            # Calculate sentiment percentages and create enhanced analysis
            category_analysis = []
            for category in category_stats.index:
                cat_df = df[df['main_category'] == category]
                total_reviews = len(cat_df)
                positive_pct = (cat_df['sentiment'] == 'Positive').sum() / total_reviews * 100
                negative_pct = (cat_df['sentiment'] == 'Negative').sum() / total_reviews * 100
                avg_rating = category_stats.loc[category, 'Avg_Rating']
                
                category_analysis.append({
                    'name': category,
                    'total_reviews': total_reviews,
                    'avg_rating': avg_rating,
                    'positive_pct': positive_pct,
                    'negative_pct': negative_pct,
                    'unique_products': category_stats.loc[category, 'Unique_Products'],
                    'market_share': (total_reviews / total_reviews_all) * 100
                })
            
            # AI-driven executive summary
            response_text += "**� Executive Summary:**\n"
            
            # Find best and worst performers
            best_rating = max(category_analysis, key=lambda x: x['avg_rating'])
            best_sentiment = max(category_analysis, key=lambda x: x['positive_pct'])
            largest_category = max(category_analysis, key=lambda x: x['total_reviews'])
            most_problematic = max(category_analysis, key=lambda x: x['negative_pct'])
            
            response_text += f"• **🏆 Highest Quality:** {best_rating['name']} leads with {best_rating['avg_rating']:.2f}⭐ average rating\n"
            response_text += f"• **😊 Most Loved:** {best_sentiment['name']} has {best_sentiment['positive_pct']:.1f}% positive sentiment\n"
            response_text += f"• **📊 Market Leader:** {largest_category['name']} dominates with {largest_category['market_share']:.1f}% of reviews\n"
            response_text += f"• **⚠️ Needs Attention:** {most_problematic['name']} has {most_problematic['negative_pct']:.1f}% negative feedback\n\n"
            
            # Smart category rankings
            response_text += "**📈 Smart Rankings:**\n"
            
            # Rating ranking
            rating_ranked = sorted(category_analysis, key=lambda x: x['avg_rating'], reverse=True)
            response_text += "🌟 **By Quality (Average Rating):**\n"
            for i, cat in enumerate(rating_ranked[:3], 1):
                quality_indicator = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                response_text += f"   {quality_indicator} {cat['name']}: {cat['avg_rating']:.2f}⭐\n"
            
            # Sentiment ranking
            sentiment_ranked = sorted(category_analysis, key=lambda x: x['positive_pct'], reverse=True)
            response_text += "\n💚 **By Customer Satisfaction (Positive Sentiment):**\n"
            for i, cat in enumerate(sentiment_ranked[:3], 1):
                satisfaction_indicator = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                response_text += f"   {satisfaction_indicator} {cat['name']}: {cat['positive_pct']:.1f}% positive\n"
            
            # Volume ranking
            volume_ranked = sorted(category_analysis, key=lambda x: x['total_reviews'], reverse=True)
            response_text += "\n📊 **By Market Presence (Review Volume):**\n"
            for i, cat in enumerate(volume_ranked[:3], 1):
                market_indicator = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                response_text += f"   {market_indicator} {cat['name']}: {cat['total_reviews']} reviews ({cat['market_share']:.1f}% market share)\n"
            
            response_text += "\n**🔍 Detailed Category Breakdown:**\n\n"
            
            # Detailed analysis with AI insights
            for cat_data in rating_ranked:
                category = cat_data['name']
                cat_df = df[df['main_category'] == category]
                
                # Generate AI insights based on data
                insights = []
                if cat_data['avg_rating'] >= 4.5:
                    insights.append("🌟 Premium quality category")
                elif cat_data['avg_rating'] <= 3.0:
                    insights.append("⚠️ Quality concerns evident")
                
                if cat_data['positive_pct'] >= 80:
                    insights.append("😊 High customer satisfaction")
                elif cat_data['positive_pct'] <= 50:
                    insights.append("😟 Customer satisfaction challenges")
                
                if cat_data['market_share'] >= 30:
                    insights.append("📊 Market dominant")
                elif cat_data['market_share'] <= 5:
                    insights.append("🔍 Niche category")
                
                response_text += f"**🏷️ {category}**\n"
                response_text += f"   • Reviews: {cat_data['total_reviews']} ({cat_data['market_share']:.1f}% market share)\n"
                response_text += f"   • Quality: {cat_data['avg_rating']:.2f}⭐ | Products: {cat_data['unique_products']}\n"
                response_text += f"   • Sentiment: {cat_data['positive_pct']:.1f}% Positive, {cat_data['negative_pct']:.1f}% Negative\n"
                
                if insights:
                    response_text += f"   • AI Insights: {' | '.join(insights)}\n"
                
                # Show top product in this category
                top_product = cat_df.nlargest(1, 'rating').iloc[0] if len(cat_df) > 0 else None
                if top_product is not None:
                    response_text += f"   • Best Product: {top_product.get('product_title', 'Unknown')[:40]}... ({top_product.get('rating', 'N/A')}⭐)\n"
                
                response_text += "\n"
            
            # Strategic recommendations
            response_text += "**💡 Strategic Recommendations:**\n"
            if best_rating['avg_rating'] - most_problematic['avg_rating'] > 1.0:
                response_text += f"• Focus on improving {most_problematic['name']} quality - significant gap vs. {best_rating['name']}\n"
            
            if largest_category['market_share'] > 40:
                response_text += f"• {largest_category['name']} dominance creates opportunity in underserved categories\n"
            
            if any(cat['negative_pct'] > 30 for cat in category_analysis):
                problematic_cats = [cat['name'] for cat in category_analysis if cat['negative_pct'] > 30]
                response_text += f"• Address quality issues in: {', '.join(problematic_cats)}\n"
            
            # Add products for display - top products from each major category
            major_categories = category_stats.head(3).index
            for category in major_categories:
                cat_df = df[df['main_category'] == category]
                top_product = cat_df.nlargest(1, 'rating').iloc[0] if len(cat_df) > 0 else None
                if top_product is not None:
                    products.append({
                        'asin': top_product.get('asin', 'N/A'),
                        'title': top_product.get('product_title', 'Unknown Product'),
                        'category': category,
                        'rating': top_product.get('rating', 'N/A'),
                        'price': top_product.get('price', 'N/A'),
                        'sentiment': top_product.get('sentiment', 'Unknown'),
                        'date': str(top_product.get('review_date', 'Unknown'))[:10]
                    })
        
        # Complaint analysis - focus on negative reviews and low ratings
        elif any(word in prompt_lower for word in ['complain', 'complaint', 'what do customers complain', 'what are the complaints', 'customer complaints', 'problems', 'issues']):
            # Find negative reviews and low-rated products
            negative_reviews = df[(df['sentiment'] == 'Negative') | (df['rating'] <= 3.0)]
            
            if len(negative_reviews) > 0:
                response_text = f"**😞 Customer Complaints Analysis:**\n\n"
                response_text += f"Found {len(negative_reviews)} negative reviews/low-rated products out of {len(df)} total reviews.\n\n"
                
                # Group complaints by category
                complaint_by_category = negative_reviews.groupby('main_category').agg({
                    'rating': ['count', 'mean'],
                    'asin': 'nunique'
                }).round(2)
                complaint_by_category.columns = ['Complaint_Count', 'Avg_Rating', 'Products_With_Issues']
                complaint_by_category = complaint_by_category.sort_values('Complaint_Count', ascending=False)
                
                response_text += "**📊 Complaints by Category:**\n"
                for category, stats in complaint_by_category.head(5).iterrows():
                    response_text += f"• **{category}**: {int(stats['Complaint_Count'])} complaints | Avg: {stats['Avg_Rating']}⭐ | {int(stats['Products_With_Issues'])} products\n"
                
                response_text += "\n**🔍 Specific Product Issues:**\n"
                
                # Show worst products
                worst_products = negative_reviews.nsmallest(5, 'rating')
                for i, (_, row) in enumerate(worst_products.iterrows(), 1):
                    product_title = row.get('product_title', 'Unknown Product')
                    rating = row.get('rating', 'N/A')
                    asin = row.get('asin', 'N/A') 
                    review_title = row.get('review_title', 'No review title')
                    
                    response_text += f"**{i}. {product_title[:50]}{'...' if len(product_title) > 50 else ''}** ({rating}⭐)\n"
                    response_text += f"   • ASIN: {asin}\n"
                    response_text += f"   • Issue: \"{review_title}\"\n\n"
                    
                    if asin != 'N/A':
                        products.append({
                            'asin': asin,
                            'title': product_title,
                            'category': row.get('main_category', 'Unknown'),
                            'rating': rating,
                            'price': row.get('price', 'N/A'),
                            'sentiment': row.get('sentiment', 'Negative'),
                            'date': str(row.get('review_date', 'Unknown'))[:10]
                        })
            else:
                response_text = "**😊 Great News!**\n\nNo significant complaints found in the current dataset. Most customers seem satisfied with their purchases!"
        
        # General product analysis
        else:
            response_text = f"**📊 Dataset Analysis:**\n\n"
            response_text += f"• Total reviews: {len(df)}\n"
            response_text += f"• Average rating: {df['rating'].mean():.2f}⭐\n"
            response_text += f"• Unique products: {df['asin'].nunique()}\n"
            response_text += f"• Top category: {df['main_category'].value_counts().index[0]}\n\n"
            
            # Show some top products
            top_3 = df.nlargest(3, 'rating')
            response_text += "**Top 3 Products:**\n"
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                response_text += f"{i}. {row.get('product_title', 'Unknown')} ({row.get('rating', 'N/A')}⭐)\n"
        
        return {
            'text': response_text,
            'products': products
        }
        
    except Exception as e:
        return {
            'text': f"⚠️ Error analyzing dataset: {str(e)}",
            'products': []
        }

def get_product_image_from_metadata(asin):
    """
    Future enhancement: Get actual product image from metadata file
    This function would:
    1. Load the meta_Appliances.jsonl file
    2. Search for the ASIN 
    3. Extract the 'large' image URL from the 'images' field
    4. Return the actual product image URL
    """
    # Placeholder for future implementation
    return None

# Legacy function removed - now using LangChain RAG system above

if __name__ == "__main__":
    main()
