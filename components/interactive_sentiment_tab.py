# [file name]: components/interactive_sentiment_tab.py
"""
Interactive Sentiment Detective - Deep dive into sentiment analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.settings import SENTIMENT_COLORS
import numpy as np
import os
import re

def get_sentiment_analyst():
    """Create sentiment analysis specialist"""
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    except Exception as e:
        st.error(f"Failed to create sentiment analyst: {str(e)}")
        return None

def analyze_general_sentiment(df, insights):
    """Fallback analysis when specific analysis fails"""
    try:
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            total = len(df)
            
            insights.append("")
            insights.append("📊 **Overall Sentiment Distribution:**")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total) * 100
                insights.append(f"   {sentiment}: {percentage:.1f}% ({count:,} reviews)")
            
            visualization_data = sentiment_counts
            chart_type = "sentiment_distribution"
            
            return insights, visualization_data, chart_type
        else:
            insights.append("❌ No sentiment data available")
            return insights, None, None
    except Exception as e:
        insights.append(f"❌ Fallback analysis failed: {str(e)}")
        return insights, None, None

def analyze_sentiment_patterns(df, query):
    """Enhanced sentiment analysis with price range support and data cleaning"""
    query_lower = query.lower()
    
    # Initialize results
    insights = []
    visualization_data = None
    chart_type = None

    try:
        # PRICE RANGE SENTIMENT ANALYSIS - WITH DEBUGGING
        if any(word in query_lower for word in ['price', 'priced', 'expensive', 'cost', 'high-priced', 'budget']):
            if 'price' in df.columns:
                # DEBUG: Show what's actually in the price column
                price_sample = df['price'].head(10).tolist()
                price_types = df['price'].apply(type).value_counts()
                
                insights.append("🔍 **Price Data Debug Info:**")
                insights.append(f"   Sample values: {price_sample}")
                insights.append(f"   Data types: {dict(price_types)}")
                insights.append(f"   Total rows: {len(df)}")
                insights.append(f"   Non-null prices: {df['price'].notna().sum()}")
                
                # Try to clean price data
                df_clean = df.copy()
                
                # Handle different price formats
                def clean_price(value):
                    if pd.isna(value):
                        return None
                    if isinstance(value, (int, float)):
                        return float(value)
                    if isinstance(value, str):
                        # Remove currency symbols and commas
                        cleaned = re.sub(r'[^\d.]', '', str(value))
                        try:
                            return float(cleaned) if cleaned else None
                        except:
                            return None
                    return None
                
                df_clean['price_clean'] = df_clean['price'].apply(clean_price)
                df_clean = df_clean[df_clean['price_clean'].notna()]
                df_clean = df_clean[df_clean['price_clean'] >= 0]
                
                insights.append(f"   Valid cleaned prices: {len(df_clean)}")
                
                if len(df_clean) == 0:
                    insights.append("❌ **No valid price data available**")
                    insights.append("💡 **Try asking about**: sentiment by category, rating, or over time instead")
                    
                    # Fallback to general sentiment analysis
                    return analyze_general_sentiment(df, insights)
                
                # Continue with price analysis if we have valid data
                max_price = df_clean['price_clean'].max()
                insights.append(f"   Price range: ${df_clean['price_clean'].min():.2f} - ${max_price:.2f}")
                
                # Define price ranges
                if max_price <= 50:
                    price_bins = [0, 10, 25, 50]
                    price_labels = ['$0-10', '$10-25', '$25-50']
                elif max_price <= 200:
                    price_bins = [0, 25, 50, 100, 200]
                    price_labels = ['$0-25', '$25-50', '$50-100', '$100-200']
                else:
                    price_bins = [0, 50, 100, 200, 500, float('inf')]
                    price_labels = ['$0-50', '$50-100', '$100-200', '$200-500', '$500+']
                
                df_clean['price_range'] = pd.cut(df_clean['price_clean'], bins=price_bins, labels=price_labels, right=False)
                
                # Calculate sentiment distribution
                price_sentiment = pd.crosstab(df_clean['price_range'], df_clean['sentiment'], normalize='index') * 100
                price_counts = df_clean['price_range'].value_counts().sort_index()
                
                # Generate insights
                insights.append("")
                insights.append("💰 **Price-Sentiment Analysis:**")
                
                if 'Positive' in price_sentiment.columns:
                    best_price = price_sentiment['Positive'].idxmax()
                    best_score = price_sentiment['Positive'].max()
                    worst_price = price_sentiment['Positive'].idxmin() 
                    worst_score = price_sentiment['Positive'].min()
                    
                    insights.append(f"🏆 **Most Positive**: {best_price} ({best_score:.1f}% positive)")
                    insights.append(f"⚠️ **Least Positive**: {worst_price} ({worst_score:.1f}% positive)")
                    
                    # High-price analysis
                    high_price_labels = [label for label in price_labels if any(x in label for x in ['100', '200', '500', '+'])]
                    insights.append("🔍 **High-Priced Products:**")
                    for high_range in high_price_labels:
                        if high_range in price_sentiment.index:
                            high_sentiment = price_sentiment.loc[high_range, 'Positive']
                            count = price_counts.get(high_range, 0)
                            insights.append(f"   {high_range}: {high_sentiment:.1f}% positive ({count:,} reviews)")
                
                insights.append("")
                insights.append("📋 **All Price Ranges:**")
                for price_range in price_labels:
                    if price_range in price_counts.index:
                        count = price_counts[price_range]
                        pos_pct = price_sentiment.loc[price_range, 'Positive'] if price_range in price_sentiment.index and 'Positive' in price_sentiment.columns else 0
                        insights.append(f"   {price_range}: {count:,} reviews ({pos_pct:.1f}% positive)")
                
                visualization_data = price_sentiment
                chart_type = "price_sentiment"
                
            else:
                insights.append("❌ **Price column not found in dataset**")
                # Fallback to general analysis
                return analyze_general_sentiment(df, insights)

        # TIME-BASED SENTIMENT ANALYSIS
        elif any(word in query_lower for word in ['trend', 'over time', 'month', 'year', 'change']):
            if 'review_date' in df.columns:
                # Clean date data
                df_clean = df.copy()
                df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], errors='coerce')
                df_clean = df_clean[df_clean['review_date'].notna()]
                
                if len(df_clean) == 0:
                    insights.append("❌ No valid date data available")
                    return analyze_general_sentiment(df, insights)
                
                monthly_sentiment = df_clean.groupby([
                    df_clean['review_date'].dt.to_period('M'),
                    'sentiment'
                ]).size().unstack(fill_value=0)
                monthly_sentiment.index = monthly_sentiment.index.to_timestamp()

                # Calculate trends
                if 'Positive' in monthly_sentiment.columns:
                    pos_counts = monthly_sentiment['Positive']
                    pct_changes = pos_counts.pct_change()
                    valid_changes = pct_changes[~(pos_counts.shift(1) == 0) & pct_changes.notnull() & ~pct_changes.isin([np.inf, -np.inf])]
                    if not valid_changes.empty:
                        pos_trend = valid_changes.mean() * 100
                        insights.append(f"📈 Positive reviews trend: {pos_trend:+.1f}% monthly change")
                    else:
                        insights.append("📈 Positive reviews trend: N/A (insufficient data)")

                visualization_data = monthly_sentiment
                chart_type = "sentiment_trends"
            else:
                insights.append("❌ Date data not available")
                return analyze_general_sentiment(df, insights)
        
        # CATEGORY SENTIMENT ANALYSIS
        elif any(word in query_lower for word in ['category', 'categories', 'by type']):
            if 'main_category' in df.columns:
                # Clean category data
                df_clean = df.copy()
                df_clean = df_clean[df_clean['main_category'].notna()]
                
                top_categories = df_clean['main_category'].value_counts().head(8).index
                neg_counts = df_clean[df_clean['main_category'].isin(top_categories) & (df_clean['sentiment'] == 'Negative')]['main_category'].value_counts()
                category_sentiment = pd.crosstab(
                    df_clean[df_clean['main_category'].isin(top_categories)]['main_category'],
                    df_clean['sentiment'],
                    normalize='index'
                ) * 100

                if any(word in query_lower for word in ['most negative', 'negative reviews', 'worst reviews']):
                    if not neg_counts.empty:
                        most_neg_cat = neg_counts.idxmax()
                        most_neg_count = neg_counts.max()
                        insights.append(f"🚨 Most negative reviews: {most_neg_cat} ({most_neg_count:,} negative reviews)")
                    visualization_data = neg_counts
                    chart_type = "negative_review_counts"
                else:
                    if 'Positive' in category_sentiment.columns:
                        best_cat = category_sentiment['Positive'].idxmax()
                        best_score = category_sentiment['Positive'].max()
                        worst_cat = category_sentiment['Positive'].idxmin()
                        worst_score = category_sentiment['Positive'].min()
                        insights.append(f"🏆 Most positive: {best_cat} ({best_score:.1f}% positive)")
                        insights.append(f"⚠️ Least positive: {worst_cat} ({worst_score:.1f}% positive)")
                    visualization_data = category_sentiment
                    chart_type = "category_sentiment"
            else:
                insights.append("❌ Category data not available")
                return analyze_general_sentiment(df, insights)
        
        # RATING VS SENTIMENT CORRELATION
        elif any(word in query_lower for word in ['rating', 'stars', 'score']):
            if 'rating' in df.columns:
                # Clean rating data
                df_clean = df.copy()
                df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
                df_clean = df_clean[df_clean['rating'].notna()]
                
                rating_sentiment = df_clean.groupby('rating')['sentiment'].value_counts(normalize=True).unstack() * 100
                
                insights.append("⭐ Rating-Sentiment Correlation:")
                for rating in sorted(df_clean['rating'].unique()):
                    if rating in rating_sentiment.index and 'Positive' in rating_sentiment.columns:
                        pos_pct = rating_sentiment.loc[rating, 'Positive']
                        insights.append(f"  {rating} stars: {pos_pct:.1f}% positive")
                
                visualization_data = rating_sentiment
                chart_type = "rating_sentiment"
            else:
                insights.append("❌ Rating data not available")
                return analyze_general_sentiment(df, insights)
        
        # GENERAL SENTIMENT OVERVIEW
        else:
            return analyze_general_sentiment(df, insights)
        
        return insights, visualization_data, chart_type
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        insights.append(f"❌ Analysis error: {str(e)}")
        return analyze_general_sentiment(df, insights)

def create_sentiment_visualization(data, chart_type, query):
    """Enhanced visualization with price range support"""
    try:
        if chart_type == "price_sentiment":
            # Stacked bar chart for price ranges
            fig = px.bar(data, 
                        title=f"Sentiment by Price Range: {query}",
                        color_discrete_map=SENTIMENT_COLORS,
                        barmode='stack',
                        labels={'value': 'Percentage (%)', 'price_range': 'Price Range'})
            fig.update_xaxes(tickangle=0)
            fig.update_layout(showlegend=True)
            
        elif chart_type == "sentiment_trends":
            fig = px.area(data, title=f"Sentiment Trends: {query}",
                         color_discrete_map=SENTIMENT_COLORS,
                         labels={'value': 'Number of Reviews', 'review_date': 'Date'})
        
        elif chart_type == "category_sentiment":
            fig = px.bar(data, title=f"Sentiment by Category: {query}",
                        color_discrete_map=SENTIMENT_COLORS,
                        barmode='stack',
                        labels={'value': 'Percentage (%)', 'main_category': 'Category'})
            fig.update_xaxes(tickangle=45)
        
        elif chart_type == "negative_review_counts":
            fig = px.bar(data,
                        title=f"Negative Review Counts by Category: {query}",
                        labels={'value': 'Negative Reviews', 'main_category': 'Category'})
            fig.update_xaxes(tickangle=45)
        
        elif chart_type == "rating_sentiment":
            fig = px.bar(data, title=f"Sentiment by Rating: {query}",
                        color_discrete_map=SENTIMENT_COLORS,
                        barmode='stack',
                        labels={'value': 'Percentage (%)', 'rating': 'Rating'})
        
        elif chart_type == "sentiment_distribution":
            fig = px.pie(data, values=data.values, names=data.index,
                        title=f"Sentiment Distribution: {query}",
                        color=data.index,
                        color_discrete_map=SENTIMENT_COLORS)
        
        else:
            # Default bar chart
            fig = px.bar(x=data.index, y=data.values, 
                        title=f"Sentiment Analysis: {query}",
                        labels={'x': 'Category', 'y': 'Count'})
        
        return fig
        
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

def generate_sentiment_insights(llm, query, data_insights, df):
    """Generate AI-powered sentiment insights with better context"""
    try:
        # Prepare enhanced context
        context = "\n".join(data_insights)
        
        # Add dataset context
        dataset_info = f"""
Dataset Overview:
- Total reviews: {len(df):,}
- Overall positive rate: {(df['sentiment'] == 'Positive').mean()*100:.1f}%
"""
        
        # Add specific columns info
        if 'price' in df.columns:
            valid_prices = pd.to_numeric(df['price'], errors='coerce').notna().sum()
            dataset_info += f"- Valid prices: {valid_prices:,}\n"
        
        if 'review_date' in df.columns:
            dataset_info += f"- Date range: {df['review_date'].min().strftime('%Y-%m-%d')} to {df['review_date'].max().strftime('%Y-%m-%d')}\n"
        
        if 'main_category' in df.columns:
            dataset_info += f"- Categories: {df['main_category'].nunique()}\n"

        prompt = f"""As a sentiment analysis expert, analyze this Amazon reviews data and provide SPECIFIC insights.

USER QUERY: {query}

DATA ANALYSIS RESULTS:
{context}

DATASET CONTEXT:
{dataset_info}

Please provide CONCRETE, ACTIONABLE insights:

1. **Key Findings**: What specific patterns did we discover? Use numbers from the data.
2. **Business Implications**: What should the business do based on these findings?
3. **Recommendations**: Specific, actionable recommendations.
4. **Potential Risks/Opportunities**: What to watch out for or capitalize on.

Focus on being SPECIFIC and DATA-DRIVEN. Use the actual numbers from the analysis:"""

        response = llm.predict(prompt)
        return response
        
    except Exception as e:
        return f"❌ Could not generate AI insights: {str(e)}"

def render_interactive_sentiment_tab(df_filtered):
    """Render interactive sentiment analysis tab"""
    
    st.header("🎭 Sentiment Detective")
    st.markdown("Deep dive into customer sentiments! Ask specific questions about emotions, trends, and patterns.")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("🔑 OpenAI API key required")
        return
    
    # Initialize sentiment analyst
    if "sentiment_llm" not in st.session_state:
        with st.spinner("🎭 Initializing sentiment detective..."):
            llm = get_sentiment_analyst()
            if llm:
                st.session_state.sentiment_llm = llm
                st.success("✅ Sentiment detective ready!")
            else:
                st.error("❌ Failed to initialize sentiment analysis")
                return
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pos_pct = (df_filtered['sentiment'] == 'Positive').mean() * 100
        st.metric("😊 Positive", f"{pos_pct:.1f}%")
    with col2:
        neg_pct = (df_filtered['sentiment'] == 'Negative').mean() * 100
        st.metric("😞 Negative", f"{neg_pct:.1f}%")
    with col3:
        neutral_pct = (df_filtered['sentiment'] == 'Neutral').mean() * 100
        st.metric("😐 Neutral", f"{neutral_pct:.1f}%")
    with col4:
        st.metric("📊 Total", f"{len(df_filtered):,}")
    
    # Example sentiment questions
    with st.expander("💡 Try These Sentiment Questions"):
        examples = [
            "How has customer sentiment changed over time?",
            "Which categories have the most negative reviews?",
            "What's the relationship between ratings and sentiment?",
            "Show sentiment distribution by month",
            "Which products have improving sentiment trends?",
            "Compare sentiment between verified and unverified purchases",
            "What's the sentiment pattern for high-priced products?",
            "Show me sentiment trends for kitchen appliances"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            col = cols[i % 2]
            if col.button(example, key=f"sent_ex_{i}"):
                st.session_state.sentiment_query = example
    
    # Query input
    query = st.text_input(
        "Ask about customer sentiments:",
        value=st.session_state.get('sentiment_query', ''),
        placeholder="e.g., 'How has sentiment changed over time?' or 'Which categories have the worst sentiment?'"
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary") and query.strip():
        with st.spinner("🎭 Analyzing sentiment patterns..."):
            # Analyze sentiment patterns
            data_insights, viz_data, chart_type = analyze_sentiment_patterns(df_filtered, query)
            
            # Generate AI insights
            ai_insights = generate_sentiment_insights(
                st.session_state.sentiment_llm, query, data_insights, df_filtered
            )
            
            # Store in history
            if "sentiment_history" not in st.session_state:
                st.session_state.sentiment_history = []
            
            st.session_state.sentiment_history.append({
                "question": query,
                "data_insights": data_insights,
                "ai_insights": ai_insights,
                "visualization_data": viz_data,
                "chart_type": chart_type,
                "timestamp": pd.Timestamp.now().strftime("%H:%M")
            })
            
            # Clear input
            if 'sentiment_query' in st.session_state:
                del st.session_state.sentiment_query
        
        st.rerun()
    
    # Display current analysis
    if st.session_state.get('sentiment_history'):
        latest = st.session_state.sentiment_history[-1]
        
        st.markdown("---")
        st.subheader("📊 Sentiment Analysis Results")
        
        # Question
        st.markdown(f"**❓ Question:** {latest['question']}")
        
        # Data insights
        st.markdown("**📈 Data Analysis:**")
        for insight in latest['data_insights']:
            st.markdown(f"- {insight}")
        
        # AI insights
        st.markdown("**🤖 Expert Insights:**")
        st.markdown(latest['ai_insights'])
        
        # Visualization
        if latest['visualization_data'] is not None:
            st.subheader("📊 Visualization")
            fig = create_sentiment_visualization(
                latest['visualization_data'],
                latest['chart_type'],
                latest['question']
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Show history
    if len(st.session_state.get('sentiment_history', [])) > 1:
        with st.expander("📝 Sentiment Analysis History"):
            for i, item in enumerate(reversed(st.session_state.sentiment_history[:-1])):
                st.markdown(f"**{i+1}. {item['question']}**")
                st.caption(f"Analyzed at {item['timestamp']}")
                st.markdown("---")