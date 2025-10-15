# [file name]: components/advanced_tab.py
"""
Advanced Analytics Tab - Review length, price analysis, sentiment deep dive
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from config.settings import SENTIMENT_COLORS, PRICE_RANGES, PRICE_LABELS, REVIEW_LENGTH_BINS, REVIEW_LENGTH_LABELS

def render_advanced_tab(df_filtered):
    """Render advanced analytics tab with deep insights"""
    
    # Main analytics section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Review Length Analysis")
        
        # Create a copy to avoid SettingWithCopyWarning
        df_analysis = df_filtered.copy()
        
        # Find available text column
        text_column = None
        possible_text_columns = ['reviewText', 'review_text', 'summary', 'text', 'review', 'content']
        
        for col in possible_text_columns:
            if col in df_analysis.columns:
                text_column = col
                break
        
        if text_column is None:
            st.error("❌ No text column found for length analysis. Available columns: " + ", ".join(df_analysis.columns.tolist()))
            return
        
        # Length analysis option selector
        length_metric = st.radio(
            "Choose length metric:",
            options=["Characters", "Words"],
            horizontal=True,
            help="Characters: total character count | Words: word count (more intuitive)"
        )
        
        if length_metric == "Words":
            # Word count calculation
            df_analysis['review_length'] = df_analysis[text_column].str.split().str.len()
            # Word-based bins: Short (<20), Medium (20-75), Long (75-250), Very Long (>250)
            word_bins = [0, 20, 75, 250, float('inf')]
            word_labels = ['Short (<20 words)', 'Medium (20-75 words)', 'Long (75-250 words)', 'Very Long (>250 words)']
            df_analysis['length_category'] = pd.cut(
                df_analysis['review_length'], 
                bins=word_bins,
                labels=word_labels
            )
            length_unit = "words"
        else:
            # Character count calculation (original)
            df_analysis['review_length'] = df_analysis[text_column].str.len()
            df_analysis['length_category'] = pd.cut(
                df_analysis['review_length'], 
                bins=REVIEW_LENGTH_BINS,
                labels=REVIEW_LENGTH_LABELS
            )
            length_unit = "characters"
        
        # Calculate average rating by length category
        length_rating = df_analysis.groupby('length_category', observed=False)['rating'].mean()
        
        fig_length = px.bar(
            x=length_rating.index,
            y=length_rating.values,
            title="Average Rating by Review Length",
            color=length_rating.values,
            color_continuous_scale='viridis',
            labels={'x': 'Review Length Category', 'y': 'Average Rating'},
            text=length_rating.values
        )
        fig_length.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
        fig_length.update_xaxes(title="Review Length Category")
        fig_length.update_yaxes(title="Average Rating (Stars)", range=[0, 5])
        fig_length.update_layout(showlegend=False)
        st.plotly_chart(fig_length, config={'displayModeBar': False}, use_container_width=True)
        
        # Review length insights
        avg_length = df_analysis['review_length'].mean()
        median_length = df_analysis['review_length'].median()
        st.info(f"📊 **Length Stats**\nAverage: {avg_length:.0f} {length_unit}\nMedian: {median_length:.0f} {length_unit}")
    
    with col2:
        st.subheader("💰 Price Analysis")
        
        if 'price' in df_filtered.columns and df_filtered['price'].notna().sum() > 0:
            # Create price analysis
            price_data = df_filtered[df_filtered['price'].notna()].copy()
            
            if len(price_data) > 10:
                # Extract numeric price (handle various formats)
                try:
                    price_data['price_numeric'] = (
                        price_data['price']
                        .astype(str)
                        .str.replace('$', '', regex=False)
                        .str.replace(',', '', regex=False)
                        .str.replace('USD', '', regex=False)
                        .str.strip()
                        .astype(float)
                    )
                    
                    # Create price ranges
                    price_data['price_range'] = pd.cut(
                        price_data['price_numeric'],
                        bins=PRICE_RANGES,
                        labels=PRICE_LABELS
                    )
                    
                    # Calculate average rating by price range
                    price_rating = price_data.groupby('price_range', observed=False)['rating'].mean()
                    
                    fig_price = px.bar(
                        x=price_rating.index,
                        y=price_rating.values,
                        title="Average Rating by Price Range",
                        color=price_rating.values,
                        color_continuous_scale='Oranges',
                        labels={'x': 'Price Range', 'y': 'Average Rating'},
                        text=price_rating.values
                    )
                    fig_price.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
                    fig_price.update_xaxes(title="Price Range")
                    fig_price.update_yaxes(title="Average Rating (Stars)", range=[0, 5])
                    fig_price.update_layout(showlegend=False)
                    st.plotly_chart(fig_price, config={'displayModeBar': False}, use_container_width=True)
                    
                    # Price insights in info card (matching Review Length format)
                    avg_price = price_data['price_numeric'].mean()
                    median_price = price_data['price_numeric'].median()
                    
                    # Format large numbers properly without markdown issues
                    def format_price(price):
                        if price >= 1_000_000:
                            return f"${price/1_000_000:.1f}M"
                        elif price >= 1_000:
                            return f"${price/1_000:.1f}K"
                        else:
                            return f"${price:.2f}"
                    
                    avg_formatted = format_price(avg_price)
                    median_formatted = format_price(median_price)
                    
                    # Use info card like Review Length section
                    st.info(f"💰 **Price Stats**\nAverage: `{avg_formatted}`\nMedian: `{median_formatted}`")
                    
                except (ValueError, TypeError) as e:
                    st.warning("⚠️ Price data format issues detected")
                    st.info("Price analysis unavailable due to data formatting")
            else:
                st.info("Insufficient price data for analysis (need >10 records)")
        else:
            st.info("Price data not available in this sample")
    
    # Sentiment deep dive section
    st.subheader("😊 Sentiment Deep Dive")
    
    sentiment_col1, sentiment_col2 = st.columns(2)
    
    with sentiment_col1:
        st.subheader("📊 Sentiment by Category")
        
        # Calculate sentiment distribution by category
        sentiment_category = pd.crosstab(
            df_filtered['main_category'], 
            df_filtered['sentiment'], 
            normalize='index'
        ) * 100
        
        # Show top 8 categories by volume
        top_categories_sentiment = df_filtered['main_category'].value_counts().head(8).index
        sentiment_category_top = sentiment_category.loc[top_categories_sentiment]
        
        fig_sent_cat = px.bar(
            sentiment_category_top,
            title="Sentiment Distribution by Category (%)",
            color_discrete_map=SENTIMENT_COLORS,
            labels={'index': 'Category', 'value': 'Percentage (%)', 'variable': 'Sentiment'},
            barmode='stack'
        )
        fig_sent_cat.update_xaxes(title="Product Category", tickangle=45)
        fig_sent_cat.update_yaxes(title="Percentage (%)")
        fig_sent_cat.update_layout(height=500)
        st.plotly_chart(fig_sent_cat, config={'displayModeBar': False}, use_container_width=True)
    
    with sentiment_col2:
        st.subheader("📈 Sentiment Over Time")
        
        # Calculate sentiment trends over time
        sentiment_time = df_filtered.groupby([df_filtered['year'], 'sentiment']).size().unstack(fill_value=0)
        sentiment_time_pct = sentiment_time.div(sentiment_time.sum(axis=1), axis=0) * 100
        
        fig_sent_time = px.line(
            sentiment_time_pct,
            title="Sentiment Trends Over Time (%)",
            color_discrete_map=SENTIMENT_COLORS,
            labels={'index': 'Year', 'value': 'Percentage (%)', 'variable': 'Sentiment'},
            markers=True
        )
        fig_sent_time.update_traces(line=dict(width=3), marker=dict(size=8))
        fig_sent_time.update_xaxes(title="Year")
        fig_sent_time.update_yaxes(title="Percentage (%)", range=[0, 100])
        fig_sent_time.update_layout(height=500)
        st.plotly_chart(fig_sent_time, config={'displayModeBar': False}, use_container_width=True)
    
    # Advanced insights section
    st.subheader("🔍 Advanced Insights")
    
    advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
    
    with advanced_col1:
        # Correlation insight
        rating_sentiment_corr = df_filtered.groupby('sentiment')['rating'].mean()
        best_sentiment = rating_sentiment_corr.idxmax()
        best_sentiment_rating = rating_sentiment_corr.max()
        st.success(f"📈 **Rating-Sentiment**\n{best_sentiment} reviews\nhave {best_sentiment_rating:.2f}⭐ avg rating")
    
    with advanced_col2:
        # Review depth insight
        if len(df_analysis) > 0:
            detailed_reviews = df_analysis[df_analysis['length_category'].isin(['Long (300-1000)', 'Very Long (>1000)'])]
            detailed_pct = len(detailed_reviews) / len(df_analysis) * 100
            detailed_avg_rating = detailed_reviews['rating'].mean() if len(detailed_reviews) > 0 else 0
            st.info(f"📝 **Detailed Reviews**\n{detailed_pct:.1f}% are long\nAvg rating: {detailed_avg_rating:.2f}⭐")
    
    with advanced_col3:
        # Sentiment stability
        if len(sentiment_time_pct) > 1:
            positive_stability = sentiment_time_pct['Positive'].std() if 'Positive' in sentiment_time_pct.columns else 0
            stability_level = "Stable" if positive_stability < 5 else "Variable"
            st.warning(f"📊 **Sentiment Stability**\n{stability_level} over time\nσ = {positive_stability:.1f}%")