# [file name]: components/overview_tab.py
"""
Overview Tab - Key metrics and overview charts
"""
import streamlit as st
import plotly.express as px
from config.settings import SENTIMENT_COLORS

def render_overview_tab(df_filtered):
    """Render the overview tab with key metrics and charts"""
    
    st.header("📊 Key Metrics Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Reviews",
            value=f"{len(df_filtered):,}",
            help="Number of reviews in filtered dataset"
        )
    
    with col2:
        avg_rating = df_filtered['rating'].mean()
        st.metric(
            label="⭐ Average Rating",
            value=f"{avg_rating:.2f}",
            help="Mean rating across all filtered reviews"
        )
    
    with col3:
        st.metric(
            label="🏷️ Categories",
            value=df_filtered['main_category'].nunique(),
            help="Number of unique product categories"
        )
    
    with col4:
        years_span = df_filtered['year'].max() - df_filtered['year'].min() + 1
        st.metric(
            label="📅 Time Span",
            value=f"{years_span} years",
            delta=f"{df_filtered['year'].min()}-{df_filtered['year'].max()}",
            help="Date range of reviews"
        )
    
    # Key insights section
    st.subheader("🎯 Key Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        top_category = df_filtered['main_category'].value_counts().index[0]
        top_category_count = df_filtered['main_category'].value_counts().iloc[0]
        st.info(f"🏆 **Top Category**\n{top_category}\n({top_category_count:,} reviews)")
        
    with insight_col2:
        positive_pct = (df_filtered['sentiment'] == 'Positive').sum() / len(df_filtered) * 100
        positive_count = (df_filtered['sentiment'] == 'Positive').sum()
        st.success(f"😊 **Positive Reviews**\n{positive_pct:.1f}%\n({positive_count:,} reviews)")
        
    with insight_col3:
        recent_reviews = df_filtered[df_filtered['year'] >= df_filtered['year'].max() - 1]
        recent_pct = len(recent_reviews) / len(df_filtered) * 100
        st.warning(f"🕐 **Recent Reviews**\nLast 2 Years: {recent_pct:.1f}%\n({len(recent_reviews):,} reviews)")

    # Overview charts section
    st.subheader("📈 Overview Charts")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Sentiment pie chart
        sentiment_counts = df_filtered['sentiment'].value_counts()
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map=SENTIMENT_COLORS
        )
        fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sentiment, config={'displayModeBar': False}, use_container_width=True)
    
    with chart_col2:
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
        fig_ratings.update_layout(showlegend=False)
        st.plotly_chart(fig_ratings, config={'displayModeBar': False}, use_container_width=True)