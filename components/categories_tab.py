# [file name]: components/categories_tab.py
"""
Categories & Ratings Tab - Category analysis and rating comparisons
"""
import streamlit as st
import plotly.express as px

def render_categories_tab(df_filtered):
    """Render categories and ratings analysis tab"""
    
    # Top categories section
    st.subheader("📊 Category Performance")
    
    # Calculate category statistics
    category_stats = df_filtered.groupby('main_category').agg({
        'rating': ['count', 'mean']
    }).round(2)
    category_stats.columns = ['Review Count', 'Avg Rating']
    category_stats = category_stats.sort_values('Review Count', ascending=False).head(10)
    
    # Create category performance chart
    fig_categories = px.bar(
        category_stats.reset_index(),
        x='main_category',
        y='Review Count',
        title="Top 10 Categories by Review Count",
        color='Avg Rating',
        color_continuous_scale='RdYlGn',
        text='Avg Rating',
        labels={'main_category': 'Category', 'Review Count': 'Number of Reviews'},
        hover_data={'Avg Rating': ':.2f'}
    )
    fig_categories.update_traces(texttemplate='%{text:.1f}⭐', textposition='outside')
    fig_categories.update_xaxes(tickangle=45, title="Product Category")
    fig_categories.update_yaxes(title="Number of Reviews")
    fig_categories.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_categories, config={'displayModeBar': False}, use_container_width=True)
    
    # Category details section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Category Ratings Comparison")
        
        # Get top categories by rating
        category_avg_ratings = df_filtered.groupby('main_category')['rating'].mean().sort_values(ascending=False).head(10)
        
        # Create horizontal bar chart
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
            labels={'Avg_Rating': 'Average Rating (Stars)', 'Category': 'Product Category'},
            text='Avg_Rating'
        )
        
        # Format text display
        fig_cat_ratings.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
        
        # Force descending order: highest ratings at TOP of chart
        fig_cat_ratings.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_colorbar=dict(title="Rating"),
            showlegend=False
        )
        fig_cat_ratings.update_xaxes(title="Average Rating (Stars)", range=[0, 5])
        fig_cat_ratings.update_yaxes(title="Product Category")
        st.plotly_chart(fig_cat_ratings, config={'displayModeBar': False}, use_container_width=True)
    
    with col2:
        st.subheader("📈 Rating Trends by Categories")
        
        # Get top categories by volume for selection
        top_categories = df_filtered['main_category'].value_counts().head(10).index.tolist()
        
        # Category filter for trends chart
        selected_trend_categories = st.multiselect(
            "Select categories to display in trends:",
            options=top_categories,
            default=top_categories[:3],
            help="Select 1-3 categories for clearest visualization"
        )
        
        if selected_trend_categories:
            # Filter data for selected categories
            df_trend_cats = df_filtered[df_filtered['main_category'].isin(selected_trend_categories)]
            
            if len(df_trend_cats) > 0:
                # Calculate monthly trends
                monthly_cat_rating = df_trend_cats.groupby([
                    df_trend_cats['review_date'].dt.to_period('M'), 
                    'main_category'
                ])['rating'].mean().reset_index()
                monthly_cat_rating['month'] = monthly_cat_rating['review_date'].dt.to_timestamp()
                
                # Create trends chart
                fig_trends = px.line(
                    monthly_cat_rating,
                    x='month',
                    y='rating',
                    color='main_category',
                    title=f"Rating Trends ({len(selected_trend_categories)} Selected Categories)",
                    labels={'month': 'Month', 'rating': 'Average Rating', 'main_category': 'Category'},
                    markers=True
                )
                fig_trends.update_traces(line=dict(width=3), marker=dict(size=6))
                fig_trends.update_xaxes(title="Time (Months)")
                fig_trends.update_yaxes(title="Average Rating (Stars)", range=[1, 5])
                fig_trends.update_layout(hovermode='x unified')
                st.plotly_chart(fig_trends, config={'displayModeBar': False}, use_container_width=True)
            else:
                st.info("No data available for selected categories")
        else:
            st.info("👆 Select categories above to view rating trends")
    
    # Category insights section
    st.subheader("🔍 Category Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        # Highest rated category
        best_category = category_avg_ratings.index[0]
        best_rating = category_avg_ratings.iloc[0]
        best_count = df_filtered[df_filtered['main_category'] == best_category].shape[0]
        st.success(f"🏅 **Highest Rated**\n{best_category}\n{best_rating:.2f}⭐ ({best_count:,} reviews)")
    
    with insight_col2:
        # Most reviewed category
        most_reviewed = category_stats.index[0]
        review_count = int(category_stats.loc[most_reviewed, 'Review Count'])
        avg_rating_most = category_stats.loc[most_reviewed, 'Avg Rating']
        st.info(f"📊 **Most Reviewed**\n{most_reviewed}\n{review_count:,} reviews ({avg_rating_most:.2f}⭐)")
    
    with insight_col3:
        # Category diversity
        total_categories = df_filtered['main_category'].nunique()
        total_reviews = len(df_filtered)
        avg_reviews_per_category = total_reviews / total_categories
        st.warning(f"🏷️ **Category Spread**\n{total_categories} categories\nAvg {avg_reviews_per_category:.0f} reviews/category")