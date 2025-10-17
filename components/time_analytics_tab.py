# [file name]: components/time_analytics_tab.py
"""
Time Analytics Tab - Temporal analysis and trends
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def render_time_analytics_tab(df_filtered):
    """Render time analytics tab with temporal trends and analysis"""
    
    # Monthly review trends
    st.subheader("📈 Review Volume Trends")
    
    # Group by month and calculate metrics
    monthly_reviews = df_filtered.groupby([
        df_filtered['review_date'].dt.to_period('M')
    ])['rating'].agg(['count', 'mean']).reset_index()
    monthly_reviews['month'] = monthly_reviews['review_date'].dt.to_timestamp()
    
    # Create timeline chart
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=monthly_reviews['month'],
        y=monthly_reviews['count'],
        mode='lines+markers',
        name='Review Count',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<b>%{x}</b><br>Reviews: %{y}<extra></extra>'
    ))
    
    fig_timeline.update_layout(
        title="Monthly Review Volume Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Reviews",
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig_timeline, config={'displayModeBar': False})
    
    # Yearly analysis section
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
            color_continuous_scale='Blues',
            text=yearly_counts.values
        )
        fig_yearly.update_traces(texttemplate='%{text}', textposition='outside')
        fig_yearly.update_xaxes(title="Year")
        fig_yearly.update_yaxes(title="Number of Reviews")
        fig_yearly.update_layout(showlegend=False)
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
        fig_rating_year.update_traces(
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Avg Rating: %{y:.2f}<extra></extra>'
        )
        fig_rating_year.update_xaxes(title="Year")
        fig_rating_year.update_yaxes(title="Average Rating (1-5 Stars)", range=[1, 5])
        st.plotly_chart(fig_rating_year, config={'displayModeBar': False})
    
    # Additional temporal insights
    st.subheader("📅 Temporal Insights")
    
    # Create insights columns
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        # Peak review month
        peak_month_data = monthly_reviews.loc[monthly_reviews['count'].idxmax()]
        peak_month = peak_month_data['month'].strftime('%B %Y')
        peak_count = int(peak_month_data['count'])
        st.info(f"📈 **Peak Month**\n{peak_month}\n{peak_count:,} reviews")
    
    with insight_col2:
        # Rating trend
        first_year_rating = yearly_ratings.iloc[0]
        last_year_rating = yearly_ratings.iloc[-1]
        trend_direction = "↗️" if last_year_rating > first_year_rating else "↘️"
        trend_change = abs(last_year_rating - first_year_rating)
        st.info(f"📊 **Rating Trend**\n{trend_direction} {trend_change:.2f} points\n{first_year_rating:.2f} → {last_year_rating:.2f}")
    
    with insight_col3:
        # Review growth analysis
        if len(yearly_counts) > 2:
            last_year_count = yearly_counts.iloc[-1]
            second_last_count = yearly_counts.iloc[-2]
            third_last_count = yearly_counts.iloc[-3]
            
            # If last year is significantly lower than previous years, likely incomplete
            avg_previous_two = (second_last_count + third_last_count) / 2
            is_likely_incomplete = last_year_count < (avg_previous_two * 0.6)
            
            if is_likely_incomplete:
                growth_between_complete = second_last_count - third_last_count
                growth_direction = "📈" if growth_between_complete > 0 else "📉"
                last_complete_year = yearly_counts.index[-2]
                st.warning(f"📊 **Growth Trend**\n{growth_direction} {abs(growth_between_complete):,} reviews\n{yearly_counts.index[-3]} → {last_complete_year}\n⚠️ {yearly_counts.index[-1]} data incomplete")
            else:
                recent_growth = last_year_count - second_last_count
                growth_direction = "📈" if recent_growth > 0 else "📉"
                st.info(f"📆 **Recent Growth**\n{growth_direction} {abs(recent_growth):,} reviews\nYear-over-year change")
        elif len(yearly_counts) > 1:
            recent_growth = yearly_counts.iloc[-1] - yearly_counts.iloc[-2]
            growth_direction = "📈" if recent_growth > 0 else "📉"
            st.info(f"📆 **Recent Growth**\n{growth_direction} {abs(recent_growth):,} reviews\nYear-over-year change")
        else:
            st.info("📆 **Data Span**\nSingle year data\nNo growth comparison")