# [file name]: components/interactive_analyst_tab.py
"""
Interactive Data Analyst - Using LangChain Agent with proper error handling
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

def get_pandas_agent(df):
    """Create LangChain pandas agent with proper security settings"""
    
    # Check for OpenAI API key first
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('test'):
        st.warning("⚠️ OpenAI API key not found or invalid. Agent analysis will use fallback mode.")
        return None
    
    try:
        from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
        from langchain_openai import ChatOpenAI
        
        # Create the LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key,
            max_tokens=1000  # Limit response size
        )
        
        # Create agent with simplified configuration that works with current version
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,  # Required for pandas operations
            agent_type="openai-tools",  # Use the newer agent type
            number_of_head_rows=5,
            max_iterations=3,  # Prevent infinite loops
            early_stopping_method="generate"
        )
        
        return agent
        
    except Exception as e:
        st.error(f"Agent creation failed: {str(e)}")
        return None

def analyze_with_agent_safely(agent, query, df):
    """Use agent to analyze data with robust error handling"""
    try:
        # Enhanced prompt to prevent code generation issues
        dataset_context = f"""
        You are an experienced data analyst and analyzing an Amazon product reviews dataset with the following structure:
        - {len(df)} total reviews
        - Columns available: {', '.join(df.columns.tolist())}
        - Date range: {df['review_date'].min().strftime('%Y-%m-%d')} to {df['review_date'].max().strftime('%Y-%m-%d')}
        - {df['main_category'].nunique()} product categories
        - Average rating: {df['rating'].mean():.2f}/5
        
        Important columns:
        - rating: Numerical rating from 1-5 stars
        - sentiment: Categorical (Positive, Neutral, Negative)  
        - main_category: Product category
        - review_date: Date of review
        - price: Product price (may have missing values)
        
        IMPORTANT INSTRUCTIONS:
        1. Generate data driven results
        2. Start your answer with visualizations or tables
        3. Provide analysis in clear text with statistics
        4. If you need to analyze seasonal patterns, describe the patterns you would look for
        5. Focus on insights and business implications for the Amazon
        6. Focus on getting the result instead of discussing how to get it done
        
        Please provide a clear, data-driven answer to the user's question without generating problematic code.
        """
        
        enhanced_query = f"""
        
        User Question: {query}
        
        Answer: 
        """
             # After getting the response, check if it's theoretical
        response_text = response["output"] if isinstance(response, dict) and "output" in response else str(response)
        
        # Check if response is theoretical (mentions "we would need to" or similar)
        theoretical_indicators = [
            'we would need to', 'we can approach', 'calculate the', 'group by',
            'identify patterns', 'look for', 'analyze the data', 'steps to'
        ]
        
        if any(indicator in response_text.lower() for indicator in theoretical_indicators):
            st.info("🔍 Agent provided theoretical framework - generating actual data analysis...")
            if any(word in query.lower() for word in ['seasonal', 'month', 'pattern', 'trend']):
                return enhanced_seasonal_fallback(query, df), None, "enhanced_fallback"
        
        return response_text, None, "agent_analysis"
        
    except Exception as e:
        
        # Try invoking the agent with proper error handling
        try:
            with st.spinner("🤖 Agent is analyzing the data..."):
                response = agent.invoke({"input": enhanced_query})
                
                # Handle different response formats
                if isinstance(response, dict):
                    if "output" in response:
                        return response["output"], None, "agent_analysis"
                    else:
                        return str(response), None, "agent_analysis"
                else:
                    return str(response), None, "agent_analysis"
                    
        except Exception as invoke_error:
            error_msg = str(invoke_error)
            if "max iterations" in error_msg.lower() or "stopped" in error_msg.lower():
                return "Agent analysis was stopped to prevent infinite loops. The question may be too complex for automated analysis. Try rephrasing or using the visualization tools below.", None, "agent_limited"
            else:
                raise invoke_error
        
    except Exception as e:
        error_msg = f"Agent analysis failed: {str(e)}"
        st.warning(f"⚠️ {error_msg}")
        
        # Enhanced fallback for seasonal analysis
        if any(word in query.lower() for word in ['seasonal', 'month', 'pattern', 'trend']):
            return enhanced_seasonal_fallback(query, df), None, "fallback_analysis"
        
        # Fallback to safe analysis for common queries
        fallback_response = fallback_analysis(query, df)
        return fallback_response, None, "fallback_analysis"

def enhanced_seasonal_fallback(query, df):
    """Enhanced fallback specifically for seasonal pattern questions"""
    try:
        if 'review_date' not in df.columns:
            return "Seasonal analysis requires review date data, which is not available."
        
        # Convert to datetime
        df_clean = df.copy()
        df_clean['review_date'] = pd.to_datetime(df_clean['review_date'], errors='coerce')
        df_clean = df_clean[df_clean['review_date'].notna()]
        
        if len(df_clean) == 0:
            return "No valid date data available for seasonal analysis."
        
        # Extract month and year
        df_clean['month'] = df_clean['review_date'].dt.month
        df_clean['year'] = df_clean['review_date'].dt.year
        df_clean['month_name'] = df_clean['review_date'].dt.month_name()
        
        # Monthly review volume
        monthly_volume = df_clean.groupby('month').size()
        monthly_volume_named = df_clean.groupby('month_name').size()
        
        # Monthly sentiment
        monthly_sentiment = df_clean.groupby(['month_name', 'sentiment']).size().unstack(fill_value=0)
        monthly_sentiment_pct = monthly_sentiment.div(monthly_sentiment.sum(axis=1), axis=0) * 100
        
        # Find peak months
        peak_month = monthly_volume_named.idxmax()
        peak_volume = monthly_volume_named.max()
        low_month = monthly_volume_named.idxmin()
        low_volume = monthly_volume_named.min()
        
        # Analyze sentiment patterns
        response = f"""
**Seasonal Pattern Analysis (Enhanced Fallback):**

📊 **Review Volume Patterns:**
- Peak review month: **{peak_month}** ({peak_volume:,} reviews)
- Lowest review month: **{low_month}** ({low_volume:,} reviews)
- Seasonal variation: {((peak_volume - low_volume) / low_volume * 100):.1f}% difference

🎭 **Sentiment Patterns by Month:**
"""
        
        # Add sentiment insights for each month
        for month in monthly_sentiment_pct.index:
            if 'Positive' in monthly_sentiment_pct.columns:
                pos_pct = monthly_sentiment_pct.loc[month, 'Positive']
                response += f"- {month}: {pos_pct:.1f}% positive\n"
        
        # Overall insights
        if 'Positive' in monthly_sentiment_pct.columns:
            best_month = monthly_sentiment_pct['Positive'].idxmax()
            best_sentiment = monthly_sentiment_pct['Positive'].max()
            worst_month = monthly_sentiment_pct['Positive'].idxmin()
            worst_sentiment = monthly_sentiment_pct['Positive'].min()
            
            response += f"""
💡 **Key Insights:**
- Most positive reviews: **{best_month}** ({best_sentiment:.1f}% positive)
- Least positive reviews: **{worst_month}** ({worst_sentiment:.1f}% positive)
- Sentiment variation: {best_sentiment - worst_sentiment:.1f}% difference

📈 **Business Implications:**
- Consider increasing support during {worst_month} when sentiment is lowest
- Leverage high-engagement months ({peak_month}) for marketing campaigns
- Monitor {low_month} for potential seasonal business challenges
"""
        
        return response
        
    except Exception as e:
        return f"Seasonal analysis failed: {str(e)}"

def fallback_analysis(query, df):
    """Fallback analysis when agent fails"""
    query_lower = query.lower()
    
    try:
        if any(word in query_lower for word in ['sentiment', 'positive', 'negative']) and any(word in query_lower for word in ['price', 'cost']):
            # Sentiment by price range analysis
            if 'price' in df.columns and df['price'].notna().sum() > 0:
                price_data = df[df['price'].notna()].copy()
                try:
                    price_data['price_numeric'] = (
                        price_data['price']
                        .astype(str)
                        .str.replace('$', '', regex=False)
                        .str.replace(',', '', regex=False)
                        .str.strip()
                        .astype(float)
                    )
                    
                    # Create price ranges
                    price_ranges = [0, 50, 100, 200, 500, float('inf')]
                    price_labels = ['<$50', '$50-100', '$100-200', '$200-500', '>$500']
                    
                    price_data['price_range'] = pd.cut(
                        price_data['price_numeric'],
                        bins=price_ranges,
                        labels=price_labels
                    )
                    
                    sentiment_by_price = pd.crosstab(
                        price_data['price_range'], 
                        price_data['sentiment'], 
                        normalize='index'
                    ) * 100
                    
                    response = "**Sentiment by Price Range (Fallback Analysis):**\n\n"
                    for price_range in price_labels:
                        if price_range in sentiment_by_price.index:
                            row = sentiment_by_price.loc[price_range]
                            response += f"**{price_range}:**\n"
                            for sentiment in ['Positive', 'Neutral', 'Negative']:
                                if sentiment in row:
                                    response += f"  - {sentiment}: {row[sentiment]:.1f}%\n"
                            response += "\n"
                    
                    return response
                    
                except Exception as price_error:
                    return f"Price analysis failed: {str(price_error)}"
            else:
                return "Price data not available for analysis"
            
        elif 'distribution' in query_lower and 'rating' in query_lower:
            result = df['rating'].value_counts().sort_index()
            response = "**Rating Distribution (Fallback Analysis):**\n\n"
            total = len(df)
            for rating, count in result.items():
                percentage = (count / total) * 100
                response += f"- {rating}⭐: {count:,} reviews ({percentage:.1f}%)\n"
            return response
            
        else:
            return f"""
**Dataset Overview (Fallback Analysis):**
- Total reviews: {len(df):,}
- Categories: {df['main_category'].nunique()}
- Date range: {df['review_date'].min().strftime('%Y-%m-%d')} to {df['review_date'].max().strftime('%Y-%m-%d')}
- Average rating: {df['rating'].mean():.2f}⭐
- Positive sentiment: {(df['sentiment'] == 'Positive').mean()*100:.1f}%

*Note: Advanced agent analysis is temporarily unavailable. Basic statistics shown above.*
"""
            
    except Exception as e:
        return f"Fallback analysis also failed: {str(e)}"

def create_rating_trend_visualization(df, response_text):
    """Plot average rating over time (year) for top categories and show summary table and insights"""
    if 'main_category' not in df.columns or 'rating' not in df.columns or 'review_date' not in df.columns:
        st.warning("Required columns not found for rating trend analysis.")
        return
    try:
        df_viz = df.copy()
        df_viz['review_date'] = pd.to_datetime(df_viz['review_date'], errors='coerce')
        df_viz = df_viz[df_viz['review_date'].notna()]
        df_viz['year'] = df_viz['review_date'].dt.year
        # Top categories by review count
        top_categories = df_viz['main_category'].value_counts().head(5).index
        df_viz = df_viz[df_viz['main_category'].isin(top_categories)]
        # Group by year and category
        rating_trends = df_viz.groupby(['year', 'main_category'])['rating'].mean().unstack()
        st.subheader("📈 Average Rating Over Time by Category")
        fig = px.line(rating_trends, x=rating_trends.index, y=rating_trends.columns,
                     labels={'value': 'Average Rating', 'year': 'Year'},
                     title='Average Rating Trends by Category (Yearly)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### 📋 Rating Trend Table")
        st.dataframe(rating_trends.round(2))
        # --- Insights ---
        st.markdown("#### 🧠 Analyst Insights: Rating Trends")
        # Find categories with biggest increase/decrease
        trend_summary = {}
        for cat in rating_trends.columns:
            series = rating_trends[cat].dropna()
            if len(series) > 1:
                change = series.iloc[-1] - series.iloc[0]
                trend_summary[cat] = change
        if trend_summary:
            best_up = max(trend_summary, key=lambda k: trend_summary[k])
            best_down = min(trend_summary, key=lambda k: trend_summary[k])
            st.info(f"Biggest improvement: **{best_up}** (+{trend_summary[best_up]:.2f}⭐)\nBiggest decline: **{best_down}** ({trend_summary[best_down]:.2f}⭐)")
        else:
            st.info("No significant rating changes detected across years.")
    except Exception as e:
        st.error(f"Rating trend visualization failed: {str(e)}")

def create_seasonal_visualization(df, response_text):
    """Create seasonal/monthly analysis visualizations"""
    if 'review_date' not in df.columns:
        st.warning("Review date column not found for seasonal analysis")
        return
        
    try:
        # Convert to datetime if not already
        df_viz = df.copy()
        df_viz['review_date'] = pd.to_datetime(df_viz['review_date'], errors='coerce')
        df_viz['month'] = df_viz['review_date'].dt.month
        df_viz['month_name'] = df_viz['review_date'].dt.month_name()
        
        # Monthly review volume
        monthly_volume = df_viz['month'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Monthly Review Volume")
            fig_volume = px.bar(
                x=monthly_volume.index,
                y=monthly_volume.values,
                labels={'x': 'Month', 'y': 'Number of Reviews'},
                title='Review Volume by Month'
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            st.subheader("💭 Monthly Sentiment Distribution")
            if 'sentiment' in df_viz.columns:
                sentiment_monthly = df_viz.groupby(['month_name', 'sentiment']).size().unstack(fill_value=0)
                fig_sentiment = px.bar(
                    sentiment_monthly,
                    title='Sentiment Distribution by Month',
                    labels={'value': 'Number of Reviews', 'index': 'Month'}
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
    except Exception as e:
        st.error(f"Seasonal visualization failed: {str(e)}")

def create_sentiment_visualization(df, response_text):
    """Create sentiment analysis visualizations"""
    if 'sentiment' not in df.columns:
        st.warning("Sentiment column not found")
        return
        
    try:
        sentiment_counts = df['sentiment'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎭 Sentiment Distribution")
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Overall Sentiment Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 Sentiment by Rating")
            if 'rating' in df.columns:
                sentiment_rating = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
                fig_rating = px.bar(
                    sentiment_rating,
                    title='Sentiment Distribution by Rating',
                    labels={'value': 'Number of Reviews', 'index': 'Rating'}
                )
                st.plotly_chart(fig_rating, use_container_width=True)
                
    except Exception as e:
        st.error(f"Sentiment visualization failed: {str(e)}")

def create_category_visualization(df, response_text):
    """Create category analysis visualizations"""
    if 'main_category' not in df.columns:
        st.warning("Category column not found")
        return
        
    try:
        # Top categories by volume
        category_counts = df['main_category'].value_counts().head(10)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏷️ Top Categories by Volume")
            fig_cat = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title='Review Count by Category (Top 10)',
                labels={'x': 'Number of Reviews', 'y': 'Category'}
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col2:
            st.subheader("⭐ Average Rating by Category")
            if 'rating' in df.columns:
                avg_rating_by_cat = df.groupby('main_category')['rating'].mean().sort_values(ascending=False).head(10)
                fig_rating = px.bar(
                    x=avg_rating_by_cat.values,
                    y=avg_rating_by_cat.index,
                    orientation='h',
                    title='Average Rating by Category (Top 10)',
                    labels={'x': 'Average Rating', 'y': 'Category'}
                )
                st.plotly_chart(fig_rating, use_container_width=True)

                # --- Actionable Insights ---
                st.markdown("#### 📊 Analyst Insights: Average Rating Trends by Category")
                # Find highest and lowest rated categories
                highest_cat = avg_rating_by_cat.idxmax()
                highest_rating = avg_rating_by_cat.max()
                lowest_cat = avg_rating_by_cat.idxmin()
                lowest_rating = avg_rating_by_cat.min()
                st.info(f"Highest average rating: **{highest_cat}** ({highest_rating:.2f}⭐)\nLowest average rating: **{lowest_cat}** ({lowest_rating:.2f}⭐)")

                # Detect outliers or notable changes
                rating_diff = highest_rating - lowest_rating
                if rating_diff > 1.0:
                    st.warning(f"Significant gap detected: {highest_cat} is rated over 1 star higher than {lowest_cat}. Consider investigating factors driving this difference.")

                # Business implication suggestion
                st.markdown("- Categories with consistently high ratings may indicate strong product-market fit and customer satisfaction.\n- Categories with lower ratings may benefit from targeted improvements, better support, or product innovation.")

    except Exception as e:
        st.error(f"Category visualization failed: {str(e)}")

def create_rating_visualization(df, response_text):
    """Create rating analysis visualizations"""
    if 'rating' not in df.columns:
        st.warning("Rating column not found")
        return
        
    try:
        rating_dist = df['rating'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⭐ Rating Distribution")
            fig_dist = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title='Distribution of Ratings',
                labels={'x': 'Rating (Stars)', 'y': 'Number of Reviews'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.subheader("📈 Rating Statistics")
            avg_rating = df['rating'].mean()
            median_rating = df['rating'].median()
            
            fig_stats = go.Figure()
            fig_stats.add_trace(go.Indicator(
                mode = "gauge+number",
                value = avg_rating,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Rating"},
                gauge = {
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2.5], 'color': "lightgray"},
                        {'range': [2.5, 4], 'color': "yellow"},
                        {'range': [4, 5], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 4.5}}
            ))
            st.plotly_chart(fig_stats, use_container_width=True)
            
    except Exception as e:
        st.error(f"Rating visualization failed: {str(e)}")

def create_agent_visualization(response_text, query):
    """Create visualizations based on agent analysis results"""
    try:
        query_lower = query.lower()
        
        # Check if this is a query that can benefit from visualization
        viz_keywords = [
            'seasonal', 'monthly', 'trend', 'pattern', 'distribution', 
            'volume', 'sentiment', 'category', 'rating', 'time', 
            'correlation', 'comparison', 'over time'
        ]
        
        if not any(keyword in query_lower for keyword in viz_keywords):
            return
            
        st.markdown("### 📊 **Agent-Generated Insights Visualization**")
        
        # Get the current dataframe from session state
        if 'df_filtered' not in st.session_state:
            st.warning("No data available for visualization")
            return
            
        df = st.session_state.df_filtered
        
        # Create visualizations based on query type
        if any(word in query_lower for word in ['seasonal', 'monthly', 'month', 'pattern']):
            create_seasonal_visualization(df, response_text)
        elif any(word in query_lower for word in ['average rating over time', 'rating trend', 'rating trends', 'rating change', 'rating by year', 'rating by month', 'rating over years', 'rating over time']):
            create_rating_trend_visualization(df, response_text)
        elif any(word in query_lower for word in ['sentiment', 'positive', 'negative']):
            create_sentiment_visualization(df, response_text)
        elif any(word in query_lower for word in ['category', 'categories']):
            create_category_visualization(df, response_text)
        elif any(word in query_lower for word in ['rating', 'ratings', 'score']):
            create_rating_visualization(df, response_text)
        else:
            # Generic visualization suggestion
            st.info("💡 **Visualization Suggestion:** Try asking for specific visualizations like:\n"
                   "- 'Show me a chart of sentiment by month'\n"
                   "- 'Visualize rating distribution by category'\n"
                   "- 'Create a plot of review volume over time'")
    except Exception as e:
        st.error(f"Visualization creation failed: {str(e)}")

def render_interactive_analyst_tab(df_filtered):
    """Render interactive data analyst tab using AI agent"""
    
    # Store dataframe in session state for visualization functions
    st.session_state.df_filtered = df_filtered
    
    st.header("🔍 AI Data Analyst Agent")
    st.markdown("Ask complex analytical questions and let the AI agent analyze your dataset automatically!")
    
    # Security notice
    with st.expander("🔒 Security Notice", expanded=False):
        st.info("""
        **AI Agent Security Information:**
        - This agent uses LangChain's pandas dataframe agent
        - It can execute Python code for data analysis
        - Code execution is sandboxed and limited
        - Only basic pandas operations are allowed
        - No external system calls or file operations
        """)
    
    # Initialize agent
    if "data_agent" not in st.session_state:
        with st.spinner("🤖 Initializing AI data analyst agent..."):
            agent = get_pandas_agent(df_filtered)
            if agent:
                st.session_state.data_agent = agent
                st.success("✅ AI agent initialized successfully!")
            else:
                st.error("❌ Failed to initialize agent. Using fallback analysis.")
                st.session_state.data_agent = None
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Reviews", f"{len(df_filtered):,}")
    with col2:
        st.metric("🏷️ Categories", df_filtered['main_category'].nunique())
    with col3:
        st.metric("📅 Date Range", f"{df_filtered['year'].min()}-{df_filtered['year'].max()}")
    
    # Example questions for agent
    with st.expander("💡 Try These Complex Analysis Questions"):
        st.markdown("""
        **Advanced Agent Questions:**
        - "How does customer sentiment vary across different price ranges?"
        - "What's the correlation between review length and rating?"
        - "Which product categories have the most inconsistent ratings?"
        - "How has the average rating changed over time for different categories?"
        - "What factors are most correlated with positive reviews?"
        - "Analyze seasonal patterns in review volume and sentiment"
        """)
    
    # Query input
    query = st.text_area(
        "Ask a complex analytical question:",
        height=100,
        value=st.session_state.get('analyst_query', ''),
        placeholder="e.g., 'How does sentiment vary by price range?' or 'Analyze rating trends over time by category'"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🤖 Ask AI Agent", type="primary") and query.strip():
            with st.spinner("🔄 AI Agent is analyzing your data..."):
                if st.session_state.get('data_agent'):
                    # Use agent for analysis
                    response, viz_data, analysis_type = analyze_with_agent_safely(
                        st.session_state.data_agent, query, df_filtered
                    )
                else:
                    # Fallback to safe analysis
                    response = fallback_analysis(query, df_filtered)
                    viz_data = None
                    analysis_type = "fallback"
                
                # Store in history
                if "analyst_history" not in st.session_state:
                    st.session_state.analyst_history = []
                
                st.session_state.analyst_history.append({
                    "question": query,
                    "answer": response,
                    "analysis_type": analysis_type,
                    "timestamp": pd.Timestamp.now().strftime("%H:%M"),
                    "used_agent": bool(st.session_state.get('data_agent'))
                })
                
                # Clear input
                if 'analyst_query' in st.session_state:
                    del st.session_state.analyst_query
            
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Agent"):
            if "data_agent" in st.session_state:
                del st.session_state.data_agent
            if "analyst_history" in st.session_state:
                st.session_state.analyst_history = []
            st.success("Agent reset complete!")
            st.rerun()
    
    # Display current analysis
    if st.session_state.get('analyst_history'):
        latest = st.session_state.analyst_history[-1]
        
        st.markdown("---")
        st.subheader("📊 AI Agent Analysis Results")
        
        # Show agent status
        if latest.get('used_agent'):
            if "max iterations" in latest['answer'].lower() or "stopped" in latest['answer'].lower():
                st.warning("⚠️ **Analysis Method:** AI Agent (Limited - Complex Query)")
            else:
                st.success("🤖 **Analysis Method:** AI Agent (LangChain)")
        else:
            st.warning("⚠️ **Analysis Method:** Fallback (Agent Unavailable)")
        
        # Question
        st.markdown(f"**❓ Question:** {latest['question']}")
        
        # Answer
        st.markdown(f"**🤖 Analysis:** {latest['answer']}")
        
        # Visualization
        if latest.get('used_agent'):
            create_agent_visualization(latest['answer'], latest['question'])
    
    # Show history
    if len(st.session_state.get('analyst_history', [])) > 1:
        with st.expander("📝 Agent Analysis History"):
            for i, item in enumerate(reversed(st.session_state.analyst_history[:-1])):
                method = "🤖 Agent" if item.get('used_agent') else "⚠️ Fallback"
                st.markdown(f"**{i+1}. {item['question'][:50]}...** ({method})")
                st.caption(f"Analyzed at {item['timestamp']}")
                st.markdown("---")