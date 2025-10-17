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
    
    # Enhanced prompt to prevent code generation issues
    dataset_context = f"""
    You are an experienced data analyst and analyzing an Amazon product reviews dataset with the following structure:
    - {len(df)} total reviews
    - Columns available: {', '.join(df.columns.tolist())}
    - Date range: {df['review_date'].min().strftime('%Y-%m-%d')} to {df['review_date'].max().strftime('%Y-%m-%d')}
    - {df['main_category'].nunique()} product categories
    - Average rating: {df['rating'].mean():.2f}/5
    
    **SAMPLE DATA (first 3 rows):**
    {df.head(3).to_string()}
    
    **DATA DICTIONARY:**
    - asin: Product ID
    - rating: Numerical rating from 1-5 stars
    - review_title: Title of the review
    - review_text: Full review content  
    - review_date: Date when review was posted
    - year: Year extracted from review_date
    - sentiment: Categorical (Positive, Neutral, Negative)  
    - product_title: Name of the product
    - main_category: Product category
    - price: Product price in USD format (may have missing values)
    
    **DATA QUALITY NOTES:**
    - Price column has missing values (NaN) - handle appropriately
    - Sentiment is derived from rating (4-5: Positive, 3: Neutral, 1-2: Negative)
    - Reviews span multiple years (20005-2023 based on sample)
    
    IMPORTANT INSTRUCTIONS:
    1. Generate data driven results using ACTUAL data analysis
    2. Start your answer with visualizations or tables when possible
    3. Provide analysis in clear text with statistics
    4. Focus on insights and business implications for Amazon
    5. If data is insufficient, suggest what additional data would be helpful
    6. Handle missing price values gracefully in analysis
    7. DO NOT generate theoretical approaches - provide actual findings
    
    **VISUALIZATION GUIDANCE:**
    Based on your analysis, suggest specific visualizations that would help illustrate your findings.
    Use these patterns:
    - For price analysis: "This could be visualized with a price distribution histogram"
    - For temporal trends: "A line chart of monthly review volume would illustrate this trend well"
    - For categories: "A bar chart of top categories by review volume would show this distribution clearly"
    - For sentiment: "A pie chart of sentiment distribution would demonstrate this pattern effectively"
    
    Please provide a clear, data-driven answer to the user's question without generating problematic code.
    """
    
    enhanced_query = f"{dataset_context}\n\nUser Question: {query}\n\nAnswer:"
    
    try:
        # Try invoking the agent with proper error handling
        with st.spinner("🤖 Agent is analyzing the data..."):
            response = agent.invoke({"input": enhanced_query})
            
            # Handle different response formats
            if isinstance(response, dict):
                response_text = response.get("output", str(response))
            else:
                response_text = str(response)
            
            # Check if response is theoretical (mentions "we would need to" or similar)
            theoretical_indicators = [
                'we would need to', 'we can approach', 'calculate the', 'group by',
                'identify patterns', 'look for', 'analyze the data', 'steps to'
            ]
            
            if any(indicator in response_text.lower() for indicator in theoretical_indicators):
                st.info("🔍 Agent provided theoretical framework - generating actual data analysis...")
                if any(word in query.lower() for word in ['seasonal', 'month', 'pattern', 'trend']):
                    return enhanced_seasonal_fallback(query, df), None, "enhanced_fallback"
                elif any(word in query.lower() for word in ['price', 'cost']):
                    return enhanced_price_fallback(query, df), None, "enhanced_fallback"
                else:
                    return enhanced_general_fallback(query, df), None, "enhanced_fallback"
            
            return response_text, None, "agent_analysis"
            
    except Exception as e:
        error_msg = str(e)
        if "max iterations" in error_msg.lower() or "stopped" in error_msg.lower():
            return "Agent analysis was stopped to prevent infinite loops. The question may be too complex for automated analysis. Try rephrasing or using the visualization tools below.", None, "agent_limited"
        else:
            st.warning(f"⚠️ Agent analysis failed: {error_msg}")
            
            # Enhanced fallback based on query type
            if any(word in query.lower() for word in ['seasonal', 'month', 'pattern', 'trend']):
                return enhanced_seasonal_fallback(query, df), None, "fallback_analysis"
            elif any(word in query.lower() for word in ['price', 'cost']):
                return enhanced_price_fallback(query, df), None, "fallback_analysis"
            else:
                return enhanced_general_fallback(query, df), None, "fallback_analysis"

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

def enhanced_price_fallback(query, df):
    """Enhanced fallback specifically for price-related questions"""
    try:
        if 'price' not in df.columns:
            return "Price data is not available in this dataset."
        
        # Check if we have any price data
        price_data = df[df['price'].notna()]
        if len(price_data) == 0:
            return "No price data available for analysis. All price values are missing."
        
        # Convert price to numeric
        price_data = price_data.copy()
        price_data['price_numeric'] = (
            price_data['price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        # Filter out empty strings after cleaning
        price_data = price_data[price_data['price_numeric'] != '']
        price_data['price_numeric'] = price_data['price_numeric'].astype(float)
        
        if len(price_data) == 0:
            return "No valid price data available for analysis."
        
        # Create price ranges
        price_ranges = [0, 25, 50, 100, 200, float('inf')]
        price_labels = ['<$25', '$25-50', '$50-100', '$100-200', '>$200']
        
        price_data['price_range'] = pd.cut(
            price_data['price_numeric'],
            bins=price_ranges,
            labels=price_labels
        )
        
        # Analyze by price range
        analysis_results = []
        
        # Price range distribution
        range_dist = price_data['price_range'].value_counts().sort_index()
        analysis_results.append("**📊 Price Range Distribution:**")
        for price_range, count in range_dist.items():
            pct = (count / len(price_data)) * 100
            analysis_results.append(f"- {price_range}: {count:,} products ({pct:.1f}%)")
        
        # Sentiment by price range
        if 'sentiment' in price_data.columns:
            sentiment_by_price = pd.crosstab(
                price_data['price_range'], 
                price_data['sentiment'], 
                normalize='index'
            ) * 100
            
            analysis_results.append("\n**🎭 Sentiment by Price Range:**")
            for price_range in price_labels:
                if price_range in sentiment_by_price.index:
                    row = sentiment_by_price.loc[price_range]
                    analysis_results.append(f"**{price_range}:**")
                    for sentiment in ['Positive', 'Neutral', 'Negative']:
                        if sentiment in row:
                            analysis_results.append(f"  - {sentiment}: {row[sentiment]:.1f}%")
        
        # Rating by price range
        if 'rating' in price_data.columns:
            rating_by_price = price_data.groupby('price_range')['rating'].mean()
            analysis_results.append("\n**⭐ Average Rating by Price Range:**")
            for price_range in price_labels:
                if price_range in rating_by_price.index:
                    analysis_results.append(f"- {price_range}: {rating_by_price[price_range]:.2f}⭐")
        
        # Summary statistics
        avg_price = price_data['price_numeric'].mean()
        median_price = price_data['price_numeric'].median()
        analysis_results.append(f"\n**💰 Price Statistics:**")
        analysis_results.append(f"- Average price: ${avg_price:.2f}")
        analysis_results.append(f"- Median price: ${median_price:.2f}")
        analysis_results.append(f"- Products with price data: {len(price_data):,} of {len(df):,} total")
        
        return "\n".join(analysis_results)
        
    except Exception as e:
        return f"Price analysis failed: {str(e)}"

def enhanced_general_fallback(query, df):
    """Enhanced fallback for general queries"""
    try:
        query_lower = query.lower()
        

                # Review length analysis
        if any(word in query_lower for word in ['review length', 'correlation', 'word count', 'text length']):
            if 'review_text' in df.columns or 'review_title' in df.columns:
                # Calculate review length
                df_analysis = df.copy()
                df_analysis['review_title'] = df_analysis['review_title'].fillna('')
                df_analysis['review_text'] = df_analysis['review_text'].fillna('')
                df_analysis['full_review'] = df_analysis['review_title'] + ' ' + df_analysis['review_text']
                df_analysis['review_length'] = df_analysis['full_review'].str.len()
                df_analysis = df_analysis[df_analysis['review_length'] > 0]
                
                if len(df_analysis) == 0:
                    return "No review text available for length analysis."
                
                # Calculate correlation
                correlation = df_analysis['review_length'].corr(df_analysis['rating'])
                
                # Calculate average lengths by rating
                avg_by_rating = df_analysis.groupby('rating')['review_length'].mean()
                
                response = [
                    f"**📊 Review Length vs Rating Analysis:**",
                    f"",
                    f"**Correlation Coefficient:** {correlation:.3f}",
                    f"",
                    f"**Interpretation:**",
                    f"- {'Negative' if correlation < 0 else 'Positive'} correlation",
                    f"- {'Weak' if abs(correlation) < 0.3 else 'Moderate' if abs(correlation) < 0.7 else 'Strong'} relationship",
                    f"",
                    f"**Average Review Length by Rating:**"
                ]
                
                for rating in sorted(avg_by_rating.index):
                    response.append(f"- {rating}⭐: {avg_by_rating[rating]:.0f} characters")
                
                response.extend([
                    f"",
                    f"**Insight:** {'Longer reviews tend to have lower ratings' if correlation < 0 else 'Longer reviews tend to have higher ratings'}",
                    f"",
                    f"💡 **Visualization:** A scatter plot showing review length vs rating would illustrate this relationship clearly."
                ])
                
                return "\n".join(response)
            
        # Rating distribution analysis
        if any(word in query_lower for word in ['rating', 'star', 'score']):
            result = df['rating'].value_counts().sort_index()
            response = [
                "**⭐ Rating Distribution Analysis:**",
                ""
            ]
            total = len(df)
            for rating, count in result.items():
                percentage = (count / total) * 100
                response.append(f"- {rating}⭐: {count:,} reviews ({percentage:.1f}%)")
            
            avg_rating = df['rating'].mean()
            response.extend([
                "",
                f"**📊 Summary Statistics:**",
                f"- Average rating: {avg_rating:.2f}⭐",
                f"- Total reviews analyzed: {total:,}",
                f"- Most common rating: {result.idxmax()}⭐ ({result.max():,} reviews)"
            ])
            return "\n".join(response)
        
        # Sentiment analysis
        elif any(word in query_lower for word in ['sentiment', 'positive', 'negative', 'neutral']):
            sentiment_counts = df['sentiment'].value_counts()
            response = [
                "**🎭 Sentiment Analysis:**",
                ""
            ]
            total = len(df)
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total) * 100
                response.append(f"- {sentiment}: {count:,} reviews ({percentage:.1f}%)")
            
            # Sentiment by category if available
            if 'main_category' in df.columns:
                top_categories = df['main_category'].value_counts().head(5).index
                response.extend([
                    "",
                    "**🏷️ Sentiment in Top Categories:**"
                ])
                for category in top_categories:
                    cat_data = df[df['main_category'] == category]
                    cat_sentiment = cat_data['sentiment'].value_counts(normalize=True) * 100
                    if 'Positive' in cat_sentiment:
                        response.append(f"- {category}: {cat_sentiment['Positive']:.1f}% positive")
            
            return "\n".join(response)
        
        # Category analysis
        elif any(word in query_lower for word in ['category', 'categories', 'product type']):
            if 'main_category' in df.columns:
                category_counts = df['main_category'].value_counts().head(10)
                response = [
                    "**🏷️ Category Analysis:**",
                    "",
                    "**Top Categories by Review Volume:**"
                ]
                
                for category, count in category_counts.items():
                    response.append(f"- {category}: {count:,} reviews")
                
                # Average rating by category
                if 'rating' in df.columns:
                    avg_rating_by_cat = df.groupby('main_category')['rating'].mean().sort_values(ascending=False).head(5)
                    response.extend([
                        "",
                        "**⭐ Highest Rated Categories:**"
                    ])
                    for category, rating in avg_rating_by_cat.items():
                        response.append(f"- {category}: {rating:.2f}⭐")
                
                return "\n".join(response)

        # Marketing prioritization request
        if any(word in query_lower for word in ['priorit', 'marketing', 'invest', 'investment', 'prioritize']):
            return enhanced_category_marketing_analysis(query, df)
        
        # Default overview
        return f"""
**📊 Dataset Overview (Enhanced Analysis):**

**Basic Statistics:**
- Total reviews: {len(df):,}
- Categories: {df['main_category'].nunique() if 'main_category' in df.columns else 'N/A'}
- Date range: {df['review_date'].min().strftime('%Y-%m-%d') if 'review_date' in df.columns else 'N/A'} to {df['review_date'].max().strftime('%Y-%m-%d') if 'review_date' in df.columns else 'N/A'}
- Average rating: {df['rating'].mean():.2f}⭐
- Positive sentiment: {(df['sentiment'] == 'Positive').mean()*100:.1f}% if available

**Data Quality Notes:**
- Price data: {df['price'].notna().sum() if 'price' in df.columns else 0} of {len(df)} records have price information
- Review text: {df['review_text'].notna().sum() if 'review_text' in df.columns else 0} records have detailed reviews

*Note: Advanced agent analysis is temporarily unavailable. Basic statistics shown above.*
"""
            
    except Exception as e:
        return f"Enhanced analysis failed: {str(e)}"


def enhanced_category_marketing_analysis(query, df):
    """Compare average rating and review volume across top 5 categories over the last 3 years and recommend a category to prioritize."""
    try:
        if 'review_date' not in df.columns or 'rating' not in df.columns or 'main_category' not in df.columns:
            return "Marketing prioritization requires 'review_date', 'rating' and 'main_category' columns."

        # Robust date parsing
        df_local = df.copy()
        df_local['review_date'] = pd.to_datetime(df_local['review_date'], errors='coerce')
        df_local = df_local[df_local['review_date'].notna()]
        if df_local.empty:
            return "No valid review_date data available for the requested period."

        max_year = df_local['review_date'].dt.year.max()
        last_three = df_local[df_local['review_date'].dt.year.between(max_year-2, max_year)]

        if last_three.empty:
            return "No reviews found in the last three years."

        # Compute metrics
        cat_stats = last_three.groupby('main_category').agg(
            avg_rating=('rating', 'mean'),
            review_volume=('asin', 'count')
        )

        if cat_stats.empty:
            return "No category statistics available."

        # Select top 5 by review volume
        top5 = cat_stats.sort_values('review_volume', ascending=False).head(5)

        # Compute simple growth score: prefer categories with increasing volume year-over-year
        yoy = last_three.copy()
        yoy['year'] = yoy['review_date'].dt.year
        yoy_counts = yoy.groupby(['main_category', 'year']).size().unstack(fill_value=0)

        growth_scores = {}
        for cat in top5.index:
            if cat in yoy_counts.index:
                years = sorted(yoy_counts.columns)
                if len(years) >= 2:
                    # compute CAGR-like trend via slope
                    vals = yoy_counts.loc[cat].reindex(years, fill_value=0).values
                    x = list(range(len(vals)))
                    vx = pd.Series(x)
                    vy = pd.Series(vals)
                    denom = ((vx - vx.mean())**2).sum()
                    slope = ((vx - vx.mean())*(vy - vy.mean())).sum() / denom if denom != 0 else 0
                else:
                    slope = 0
            else:
                slope = 0
            growth_scores[cat] = slope

        # Add growth score to top5
        top5['growth_score'] = [growth_scores.get(cat, 0) for cat in top5.index]

        # Ranking heuristic: prefer high growth and high avg rating weighted by volume
        # score = normalize(growth) * 0.5 + normalize(avg_rating) * 0.3 + normalize(volume) * 0.2
        norm = lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0.0
        growth_norm = pd.Series(top5['growth_score']).fillna(0)
        rating_norm = pd.Series(top5['avg_rating']).fillna(0)
        vol_norm = pd.Series(top5['review_volume']).fillna(0)

        g_n = norm(growth_norm)
        r_n = norm(rating_norm)
        v_n = norm(vol_norm)

        score = 0.5 * g_n + 0.3 * r_n + 0.2 * v_n
        top5['priority_score'] = score.values

        # Recommend top category
        recommended = top5.sort_values('priority_score', ascending=False).iloc[0]

        # Build response
        response = ["**📣 Category Marketing Prioritization (Last 3 Years)**", ""]
        response.append("**Top 5 Categories by Review Volume (last 3 years):**")
        for cat, row in top5.iterrows():
            response.append(f"- {cat}: {int(row['review_volume']):,} reviews, avg rating {row['avg_rating']:.2f}, growth_score {row['growth_score']:.3f}")

        response.append("")
        response.append(f"**Recommendation:** Prioritize **{recommended.name}** — it has the highest combined score (priority_score={recommended['priority_score']:.3f}).")
        response.append("")
        response.append("**Rationale:** This ranking balances recent growth in review volume (demand signal), average rating (quality signal), and absolute review volume (reach).")
        response.append("")
        response.append("**Actionable Next Steps:**")
        response.append("1. Allocate marketing budget to targeted campaigns in the recommended category focusing on best-selling SKUs.")
        response.append("2. Run A/B tests for product detail page improvements in that category.")
        response.append("3. Monitor weekly review volume and sentiment for early warning signals.")

        return "\n".join(response)

    except Exception as e:
        return f"Marketing analysis failed: {str(e)}"

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
        st.plotly_chart(fig)
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
            st.plotly_chart(fig_volume)
        
        with col2:
            st.subheader("💭 Monthly Sentiment Distribution")
            if 'sentiment' in df_viz.columns:
                sentiment_monthly = df_viz.groupby(['month_name', 'sentiment']).size().unstack(fill_value=0)
                fig_sentiment = px.bar(
                    sentiment_monthly,
                    title='Sentiment Distribution by Month',
                    labels={'value': 'Number of Reviews', 'index': 'Month'}
                )
                st.plotly_chart(fig_sentiment)
                
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
            st.plotly_chart(fig_pie)
        
        with col2:
            st.subheader("📊 Sentiment by Rating")
            if 'rating' in df.columns:
                sentiment_rating = df.groupby(['rating', 'sentiment']).size().unstack(fill_value=0)
                fig_rating = px.bar(
                    sentiment_rating,
                    title='Sentiment Distribution by Rating',
                    labels={'value': 'Number of Reviews', 'index': 'Rating'}
                )
                st.plotly_chart(fig_rating)
                
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
            st.plotly_chart(fig_cat)

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
                st.plotly_chart(fig_rating)

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
            st.plotly_chart(fig_dist)
        
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
            st.plotly_chart(fig_stats)
            
    except Exception as e:
        st.error(f"Rating visualization failed: {str(e)}")

def create_ai_driven_visualizations(df, query, response_text):
    """Create visualizations based on AI analysis and query intent"""
    
    query_lower = query.lower()
    
    try:
        # Price-related visualizations
        if any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap']):
            if 'price' in df.columns and df['price'].notna().sum() > 0:
                create_price_analysis_visualizations(df, response_text)
                return True
        
        # Review length analysis
        elif any(word in query_lower for word in ['review length', 'review text', 'correlation', 'length', 'word count']):
            if 'review_text' in df.columns or 'review_title' in df.columns:
                create_review_length_analysis_visualizations(df, response_text)
                return True
        
        # Seasonal/time visualizations  
        elif any(word in query_lower for word in ['seasonal', 'month', 'trend', 'over time']):
            if 'review_date' in df.columns:
                create_temporal_analysis_visualizations(df, response_text)
                return True
        
        # Category visualizations
        elif any(word in query_lower for word in ['category', 'categories', 'product type']):
            if 'main_category' in df.columns:
                create_category_analysis_visualizations(df, response_text)
                return True
        
        # Sentiment visualizations
        elif any(word in query_lower for word in ['sentiment', 'positive', 'negative']):
            if 'sentiment' in df.columns:
                create_sentiment_analysis_visualizations(df, response_text)
                return True
        
        # Rating visualizations
        elif any(word in query_lower for word in ['rating', 'stars', 'score']):
            if 'rating' in df.columns:
                create_rating_analysis_visualizations(df, response_text)
                return True
        
        return False
        
    except Exception as e:
        st.error(f"Visualization creation failed: {str(e)}")
        return False

def create_price_analysis_visualizations(df, response_text):
    """Create comprehensive price analysis visualizations"""
    try:
        st.markdown("### 📊 AI-Driven Price Analysis Visualizations")
        
        # Filter data with prices
        price_data = df[df['price'].notna()].copy()
        if len(price_data) == 0:
            st.warning("No price data available for visualization")
            return
        
        # Convert price to numeric
        price_data['price_numeric'] = (
            price_data['price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
            .astype(float)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig_price_dist = px.histogram(
                price_data, 
                x='price_numeric',
                title='Distribution of Product Prices',
                labels={'price_numeric': 'Price ($)', 'count': 'Number of Products'}
            )
            st.plotly_chart(fig_price_dist)
        
        with col2:
            # Price vs Rating scatter plot
            if 'rating' in price_data.columns:
                fig_price_rating = px.scatter(
                    price_data,
                    x='price_numeric',
                    y='rating',
                    title='Price vs Rating Correlation',
                    labels={'price_numeric': 'Price ($)', 'rating': 'Rating'}
                )
                st.plotly_chart(fig_price_rating)
        
        # Price ranges analysis
        price_ranges = [0, 25, 50, 100, 200, float('inf')]
        price_labels = ['<$25', '$25-50', '$50-100', '$100-200', '>$200']
        
        price_data['price_range'] = pd.cut(
            price_data['price_numeric'],
            bins=price_ranges,
            labels=price_labels
        )
        
        # Sentiment by price range
        if 'sentiment' in price_data.columns:
            sentiment_by_price = price_data.groupby(['price_range', 'sentiment']).size().unstack(fill_value=0)
            fig_sentiment_price = px.bar(
                sentiment_by_price,
                title='Sentiment Distribution by Price Range',
                labels={'value': 'Number of Reviews', 'index': 'Price Range'}
            )
            st.plotly_chart(fig_sentiment_price)
            
    except Exception as e:
        st.error(f"Price visualization failed: {str(e)}")

def create_review_length_analysis_visualizations(df, response_text):
    """Create review length analysis visualizations including scatter plots"""
    try:
        st.markdown("### 📊 Review Length Analysis Visualizations")
        
        # Calculate review length
        df_viz = df.copy()
        
        # Combine title and text for review length calculation
        df_viz['review_title'] = df_viz['review_title'].fillna('')
        df_viz['review_text'] = df_viz['review_text'].fillna('')
        df_viz['full_review'] = df_viz['review_title'] + ' ' + df_viz['review_text']
        df_viz['review_length'] = df_viz['full_review'].str.len()
        
        # Remove empty reviews
        df_viz = df_viz[df_viz['review_length'] > 0]
        
        if len(df_viz) == 0:
            st.warning("No review text available for length analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Review Length vs Rating
            st.subheader("📈 Review Length vs Rating")
            fig_scatter = px.scatter(
                df_viz,
                x='review_length',
                y='rating',
                title='Review Length vs Rating Correlation',
                labels={'review_length': 'Review Length (characters)', 'rating': 'Rating'},
                opacity=0.6  # Makes overlapping points visible
            )
            st.plotly_chart(fig_scatter)
            
            # Calculate actual correlation
            correlation = df_viz['review_length'].corr(df_viz['rating'])
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
            
            # Add interpretation
            if abs(correlation) < 0.1:
                corr_strength = "Very Weak"
            elif abs(correlation) < 0.3:
                corr_strength = "Weak" 
            elif abs(correlation) < 0.5:
                corr_strength = "Moderate"
            else:
                corr_strength = "Strong"
                
            direction = "negative" if correlation < 0 else "positive"
            st.write(f"**Interpretation:** {corr_strength} {direction} correlation")
        
        with col2:
            # Review length distribution by rating
            st.subheader("📊 Review Length by Rating")
            fig_box = px.box(
                df_viz,
                x='rating',
                y='review_length',
                title='Review Length Distribution by Rating',
                labels={'rating': 'Rating', 'review_length': 'Review Length (characters)'}
            )
            st.plotly_chart(fig_box)
            
            # Average review length by rating
            avg_length_by_rating = df_viz.groupby('rating')['review_length'].mean()
            st.markdown("**Average Review Length by Rating:**")
            for rating in sorted(avg_length_by_rating.index):
                avg_length = avg_length_by_rating[rating]
                st.write(f"- {rating}⭐: {avg_length:.0f} characters")
        
        # Additional insights
        st.markdown("#### 🧠 Analyst Insights: Review Length Patterns")
        
        # Find patterns
        avg_length_1star = df_viz[df_viz['rating'] == 1]['review_length'].mean()
        avg_length_5star = df_viz[df_viz['rating'] == 5]['review_length'].mean()
        
        st.info(f"""
        **Key Findings:**
        - 1-star reviews average: **{avg_length_1star:.0f} characters**
        - 5-star reviews average: **{avg_length_5star:.0f} characters**
        - Difference: **{abs(avg_length_1star - avg_length_5star):.0f} characters**
        - Correlation: **{correlation:.3f}** ({corr_strength.lower()} {direction})
        """)
        
        # Business implications
        if correlation < 0:
            st.warning("""
            **Business Implication:** 
            There's a slight tendency for dissatisfied customers to write longer, more detailed reviews. 
            This suggests that negative feedback often contains more specific complaints and explanations.
            **Action:** Focus on analyzing detailed negative reviews for product improvement opportunities.
            """)
        else:
            st.success("""
            **Business Implication:** 
            Positive reviews tend to be more detailed, suggesting satisfied customers are motivated to share their experiences.
            **Action:** Encourage satisfied customers to elaborate on what they liked about the product.
            """)
            
    except Exception as e:
        st.error(f"Review length visualization failed: {str(e)}")
        # Debug information
        st.write(f"Error details: {str(e)}")

def create_temporal_analysis_visualizations(df, response_text):
    """Create temporal/seasonal analysis visualizations"""
    create_seasonal_visualization(df, response_text)

def create_category_analysis_visualizations(df, response_text):
    """Create category analysis visualizations"""
    create_category_visualization(df, response_text)

def create_sentiment_analysis_visualizations(df, response_text):
    """Create sentiment analysis visualizations"""
    create_sentiment_visualization(df, response_text)

def create_rating_analysis_visualizations(df, response_text):
    """Create rating analysis visualizations"""
    create_rating_visualization(df, response_text)

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
            # Try AI-driven visualizations first
            viz_created = create_ai_driven_visualizations(df_filtered, latest['question'], latest['answer'])
            
            # If no specific visualization was created, fall back to keyword-based
            if not viz_created:
                create_agent_visualization(latest['answer'], latest['question'])
    
    # Show history
    if len(st.session_state.get('analyst_history', [])) > 1:
        with st.expander("📝 Agent Analysis History"):
            for i, item in enumerate(reversed(st.session_state.analyst_history[:-1])):
                method = "🤖 Agent" if item.get('used_agent') else "⚠️ Fallback"
                st.markdown(f"**{i+1}. {item['question'][:50]}...** ({method})")
                st.caption(f"Analyzed at {item['timestamp']}")
                st.markdown("---")