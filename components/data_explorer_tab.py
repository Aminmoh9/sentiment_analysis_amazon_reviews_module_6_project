# [file name]: components/data_explorer_tab.py
"""
Data Explorer Tab - Dataset statistics, sample data view, and export functionality
"""
import streamlit as st
from datetime import datetime
from config.settings import DEFAULT_DISPLAY_COLUMNS

def render_data_explorer_tab(df_filtered):
    """Render data explorer tab with dataset statistics and export options"""
    
    # Dataset statistics section
    st.subheader("📊 Dataset Statistics")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Total Records", f"{len(df_filtered):,}")
        if len(df_filtered) > 0:
            date_range = f"{df_filtered['year'].min()}-{df_filtered['year'].max()}"
            st.metric("Date Range", date_range)
        else:
            st.metric("Date Range", "No data")
    
    with stats_col2:
        categories_count = df_filtered['main_category'].nunique()
        st.metric("Categories", categories_count)
        if len(df_filtered) > 0:
            avg_rating = df_filtered['rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.2f}")
        else:
            st.metric("Avg Rating", "N/A")
    
    with stats_col3:
        positive_count = (df_filtered['sentiment'] == 'Positive').sum()
        st.metric("Positive Reviews", f"{positive_count:,}")
        
        if len(df_filtered) > 0:
            recent_reviews = df_filtered[df_filtered['year'] >= df_filtered['year'].max() - 1]
            st.metric("Recent Reviews (Last 2Y)", f"{len(recent_reviews):,}")
        else:
            st.metric("Recent Reviews (Last 2Y)", "0")
    
    # Sample data view section
    st.subheader(f"🔍 Sample Records ({len(df_filtered):,} total)")
    
    if len(df_filtered) > 0:
        # Column selector
        all_columns = df_filtered.columns.tolist()
        
        # Use default columns that exist in the dataframe
        available_defaults = [col for col in DEFAULT_DISPLAY_COLUMNS if col in all_columns]
        
        display_columns = st.multiselect(
            "Select columns to display:",
            options=all_columns,
            default=available_defaults,
            help="Choose which columns to show in the data preview"
        )
        
        if display_columns:
            # Number of records to show
            max_records = min(100, len(df_filtered))
            num_records = st.slider(
                "Number of records to display:", 
                min_value=5,
                max_value=max_records,
                value=min(20, max_records),
                help=f"Display up to {max_records} records"
            )
            
            # Show sample data
            sample_df = df_filtered[display_columns].head(num_records)
            st.dataframe(sample_df, width=900, height=400)
            
            # Data quality info
            with st.expander("📋 Data Quality Summary"):
                quality_col1, quality_col2 = st.columns(2)
                
                with quality_col1:
                    st.write("**Missing Values:**")
                    missing_counts = sample_df.isnull().sum()
                    for col, missing in missing_counts.items():
                        if missing > 0:
                            st.write(f"• {col}: {missing} ({missing/len(sample_df)*100:.1f}%)")
                        else:
                            st.write(f"• {col}: ✅ Complete")
                
                with quality_col2:
                    st.write("**Data Types:**")
                    for col in display_columns:
                        dtype = str(sample_df[col].dtype)
                        st.write(f"• {col}: {dtype}")
        else:
            st.info("👆 Select columns above to preview the data")
    else:
        st.warning("No data available to display. Please adjust your filters.")
    
    # Download section
    st.subheader("💾 Data Export")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if len(df_filtered) > 0:
            # Prepare CSV data
            csv_data = df_filtered.to_csv(index=False)
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f"amazon_multicategory_filtered_{current_date}.csv"
            
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help=f"Download {len(df_filtered):,} filtered records as CSV file"
            )
            
            # File size info
            file_size_mb = len(csv_data.encode('utf-8')) / (1024 * 1024)
            st.caption(f"File size: ~{file_size_mb:.1f} MB")
            
        else:
            st.info("No data to download. Please adjust filters to include data.")
    
    with export_col2:
        # Dataset information
        st.info(""" 
        💡 **Dataset Information:**
        - **Source:** Stratified 10K sample from Amazon Appliances dataset
        - **Processing:** Cleaned and preprocessed with complete metadata
        - **Categories:** 24 product categories represented
        - **Quality:** 100% complete metadata with stratified sampling
        - **Format:** CSV with UTF-8 encoding
        """)
    
    # Advanced export options
    if len(df_filtered) > 0:
        with st.expander("🔧 Advanced Export Options"):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                # Export specific columns only
                st.write("**Custom Column Export:**")
                export_columns = st.multiselect(
                    "Select columns for export:",
                    options=df_filtered.columns.tolist(),
                    default=available_defaults,
                    key="export_columns"
                )
                
                if export_columns:
                    export_df = df_filtered[export_columns]
                    export_csv = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download Selected Columns",
                        data=export_csv,
                        file_name=f"amazon_custom_columns_{current_date}.csv",
                        mime="text/csv",
                        help=f"Download {len(export_columns)} selected columns"
                    )
            
            with adv_col2:
                # Export summary statistics
                st.write("**Summary Statistics Export:**")
                if st.button("📈 Generate Summary Stats"):
                    # Create summary statistics
                    summary_stats = {}
                    
                    # Numerical columns summary
                    numeric_cols = df_filtered.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        summary_stats['numerical_summary'] = df_filtered[numeric_cols].describe()
                    
                    # Categorical columns summary
                    categorical_cols = df_filtered.select_dtypes(include=['object']).columns
                    categorical_summary = {}
                    for col in categorical_cols:
                        value_counts = df_filtered[col].value_counts().head(10)
                        categorical_summary[col] = value_counts.to_dict()
                    
                    if categorical_summary:
                        summary_stats['categorical_summary'] = categorical_summary
                    
                    # Display summary
                    st.json(summary_stats)
    
    # Data insights section
    if len(df_filtered) > 0:
        st.subheader("🔍 Quick Data Insights")
        
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            # Most common values
            st.write("**Most Common Category:**")
            top_category = df_filtered['main_category'].value_counts().index[0]
            top_count = df_filtered['main_category'].value_counts().iloc[0]
            st.success(f"{top_category}\n{top_count:,} reviews")
        
        with insights_col2:
            # Rating distribution
            st.write("**Rating Distribution:**")
            rating_mode = df_filtered['rating'].mode()[0] if len(df_filtered['rating'].mode()) > 0 else "N/A"
            rating_count = (df_filtered['rating'] == rating_mode).sum() if rating_mode != "N/A" else 0
            st.info(f"Most common: {rating_mode}⭐\n{rating_count:,} reviews")
        
        with insights_col3:
            # Sentiment balance
            st.write("**Sentiment Balance:**")
            sentiment_counts = df_filtered['sentiment'].value_counts()
            dominant_sentiment = sentiment_counts.index[0]
            dominant_pct = sentiment_counts.iloc[0] / len(df_filtered) * 100
            st.warning(f"{dominant_sentiment}\n{dominant_pct:.1f}% of reviews")