


amazon-reviews-analytics/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── .env                           # API keys (create this)
├── .gitignore
│
├── data/                          # Data directory
│   ├── Appliances.jsonl          # Original raw data
│   ├── meta_Appliances.jsonl     # Original raw metadata
│   └── appliances_10k_sample.csv # ONLY processed 10K sample
│
├── config/
│   ├── __init__.py
│   └── settings.py               # Single dataset configuration
│
├── components/                    # All UI components
│   ├── __init__.py
│   ├── sidebar.py                # Simplified sidebar (10K only)
│   ├── overview_tab.py           # Tab 1
│   ├── time_analytics_tab.py     # Tab 2
│   ├── categories_tab.py         # Tab 3
│   ├── advanced_tab.py           # Tab 4
│   ├── data_explorer_tab.py      # Tab 5
│   ├── interactive_chatbot_tab.py    # Tab 6 - Product Assistant
│   ├── interactive_analyst_tab.py    # Tab 7 - Data Analyst
│   └── interactive_sentiment_tab.py  # Tab 8 - Sentiment Detective
│
├── utils/                         # Utilities
│   ├── __init__.py
│   ├── data_loader.py            # Simplified loader (10K only)
│   └── api_keys.py               # API key management
│
└── src/                          # Core processing
    ├── __init__.py
    ├── preprocess_data.py        # MAIN preprocessing (10K only)
    ├── data_utils.py             # Consolidated utility functions
    ├── pinecone_utils.py         # Vector store management
    └── chatbot.py                # Chatbot core functionality