# Amazon Product Reviews Analytics Dashboard

## Overview
This Streamlit dashboard lets you explore, analyze, and visualize Amazon product reviews. It is designed for business users, product managers, and analysts to gain actionable insights from review data—no coding required.

## Features
- **Interactive filtering:** Slice data by category, year, rating, and more
- **Overview tab:** Key metrics, top categories, and sentiment distribution
- **Time Analytics tab:** Trends in review volume and ratings over time
- **Categories tab:** Compare categories by review count and average rating
- **Advanced tab:** Deep dive into review length, price, and sentiment trends
- **AI-powered tabs:** (If enabled) Ask business questions and get insights using OpenAI models
- **Chatbot:** (If enabled) Chat with your data for instant answers

## Who is this for?
- E-commerce product managers
- Marketing and business analysts
- Data scientists
- Customer experience teams

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your API keys to `.env` for AI and Pinecone features:
   ```env
   OPENAI_API_KEY=your-openai-key-here
   PINECONE_API_KEY=your-pinecone-key-here
   ```
3. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

## Data
- Main dataset: `data/appliances_10k_sample.csv`
- Columns: `asin`, `rating`, `review_title`, `review_text`, `review_date`, `year`, `sentiment`, `product_title`, `main_category`, `price`

## AI Features
- Sentiment analysis and Q&A powered by OpenAI (if API key provided)
- Fallback to rule-based analysis if no API key

## Customization
- Add your own CSVs to the `data/` folder
- Adjust config in `config/settings.py`

## Extending
- Add new tabs for deeper analysis
- Integrate more advanced AI models
- Enable export/reporting features

## License
MIT

## Links
- **Original dataset:** [https://amazon-reviews-2023.github.io/]
- **Presentation slides/demo:** [Add link here]
- **Author LinkedIn:** [https://www.linkedin.com/in/sayed-mohd-amin-mohammadi-49873b96/]

## Data Limitations
- This dashboard uses a 10,000-review sample (`appliances_10k_sample.csv`) for performance and demo purposes.
- The sample may not represent all products, categories, or time periods in the full Amazon reviews dataset.
- Some columns (e.g., `verified_purchase`) are inferred or missing; results may differ from a full dataset analysis.
- Sentiment labels may be model-generated or heuristic, not always ground truth.

## Pinecone Vector Store
- The dashboard can create a Pinecone vector store for semantic search and chatbot features.
- To enable, add your Pinecone API key to the `.env` file as shown above.
- Use the sidebar button to build or refresh the vector store (requires internet and Pinecone account).

## Folder Structure
```
project/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── .env                   # API keys (create this)
├── data/                  # Data files
│   └── appliances_10k_sample.csv
├── config/                # App settings
│   └── settings.py
├── components/            # UI components (tabs, sidebar, etc.)
├── utils/                 # Utility scripts (data loading, API keys)
├── src/                   # Core processing (preprocessing, Pinecone, chatbot)
└── README.md              # Project documentation
```

---
For questions or demo requests, contact the author.