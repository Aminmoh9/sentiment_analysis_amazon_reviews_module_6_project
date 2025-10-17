# Prompt Engineering for Amazon Reviews Analytics Dashboard

This document collects all relevant prompts used (or recommended) for the AI/LLM features in this project, as well as future prompt ideas for extending or customizing the dashboard.

---

## 1. Sentiment Analysis Prompts

**System Prompt (used for OpenAI sentiment analysis):**
```
You are an expert sentiment analysis system. Analyze the sentiment of product reviews with high precision.

Return ONLY a JSON object with this exact format:
{"sentiment": "Positive", "confidence": 0.85, "reasoning": "brief explanation"}

Sentiment categories:
- Positive: Customer is satisfied, recommends product, expresses happiness/satisfaction
- Negative: Customer is dissatisfied, complains, expresses frustration/disappointment
- Neutral: Mixed feelings, factual statements, or unclear sentiment

Consider:
- Overall tone and emotion
- Specific positive/negative words
- Context of complaints vs compliments
- Rating implications (though focus on text content)

Be conservative - if uncertain, lean toward Neutral.
```

**User Prompt (for each review):**
```
Analyze the sentiment of this product review:

<REVIEW_TEXT>
```

---

## 2. Analyst/LLM Q&A Prompts

**System Prompt (for Analyst tab):**
```
You are a data analyst assistant. Answer business questions about Amazon product reviews using the provided DataFrame. Be concise, use charts if possible, and explain your reasoning.
```

**Example User Prompts:**
- "Compare average rating and review volume across top 5 categories over the last 3 years and recommend which category to prioritize for marketing investment."
- "Show product-level sentiment trends for top products in 'Appliances' category."
- "What are the most common complaints in negative reviews for 2022?"
- "Which categories have the highest price but lowest ratings?"
- "Summarize the main trends in customer sentiment over time."

---

## 3. Chatbot Prompts

**System Prompt:**
```
You are a helpful assistant for Amazon product review analytics. Answer questions, explain charts, and help users explore the data interactively.
```

**Example User Prompts:**
- "What is the average rating for smart home products?"
- "How did sentiment change after 2020?"
- "Which products have the most positive reviews?"
- "Explain the spike in negative reviews in 2018."

---

## 4. Future Prompt Ideas
- Aspect-based sentiment: "Analyze sentiment about battery life in electronics reviews."
- Anomaly detection: "Flag any sudden drops in rating for any category."
- Root cause analysis: "Why did sentiment decline in kitchen appliances in 2019?"
- Custom summary: "Summarize the top 3 insights from this filtered dataset."
- Data quality: "Are there any missing or suspicious values in the data?"

---

## 5. Prompt Engineering Tips
- Always specify output format (e.g., JSON, table, chart)
- Give clear definitions for categories (e.g., what counts as Positive/Negative)
- Use system prompts to set the assistant's role and tone
- Encourage reasoning and explanations, not just answers
- For business users, keep language simple and actionable

---

## 6. Meta-Prompts: Building This Project from Scratch

If you want to use an LLM or AI assistant to create a project like this, here are example prompts to guide the process:

### Project Planning & Setup
- "Create a Streamlit dashboard for analyzing Amazon product reviews, with tabs for overview, time analytics, category analysis, advanced analytics, and AI-powered insights."
- "Design a folder structure for a modular Streamlit analytics app, including components, config, data, utils, and src folders."
- "Write a requirements.txt for a Streamlit dashboard that uses pandas, plotly, and OpenAI."

### Data Handling
- "Write a Python function to load a CSV of Amazon reviews and preprocess columns like rating, review_text, review_date, sentiment, and price."
- "Add a function to infer verified purchases from review text if the column is missing."

### UI Components
- "Create a Streamlit sidebar for filtering reviews by category, year, and rating."
- "Implement an overview tab showing total reviews, average rating, and sentiment distribution as a pie chart."
- "Add a time analytics tab with line charts for review volume and average rating over time."
- "Build a categories tab to compare categories by review count and average rating, with interactive charts."
- "Develop an advanced tab for review length, price analysis, and sentiment trends."

### AI/LLM Integration
- "Integrate OpenAI API to perform sentiment analysis on review text and display results in the dashboard."
- "Add an analyst tab where users can ask business questions in natural language and get data-driven answers using an LLM."
- "Create a chatbot tab that lets users chat with the review data using OpenAI or LangChain."

### Vector Store (Pinecone)
- "Add a button to build a Pinecone vector store from the reviews for semantic search and chatbot features."
- "Document how to set Pinecone and OpenAI API keys in a .env file."

### Documentation
- "Write a README.md explaining the dashboard, its features, setup instructions, and data limitations."
- "Document all prompt engineering and AI prompt templates used in the project."

### Testing & Extension
- "Suggest ways to extend the dashboard with new tabs, more advanced AI, or user-uploaded datasets."
- "Add defensive checks for missing columns and data quality issues."

---

These meta-prompts can be used with an AI coding assistant to generate the code, structure, and documentation for a project like this from scratch.
