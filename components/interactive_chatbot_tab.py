# [file name]: components/interactive_chatbot_tab.py
"""
Interactive AI Chatbot - Ask questions about products and reviews
"""
import streamlit as st
import pandas as pd
import os
from utils.api_keys import check_api_keys

def get_openai_client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except ImportError:
        st.error("OpenAI library not installed.")
        return None

def get_vector_store():
    """Get Pinecone vector store"""
    try:
        from src.pinecone_utils import load_vector_store
        # Use the correct name directly
        return load_vector_store("amazon-reviews")
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def semantic_search_products(query, vectorstore, top_k=5):
    """Search for products using semantic search with intelligent filtering"""
    try:
        # Detect user intent from query
        query_lower = query.lower()
        
        # Define filters based on user intent
        filters = {}
        
        # BABY PRODUCT FILTER - NEW ADDITION
        baby_terms = ['baby', 'infant', 'toddler', 'crib', 'stroller', 'diaper', 'pacifier', 'bottle']
        if any(term in query_lower for term in baby_terms):
            filters['categories'] = ['baby', 'infant', 'toddler', 'nursery']
            
        if any(term in query_lower for term in [
            'best rated', 'highly rated', 'top rated', 'best rating', 
            'high rating', 'good rating', 'highly recommended'
        ]):
            filters['min_rating'] = 4.0
            
        if any(term in query_lower for term in [
            'kitchen', 'cooking', 'baking', 'refrigerator', 'oven', 'stove'
        ]):
            filters['categories'] = ['kitchen', 'cooking', 'appliances']
            
        if any(term in query_lower for term in [
            'positive', 'good', 'satisfied', 'happy'
        ]):
            filters['sentiment'] = 'Positive'
            
        if any(term in query_lower for term in [
            'negative', 'bad', 'complaint', 'problem', 'issue'
        ]):
            filters['sentiment'] = 'Negative'
        
        # Get more results initially for filtering
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k * 3, "fetch_k": 50, "lambda_mult": 0.7}
        )
        relevant_docs = retriever.get_relevant_documents(query)
        
        results = []
        for doc in relevant_docs:
            metadata = doc.metadata
            rating = metadata.get('rating', 'N/A')
            sentiment = metadata.get('sentiment', 'N/A')
            category = metadata.get('main_category', '').lower()
            product_name = metadata.get('product_title', '').lower()
            
            # Convert rating to float for comparison
            try:
                rating_float = float(rating) if rating != 'N/A' else 0
            except (ValueError, TypeError):
                rating_float = 0
            
            # Apply filters
            if filters.get('min_rating') and rating_float < filters['min_rating']:
                continue
                
            if filters.get('sentiment') and sentiment != filters['sentiment']:
                continue
                
            # ENHANCED CATEGORY FILTERING - NEW
            if filters.get('categories'):
                category_match = any(cat in category for cat in filters['categories'])
                product_name_match = any(cat in product_name for cat in filters['categories'])
                
                # For baby products, be more strict
                if 'baby' in str(filters['categories']):
                    if not (category_match or product_name_match):
                        continue
                else:
                    if not category_match:
                        continue
            
            results.append({
                'text': doc.page_content,
                'rating': rating,
                'rating_float': rating_float,
                'category': metadata.get('main_category', 'N/A'),
                'sentiment': sentiment,
                'product': metadata.get('product_title', 'Unknown Product'),
                'price': metadata.get('price', 'N/A'),
                'asin': metadata.get('asin', ''),
                'image': metadata.get('image', ''),
                'review_title': metadata.get('review_title', ''),
                'amazon_link': f"https://amazon.com/dp/{metadata.get('asin', '')}" if metadata.get('asin') else None,
            })
        
        # SPECIAL HANDLING FOR BABY PRODUCTS - NEW
        # If no baby products found but query is about babies, try broader search
        if len(results) == 0 and any(term in query_lower for term in baby_terms):
            st.info("🔍 Couldn't find specific baby products. Showing general highly-rated products instead.")
            # Remove baby filter and try again with just high rating
            if 'categories' in filters:
                del filters['categories']
            # Re-process with just rating filter
            filtered_results = []
            for doc in relevant_docs:
                metadata = doc.metadata
                rating = metadata.get('rating', 'N/A')
                try:
                    rating_float = float(rating) if rating != 'N/A' else 0
                except (ValueError, TypeError):
                    rating_float = 0
                
                if rating_float >= 4.0:  # Only high-rated products
                    filtered_results.append({
                        'text': doc.page_content,
                        'rating': rating,
                        'rating_float': rating_float,
                        'category': metadata.get('main_category', 'N/A'),
                        'sentiment': metadata.get('sentiment', 'N/A'),
                        'product': metadata.get('product_title', 'Unknown Product'),
                        'price': metadata.get('price', 'N/A'),
                        'asin': metadata.get('asin', ''),
                        'image': metadata.get('image', ''),
                        'review_title': metadata.get('review_title', ''),
                        'amazon_link': f"https://amazon.com/dp/{metadata.get('asin', '')}" if metadata.get('asin') else None,
                    })
            results = filtered_results
        
        # Sort by rating (highest first) and limit to top_k
        results.sort(key=lambda x: x['rating_float'], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def generate_baby_product_response(query, similar_products, client):
    """Generate specialized response for baby product queries"""
    if not similar_products:
        return "I searched through our product database but couldn't find specific baby products with high ratings. You might want to check Amazon directly for baby gear recommendations."
    
    # Check if we actually found baby products
    baby_terms = ['baby', 'infant', 'toddler', 'crib', 'stroller', 'diaper']
    has_real_baby_products = any(
        any(term in product['product'].lower() or term in product['category'].lower() 
            for term in baby_terms)
        for product in similar_products
    )
    
    if not has_real_baby_products:
        return "I couldn't find specific baby products in our current dataset, but here are some other highly-rated products that might interest you:"
    
    # Create context from found baby products
    context = "\n".join([
        f"Product: {p['product']} | Rating: {p['rating']}/5 | Category: {p['category']} | Review: {p['text'][:80]}..."
        for p in similar_products[:3]
    ])
    
    prompt = f"""You're a helpful baby product expert. Based on these products:

{context}

User Question: {query}

Provide a helpful answer that:
1. Focuses on baby product recommendations
2. Mentions specific baby products found
3. Highlights safety and quality aspects
4. Is reassuring and helpful for parents

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I encountered an error: {str(e)}"

def generate_chatbot_response(query, similar_products, client):
    """Generate conversational response about products"""
    if not similar_products:
        return "I couldn't find any products matching your query. Try asking about specific products, features, or categories."
    
    # SPECIAL HANDLING FOR BABY PRODUCTS - NEW
    baby_terms = ['baby', 'infant', 'toddler', 'crib', 'stroller', 'diaper']
    if any(term in query.lower() for term in baby_terms):
        return generate_baby_product_response(query, similar_products, client)
    
    # Original logic for other queries
    # Create context from found products
    context = "\n".join([
        f"Product: {p['product']} | Rating: {p['rating']}/5 | Sentiment: {p['sentiment']} | Review: {p['text'][:100]}..."
        for p in similar_products[:3]
    ])
    
    prompt = f"""You're a helpful Amazon product assistant. Based on these products and reviews:

{context}

User Question: {query}

Provide a helpful, conversational answer that:
1. Directly addresses the user's question
2. Mentions specific products when relevant
3. Highlights ratings and customer sentiments
4. Is friendly and informative

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I encountered an error: {str(e)}"

def render_interactive_chatbot_tab(df_filtered):
    """Render interactive chatbot tab"""
    
    st.header("🤖 Product & Review Assistant")
    st.markdown("Ask questions about specific products, features, or customer experiences!")
    
    # Check API keys
    if not check_api_keys()['openai']:
        st.error("🔑 OpenAI API key required")
        return
    
    client = get_openai_client()
    if not client:
        return
    
    # Initialize session state
    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = []
    
    # Example questions
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Best rated products?"):
                st.session_state.chatbot_input = "What are the best rated products in the dataset?"
            if st.button("⚠️ Common complaints?"):
                st.session_state.chatbot_input = "What are the most common complaints in reviews?"
            if st.button("👶 Baby products?"):  # NEW BABY PRODUCT BUTTON
                st.session_state.chatbot_input = "What are the best rated baby products?"
        with col2:
            if st.button("🏷️ Kitchen products?"):
                st.session_state.chatbot_input = "Show me highly rated kitchen appliances"
            if st.button("💸 Price vs quality?"):
                st.session_state.chatbot_input = "Do expensive products have better ratings?"
            if st.button("⭐ Top reviews?"):
                st.session_state.chatbot_input = "Show me products with 5-star reviews"
    
    # Chat input
    user_input = st.text_input(
        "Ask about products or reviews:",
        value=st.session_state.get('chatbot_input', ''),
        placeholder="e.g., 'What are the best washing machines?' or 'Show me products with delivery issues'"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Ask", type="primary") and user_input.strip():
            with st.spinner("Searching products and generating response..."):
                # Get vector store
                vectorstore = get_vector_store()
                
                if vectorstore:
                    # Search for relevant products
                    similar_products = semantic_search_products(user_input, vectorstore)
                    
                    # Generate response
                    response = generate_chatbot_response(user_input, similar_products, client)
                    
                    # Add to chat history
                    st.session_state.chatbot_messages.append({
                        "question": user_input,
                        "answer": response,
                        "products": similar_products,
                        "type": "product_query"
                    })
                else:
                    st.error("Vector store not available. Please load it first.")
            
            # Clear input
            if 'chatbot_input' in st.session_state:
                del st.session_state.chatbot_input
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chatbot_messages = []
            st.rerun()
    
    # Display conversation
    st.markdown("---")
    st.subheader("💬 Conversation")
    
    if not st.session_state.chatbot_messages:
        st.info("👆 Ask a question above to start chatting about products!")
    
    for i, chat in enumerate(reversed(st.session_state.chatbot_messages)):
        # User question
        st.markdown(f"**🧑‍💻 You:** {chat['question']}")
        
        # AI response
        st.markdown(f"**🤖 Assistant:** {chat['answer']}")
        
        # Show products if available
        if chat.get('products'):
            st.markdown("**🛍️ Related Products:**")
            
            for product in chat['products'][:3]:  # Show top 3
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Sentiment color
                    sentiment_color = {
                        'Positive': '🟢',
                        'Negative': '🔴', 
                        'Neutral': '🟡'
                    }.get(product['sentiment'], '⚪')
                    
                    st.markdown(f"**{product['product']}**")
                    st.markdown(f"{sentiment_color} **{product['sentiment']}** | ⭐ **{product['rating']}/5** | 💰 **{product['price']}**")
                    
                    # Review snippet
                    if product.get('text'):
                        st.caption(f"*\"{product['text'][:100]}...\"*")
                    
                    # Amazon link
                    if product.get('amazon_link'):
                        st.markdown(f"[🔗 View on Amazon]({product['amazon_link']})")
                
                with col2:
                    if product.get('image') and product['image'].startswith('http'):
                        st.image(product['image'], width=80)
                    else:
                        st.markdown("📦")
                
                st.markdown("---")
        
        st.markdown("")  # Spacing