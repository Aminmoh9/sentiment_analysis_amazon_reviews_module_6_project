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
        query_lower = query.lower()
        
        # EXPANDED BABY TERMS
        baby_terms = [
            'baby', 'infant', 'toddler', 'crib', 'stroller', 'diaper', 'pacifier', 
            'bottle', 'child proof', 'childproof', 'child safety', 'nursery', 
            'breast pump', 'humidifier', 'baby lock', 'fridge lock', 'dishwasher basket',
            'child', 'kids', 'parent', 'mom', 'dad'
        ]
        
        # Define filters based on user intent
        filters = {}
        
        # BABY PRODUCT FILTER - IMPROVED
        if any(term in query_lower for term in baby_terms):
            filters['categories'] = ['baby']  # Focus on main_category
            
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
        
        # GET MORE RESULTS INITIALLY
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k * 5,  # Increased from 3 to 5
                "fetch_k": 100,  # Increased from 50 to 100  
                "lambda_mult": 0.7
            }
        )
        relevant_docs = retriever.get_relevant_documents(query)
        
        # DEBUG INFO
        print(f"🔍 Initial relevant docs found: {len(relevant_docs)}")
        print(f"📝 Baby filter active: {any(term in query_lower for term in baby_terms)}")
        if filters.get('categories'):
            print(f"🎯 Category filter: {filters['categories']}")
        
        results = []
        skipped_count = 0
        
        for doc in relevant_docs:
            metadata = doc.metadata
            rating = metadata.get('rating', 'N/A')
            sentiment = metadata.get('sentiment', 'N/A')
            main_category = metadata.get('main_category', '').lower()
            product_name = metadata.get('product_title', '').lower()
            
            # Convert rating to float for comparison
            try:
                rating_float = float(rating) if rating != 'N/A' else 0
            except (ValueError, TypeError):
                rating_float = 0
            
            # IMPROVED CATEGORY FILTERING
            if filters.get('categories'):
                # Strict check on main_category for baby products
                category_match = main_category in filters['categories']
                
                if not category_match:
                    skipped_count += 1
                    continue
                    
            # Apply rating filter
            if filters.get('min_rating') and rating_float < filters['min_rating']:
                skipped_count += 1
                continue
                
            # Apply sentiment filter  
            if filters.get('sentiment') and sentiment != filters['sentiment']:
                skipped_count += 1
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
        
        # IMPROVED FALLBACK FOR BABY PRODUCTS
        if len(results) == 0 and any(term in query_lower for term in baby_terms):
            print("🔄 No baby products found with strict filtering, trying fallback search...")
            
            # Try broader search including product titles and review text
            for doc in relevant_docs:
                metadata = doc.metadata
                product_name = metadata.get('product_title', '').lower()
                review_text = doc.page_content.lower()
                main_category = metadata.get('main_category', '').lower()
                
                # Check if product title or review text contains baby terms
                title_match = any(term in product_name for term in baby_terms)
                review_match = any(term in review_text for term in baby_terms)
                category_match = main_category == 'baby'
                
                if title_match or review_match or category_match:
                    rating = metadata.get('rating', 'N/A')
                    try:
                        rating_float = float(rating) if rating != 'N/A' else 0
                    except (ValueError, TypeError):
                        rating_float = 0
                    
                    # Apply minimum rating filter if specified
                    if filters.get('min_rating') and rating_float < filters['min_rating']:
                        continue
                        
                    results.append({
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
        
        print(f"✅ Final results: {len(results)}, Skipped: {skipped_count}")
        
        # Sort by rating (highest first) and limit to top_k
        results.sort(key=lambda x: x['rating_float'], reverse=True)
        final_results = results[:top_k]
        
        # Final debug info
        if final_results:
            print(f"🎉 Returning {len(final_results)} products:")
            for i, product in enumerate(final_results):
                print(f"  {i+1}. {product['product']} | Rating: {product['rating']} | Category: {product['category']}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        st.error(f"Search error: {str(e)}")
        return []


def generate_baby_product_response(query, similar_products, client):
    """Generate specialized response for baby product queries - STRICTLY NO HALLUCINATION"""
    if not similar_products:
        return "I searched through our product database but couldn't find specific baby products with high ratings. You might want to check Amazon directly for baby gear recommendations."
    
    # Check if we actually found baby products
    baby_terms = [
        'baby', 'infant', 'toddler', 'crib', 'stroller', 'diaper', 
        'pacifier', 'bottle', 'child proof', 'childproof', 'child safety'
    ]
    
    has_real_baby_products = any(
        any(term in product['product'].lower() or term in product['category'].lower() 
            for term in baby_terms)
        for product in similar_products
    )
    
    # Create detailed product context
    actual_products_text = "\n".join([
        f"Product {i+1}: {p['product']} | Rating: {p['rating']}/5 | Category: {p['category']} | "
        f"Sentiment: {p['sentiment']} | Price: {p['price']} | "
        f"Key Review: {p['review_title']} - {p['text'][:100]}..."
        for i, p in enumerate(similar_products)
    ])
    
    if not has_real_baby_products:
        prompt = f"""You are a helpful shopping assistant. The user asked about baby products but we couldn't find specific baby items in our current dataset. 

ACTUAL PRODUCTS FOUND:
{actual_products_text}

User Question: {query}

Provide an honest response that:
1. Acknowledges we couldn't find specific baby products
2. Mentions the other highly-rated products we did find (ONLY the ones listed above)
3. Suggests checking Amazon directly for more baby product options
4. Does NOT invent or hallucinate any baby products

Answer:"""
    else:
        prompt = f"""You are a helpful baby product expert. You MUST follow these rules STRICTLY:

RULES:
1. ONLY mention products that are explicitly listed below - DO NOT make up or invent any products
2. Be honest about how many products we found
3. Focus on the actual ratings, features, and reviews from the data
4. DO NOT add any products that aren't in the list below
5. If we found fewer than 3 products, acknowledge this limitation

ACTUAL PRODUCTS FOUND:
{actual_products_text}

User Question: {query}

IMPORTANT: You MUST ONLY talk about the products listed above. Do not mention any other products.

Provide a helpful answer that:
1. ONLY discusses the specific products listed above
2. Mentions the actual ratings and key features from the reviews
3. Is honest about how many products we found
4. Does not invent or hallucinate any products
5. Groups similar products together when appropriate

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3  
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback response if API fails
        product_list = "\n".join([
            f"• {p['product']} (Rating: {p['rating']}/5, Price: {p['price']})"
            for p in similar_products
        ])
        return f"I found these products in our database:\n\n{product_list}\n\nThese are the actual products we have information on based on your query about baby products."


# ADDITIONAL DEBUG FUNCTION TO CHECK DATA
def debug_baby_products(vectorstore):
    """Debug function to check what baby products are actually in the vectorstore"""
    try:
        # Simple search for baby products
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 20}
        )
        test_docs = retriever.get_relevant_documents("baby products infant toddler")
        
        print("🔍 DEBUG: Baby Products in Vectorstore")
        print("=" * 50)
        
        baby_products = []
        other_products = []
        
        for doc in test_docs:
            metadata = doc.metadata
            category = metadata.get('main_category', '').lower()
            product_name = metadata.get('product_title', '')
            
            if category == 'baby':
                baby_products.append({
                    'product': product_name,
                    'category': category,
                    'rating': metadata.get('rating', 'N/A'),
                    'asin': metadata.get('asin', '')
                })
            else:
                other_products.append({
                    'product': product_name, 
                    'category': category,
                    'rating': metadata.get('rating', 'N/A')
                })
        
        print(f"🎯 ACTUAL BABY PRODUCTS FOUND: {len(baby_products)}")
        for product in baby_products:
            print(f"   • {product['product']} | Rating: {product['rating']} | ASIN: {product['asin']}")
        
        print(f"\n📦 OTHER PRODUCTS: {len(other_products)}")
        for product in other_products[:5]:  # Show first 5
            print(f"   • {product['product']} | Category: {product['category']} | Rating: {product['rating']}")
            
        return baby_products
        
    except Exception as e:
        print(f"❌ Debug error: {str(e)}")
        return []

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