try:
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
except ImportError:
    # Fallback imports for older versions
    try:
        from langchain.llms import OpenAI as ChatOpenAI
        from langchain.chains import RetrievalQA
    except ImportError:
        ChatOpenAI = None
        RetrievalQA = None

def create_qa_chain(vectorstore, model_name="gpt-3.5-turbo"):
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
    )
    return qa_chain

def ask_question(qa_chain, query):
    return qa_chain.run(query)

def ask_question_with_sources(vectorstore, query, model_name="gpt-3.5-turbo", search_query=None):
    """Enhanced function that returns both answer and source metadata with diversity"""
    
    # Use the enhanced search query if provided, otherwise use the original query
    actual_search_query = search_query if search_query else query
    
    # Use MMR (Maximum Marginal Relevance) for more diverse results
    # This reduces redundancy and increases diversity in retrieved documents
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 8,  # Get more documents for better coverage
            "fetch_k": 20,  # Fetch more candidates before MMR filtering
            "lambda_mult": 0.7  # Balance between relevance (1.0) and diversity (0.0)
        }
    )
    relevant_docs = retriever.get_relevant_documents(actual_search_query)
    
    # Extract text and metadata
    context_texts = []
    source_metadata = []
    
    for doc in relevant_docs:
        context_texts.append(doc.page_content)
        metadata = doc.metadata
        source_metadata.append({
            "asin": metadata.get("asin", ""),
            "product_title": metadata.get("product_title", ""),
            "review_title": metadata.get("review_title", ""),
            "main_category": metadata.get("main_category", ""),
            "image": metadata.get("image", ""),
            "review_date": metadata.get("review_date", ""),
            "sentiment": metadata.get("sentiment", ""),
            "rating": metadata.get("rating", ""),
            "price": metadata.get("price", "")
        })
    
    # Create context for LLM
    context = "\n\n".join(context_texts)
    
    # Generate answer using LLM
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    prompt = f"""Based on the following product reviews and information, please answer the question.

Context:
{context}

Question: {query}

Please provide a helpful answer based on the reviews above."""
    
    answer = llm.predict(prompt)
    
    return {
        "answer": answer,
        "sources": source_metadata,
        "relevant_docs": relevant_docs
    }
