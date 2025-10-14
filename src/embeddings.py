try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        from langchain.embeddings import OpenAIEmbeddings
    except ImportError:
        OpenAIEmbeddings = None

def create_embeddings(texts, model_name="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=model_name)
    return embeddings.embed_documents(texts)
