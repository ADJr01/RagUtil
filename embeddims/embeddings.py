from langchain_ollama import OllamaEmbeddings
def get_ollama_embeddings(model='qwen3-embedding:0.6b'):
    return OllamaEmbeddings(model=model)