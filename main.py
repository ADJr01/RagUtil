from file_parsers.Parser import Parser
from embeddims.embeddings import get_ollama_embeddings



def task():
    books = 'D:\\Training\\Course\\CG\\test'
    collection = "./chroma_db"
    embeddings = get_ollama_embeddings()
    parser_conf = {
        "storage_dir": books,
        "persist_dir": collection,
        "collection_name":"books",
        "embedding":embeddings,
        "chunk_size":2000,
        "chunk_overlap":400,
        "verbose":True
    }
    parser = Parser(parser_conf)
    parser.load()
    vector_store = parser.get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k":5},
    )
    result = retriever.invoke("what is Real-time rendering")
    print("Result:\n\n\n")
    for doc in result:
        print("="*50)
        print(doc.page_content)
        print("="*50,end='\n\n')

if __name__ == "__main__":
    task()
