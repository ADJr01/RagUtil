from Parser_2.Parser import Parser
from embeddims.embeddings import get_ollama_embeddings



def task():

    embeddings = get_ollama_embeddings()

    parser_conf = {
        "storage_dir": 'D:\\Training\\Course\\CG\\test',
        "persist_dir": "./chroma_db",
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
    parser.close()
    result = retriever.invoke("Real-time rendering")
    print("Result:\n\n\n")
    for doc in result:
        print("="*50)
        print(doc.page_content)
        print("="*50,end='\n\n')


if __name__ == "__main__":
    task()
