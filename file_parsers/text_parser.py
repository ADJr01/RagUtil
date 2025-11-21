from logging import exception

from langchain_community.document_loaders import (TextLoader, DirectoryLoader)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

class TextParser:
    def __init__(self):
        self.parser = RecursiveCharacterTextSplitter

    def load_single_text_doc_from_dir(path:str):
        try:
            return TextLoader(path,encoding="utf-8")
        except exception as e:
            print(e)
            return None

    def load_all_text_doc_from_dir(path:str,pattern:str='**/*.txt',show_progress:bool=False):
        try:
            dir_loader = DirectoryLoader(
                path=path,
                glob=pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding":"utf-8"},
                show_progress=show_progress,
            )
            files_from_dir = dir_loader.load()
            return files_from_dir
        except exception as e:
            print(e)
            return None

    def text_to_chunks(self,text:str,max_chunk_size:int=200,chunk_overlap_limit:int=20,len_counter=len):
        try:
            splitter = self.parser( #Recursive character text splitter
                separators=["\n\n", "\n", ".", " ", ""],
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap_limit,
                length_function=len,
            )
            chunks =splitter.split_text(text)
            return chunks
        except exception as e:
            print(e)
            return None

