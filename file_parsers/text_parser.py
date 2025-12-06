import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class TextParser:
    def __init__(self):
        self.file_paths = None          # can be a single file OR directory
        self.mode = None                # "single" or "directory"

    def load_from(self, path: str, directory_mode: bool = False, pattern: str = "**/*.txt"):
        """
        directory_mode = False -> Treat path as a single text file
        directory_mode = True  -> Treat path as a folder of text files
        Saves the paths for later processing.
        """
        if not isinstance(path, str):
            raise ValueError("path must be a string")

        if not os.path.exists(path):
            raise FileNotFoundError("Provided path does not exist")

        if directory_mode is False:
            # single text file
            ext = os.path.splitext(path)[1].lower()
            if ext not in [".txt", ".md"]:
                raise ValueError("File must be .txt or .md")
            self.file_paths = path
            self.mode = "single"

        else:
            # directory mode
            if not os.path.isdir(path):
                raise ValueError("Path must be a directory when directory_mode=True")
            self.file_paths = (path, pattern)
            self.mode = "directory"

    def _load_single(self, path: str):
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    def _load_directory(self, path: str, pattern: str):
        dir_loader = DirectoryLoader(
            path=path,
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False
        )
        return dir_loader.load()

    def _chunk_text(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=1200,
            chunk_overlap=150,
            length_function=len
        )
        return splitter.split_documents(documents)

    def process(self):
        """
        Loads files (single or directory), cleans nothing (only text),
        chunks using RecursiveCharacterTextSplitter and returns documents.
        """
        if not self.file_paths:
            raise RuntimeError("No file or directory loaded. Use load_from() first.")

        # Load content
        if self.mode == "single":
            docs = self._load_single(self.file_paths)

        elif self.mode == "directory":
            folder, pattern = self.file_paths
            docs = self._load_directory(folder, pattern)

        else:
            raise RuntimeError("Invalid mode state. Something went wrong.")

        # Chunk the loaded documents
        return self._chunk_text(docs)
