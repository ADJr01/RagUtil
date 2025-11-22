from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
import os


class DocParser:
    def __init__(self, simple_mode: bool=True):
        """
        simple_mode = True  -> use Docx2txtLoader
        simple_mode = False -> use UnstructuredWordDocumentLoader
        """
        self.simple_mode = simple_mode
        self.file_path = None

        # Decide which loader to use
        self.loader_cls = Docx2txtLoader if self.simple_mode else UnstructuredWordDocumentLoader

    def load_from(self, path: str):
        """
        Checks if file exists and is .doc or .docx
        Saves file_path for later processing.
        """
        if not isinstance(path, str):
            raise ValueError("path must be a string")

        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist")

        ext = os.path.splitext(path)[1].lower()
        if ext not in [".docx", ".doc"]:
            raise ValueError("File must be a .doc or .docx file")

        self.file_path = path

    def _clean_text(self, text: str) -> str:
        """
        Fast and standard cleaning for unstructured mode.
        Removes multi spaces, line noise, and keeps structure stable.
        """
        cleaned = (
            text.replace("\t", " ")
                .replace("\xa0", " ")
        )
        # remove duplicate spaces
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")

        return cleaned.strip()

    def process(self):
        """
        Process the previously saved file using selected loader.
        Returns chunked documents with metadata.
        """
        if not self.file_path:
            raise RuntimeError("No file loaded. Use load_from() first.")

        loader = self.loader_cls(self.file_path)
        docs = loader.load()

        # If simple mode → return basic chunks
        if self.simple_mode:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )
            return splitter.split_documents(docs)

        # If NOT simple mode → deeper parsing + cleaning
        processed_docs = []
        for d in docs:
            cleaned_text = self._clean_text(d.page_content)

            # Reassign cleaned text, preserve metadata
            d.page_content = cleaned_text
            processed_docs.append(d)

        # Now split structured docs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1400,
            chunk_overlap=200,
            separators=["\n## ", "\n# ", "\n\n", "\n", " ",""]
        )

        return splitter.split_documents(processed_docs)
