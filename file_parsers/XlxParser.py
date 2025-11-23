import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)



class XlxParser:
    def __init__(self):
        self.file_path = None

    def load_from(self, path: str):
        """
        Check Excel file validity and save path.
        """
        if not isinstance(path, str):
            raise ValueError("path must be a string")

        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist")

        ext = os.path.splitext(path)[1].lower()
        if ext not in [".xlsx", ".xls"]:
            raise ValueError("File must be an .xlsx or .xls file")

        self.file_path = path

    def process(self):
        """
        Load Excel → convert rows → Documents → chunk → return.
        """
        if not self.file_path:
            raise RuntimeError("No file loaded. Use load_from() first.")

        df = pd.read_excel(self.file_path)

        docs = []
        for idx, row in df.iterrows():
            text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            docs.append(
                Document(
                    page_content=text,
                    metadata={"row": idx, "source": self.file_path}
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        return splitter.split_documents(docs)
