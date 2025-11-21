import os
from logging import exception
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)



from file_parsers.helper.parsing_helper import (
    clean_pdf_text,
    is_pdf_file
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader
)


class PDFParser:
    def __init__(self, pdf_path: str = None, pdf_parser=PyMuPDFLoader):
        self.selected_pdf_path = None
        self.parser = pdf_parser
        self.text_parser = RecursiveCharacterTextSplitter
        if pdf_path is not None and is_pdf_file(pdf_path):
            self.selected_pdf_path = pdf_path

    def select_pdf(self, pdf_path: str):
        if not is_pdf_file(pdf_path):
            raise Exception(f"Invalid PDF file: {pdf_path}")
        self.selected_pdf_path = pdf_path

    def process_pdf(self):
        try:
            if self.selected_pdf_path is None or self.parser is None:
                raise Exception(f"No PDF_file selected/No PDF Parser Selected")
            pdf = self.parser(self.selected_pdf_path).load()
            dataset = []
            for page in pdf:
                cleanned_page = clean_pdf_text(page.page_content)
                if len(cleanned_page)<50:
                    continue
                data = Document(
                    page_content=cleanned_page,
                    metadatas=page.metadata
                )

            return dataset
        except exception as e:
            print(e)
            return None
