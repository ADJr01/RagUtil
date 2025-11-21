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
    PyMuPDFLoader,
)


class PDFParser:
    def __init__(self, pdf_path: str = None, pdf_parser=PyMuPDFLoader):
        self.selected_pdf_path = None
        self.parser = pdf_parser
        self.text_parser = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators= ["\n\n","\n","."," ",""]
        )
        if pdf_path is not None and is_pdf_file(pdf_path):
            self.selected_pdf_path = pdf_path

    def select_pdf(self, pdf_path: str):
        if not is_pdf_file(pdf_path):
            raise Exception(f"Invalid PDF file: {pdf_path}")
        self.selected_pdf_path = pdf_path

    def process_pdf(self)->list[Document]:
        try:
            if self.selected_pdf_path is None or self.parser is None:
                raise Exception(f"No PDF_file selected/No PDF Parser Selected")
            pdf = self.parser(self.selected_pdf_path).load()
            dataset = []
            total_pages = len(pdf)
            for page_index,page in enumerate(pdf):
                if len(page.page_content)<100:
                    continue
                cleanned_text = clean_pdf_text(page.page_content)
                data = self.text_parser.create_documents(
                    texts=[cleanned_text],
                    metadatas=[
                        {
                            **page.metadata,
                            "book": "cpp_ref_book",
                            "total_pages": total_pages,
                            "page": page_index,
                        }
                    ]
                )
                dataset.append(data)

            return dataset
        except Exception as e:
            print(f"Exception>> {e}")
            return None
