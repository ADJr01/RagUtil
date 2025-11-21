import os
from logging import exception

from langchain_community.document_loaders import (
     PyPDFLoader,
PyMuPDFLoader,
UnstructuredPDFLoader
)

import re
import unicodedata

# Ligature map used by enterprise NLP pipelines
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

def clean_pdf_text(text: str) -> str:

    # PASS 1: Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # PASS 2: Replace ligatures
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)

    # PASS 3: Remove private-use area characters (font subset garbage)
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # PASS 4: Remove strange font artifacts like {sifd}, {cid1234}, [sfi2], etc.
    text = re.sub(r"\{[^}]{1,20}\}", "", text)
    text = re.sub(r"\[[^\]]{1,20}\]", "", text)

    # PASS 5: Clean control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # PASS 6: Fix hyphenations: inter-\nnational -> international
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # PASS 7: Merge broken lines inside a paragraph
    text = re.sub(r"(?<![\.\?!:])\n(?=[a-zA-Z])", " ", text)

    # PASS 8: Remove page numbers and repeated headers/footers
    header_footer_patterns = [
        r"^\s*Page\s*\d+\s*$",
        r"^\s*\d+\s*$",
        r"^\s*Copyright.*$",
        r"^\s*All rights reserved.*$",
        r"^\s*Chapter\s+\d+.*$",
    ]
    for pat in header_footer_patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE)

    # PASS 9: Remove leftover OCR garbage
    text = re.sub(r"[■◆●◼◾◽▪▫•◊]", "", text)
    text = re.sub(r"[^\w\s.,!?:;()'\"/%\-–—]", "", text)

    # PASS 10: Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # PASS 11: Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # PASS 12: Trim individual lines
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
def is_pdf_file(path):
    return os.path.splitext(path)[1].lower() == ".pdf"

class PDFParser:
    def __init__(self,pdf_path:str=None,pdf_parser=PyMuPDFLoader):
        self.selected_pdf_path = None
        self.parser=pdf_parser
        if pdf_path is not None and is_pdf_file(pdf_path):
            self.selected_pdf_path = pdf_path

    def select_pdf(self,pdf_path:str):
        if not is_pdf_file(pdf_path):
            raise Exception(f"Invalid PDF file: {pdf_path}")
        self.selected_pdf_path = pdf_path

    def load_pdf(self):
        try:
            if self.selected_pdf_path is None or self.parser is None:
                raise Exception(f"No PDF_file selected/No PDF Parser Selected")
            pdf = self.parser(self.selected_pdf_path).load()
            dataset = [{**page, 'page_content': clean_pdf_text(page['page_content'])} for page in pdf]
            return dataset
        except exception as e:
            print(e)
            return None




