from file_parsers.pdf_parser import PDFParser
def main():
    cpp_book_pdf = "D:\\Training\\Course\\cpp\\cpp_book.pdf"
    ps = PDFParser()
    ps.select_pdf(cpp_book_pdf)
    pdf_data = ps.load_pdf()
    for page in pdf_data:
        print(page.page_content)



if __name__ == "__main__":
    main()
