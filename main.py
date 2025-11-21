from file_parsers.pdf_parser import PDFParser
def main():
    cpp_book_pdf = "D:\\Training\\Course\\cpp\\cpp_book.pdf"
    ps = PDFParser()
    ps.select_pdf(cpp_book_pdf)
    pdf_data = ps.process_pdf()
    for page in pdf_data:
        print(page[0].page_content,end="\n\n")



if __name__ == "__main__":
    main()
