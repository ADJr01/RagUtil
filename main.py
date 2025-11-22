from file_parsers.doc_parser import DocParser
def main():
   doc_parser = DocParser()
   doc_parser.load_from("C:\\Users\\STHEP\\Documents\\printing.docx")
   data = doc_parser.process()
   for page in data:
       print(page.page_content,end='\n\n')




if __name__ == "__main__":
    main()
