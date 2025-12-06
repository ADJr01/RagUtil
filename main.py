from file_parsers.Web_Parser import main
def task():
    # Example: change or supply via CLI as desired
    query = "BitCoin Price"
    # You can override proxy list here if you have proxies
    proxy_list = None  # or ['http://IP:PORT', 'http://IP2:PORT2']
    out_texts = main(query)
    for i, txt in enumerate(out_texts, 1):
        print(f"\n----- DOCUMENT {i} (first 400 chars) -----\n{txt[:400]}\n")

if __name__ == "__main__":
    task()
