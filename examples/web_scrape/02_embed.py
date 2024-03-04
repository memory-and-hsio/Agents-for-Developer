# https://github.com/jasonrobwebster/langchain-webscraper-demo/blob/master/embed.py
#
# pip install chroma langchain
# if you see compatibility issues, try: pip3 install langchain
#
# if you see encoding error. please set PYTHONUTF8=1 in your environment variables
# ref. https://dev.to/methane/python-use-utf-8-mode-on-windows-212i

import os
import json

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    if os.path.exists("./chroma"):
        print("already embedded")
        exit(0)

    loader = DirectoryLoader(
        "./scrape",
        glob="*.html",
        loader_cls=BSHTMLLoader,
        show_progress=False,
        loader_kwargs={"get_text_separator": " ", "open_encoding": "utf-8"},
        silent_errors=True,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=200,
    )
    data = loader.load()
    documents = text_splitter.split_documents(data)

    # map sources from file directory to web source
    with open("./scrape/sitemap.json", "r") as f:
        sitemap = json.loads(f.read())

    for document in documents:
        document.metadata["source"] = sitemap[
            document.metadata["source"].replace(".html", "").replace("scrape\\", "")
        ]
        #print(document.metadata["source"])

    import os
    os.environ["OPENAI_API_KEY"] = "your key"

    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002")
    db = Chroma.from_documents(documents, embedding_model, persist_directory="./chroma")
    db.persist()
    db.print_stats()
