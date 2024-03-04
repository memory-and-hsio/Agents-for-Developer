# ref. https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjZmOTc3N2E2ODU5MDc3OThlZjc5NDA2MmMwMGI2NWQ2NmMyNDBiMWIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDE4ODg3NjcxMzY3MTIxMzExMDUiLCJlbWFpbCI6ImhveW91bmcuaHVyM0BnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmJmIjoxNzA5NDg1NzkwLCJuYW1lIjoiaG95b3VuZyBodXIiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSU56UGRnRzkwbjRaRVI0X1k5Sk1UZ0NOZU9pNGZjd1BZSWJTS1lWZ25JUDE0dj1zOTYtYyIsImdpdmVuX25hbWUiOiJob3lvdW5nIiwiZmFtaWx5X25hbWUiOiJodXIiLCJsb2NhbGUiOiJlbiIsImlhdCI6MTcwOTQ4NjA5MCwiZXhwIjoxNzA5NDg5NjkwLCJqdGkiOiIzY2MzZjdjMzU1Nzg0NDRmMjhkN2YwMjlmMDVkMzQ3Yzc0ZmM1NDY1In0.B_lzol8gE1amYaZ0S-vhPwQaSWaBvji6Nc0xhPn1fMNXaHuckxFjotsgnmI5mKtJEA790IoWCU7uz1gzkL6Z2q7tviu6dksqttPg3nQPllgRl6xOyRBsjPirucPJFYNcY-VZZt0yRWh4aiWgzLtuFa9RnInQ1E6z89vsmaTnifAwkeMyFLWJ8QXnPzcpigBBS1wOVJY3pce1YiXWREQHq50_bMbxRWr-ilBTI1v0K9Sjy0UIgrRXUmdTrzVw6H5lm_mALUirnIylzm-u9nuQwnendgx2vz3-ovk61W7fqUFey2SsUhqgeYFZABl_7nBzuREPhcc5XHk6yEWmUqg7wg

import os
import json

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.xml import UnstructuredXMLLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

# define a dictionary to map file extensions to document loaders
document_loaders = {
    "txt": TextLoader,
    "pdf": PyMuPDFLoader,
    "xml": UnstructuredXMLLoader,
    "md": UnstructuredMarkdownLoader,
    "csv": CSVLoader
}
# define function to create DirectroyLoader for file type
def create_directory_loader(file_type, path):
    loader_cls = document_loaders.get(file_type)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return DirectoryLoader(path, glob=f"*.{file_type}", loader_cls=loader_cls, show_progress=True)

if __name__ == "__main__":

    #load OPENAI API key
    load_dotenv()
    if os.path.exists("./chroma"):
        print("already embedded")
        exit(0)


    # create characterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)

    all_documents = []

    # loop each file type and create DirectoryLoader
    for file_type in document_loaders.keys():
        loader = create_directory_loader(file_type, f"../../article")
        raw_documents = loader.load()
        documents = text_splitter.split_documents(raw_documents)
        all_documents.extend(documents)

    # create embeddings for all the documents
    # https://python.langchain.com/docs/integrations/text_embedding/openai
    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002")
    db = Chroma.from_documents(all_documents, embedding_model, persist_directory="./chroma")
    db.persist()

    print("chroma db created")

