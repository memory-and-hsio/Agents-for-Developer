# ref. https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjZmOTc3N2E2ODU5MDc3OThlZjc5NDA2MmMwMGI2NWQ2NmMyNDBiMWIiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDE4ODg3NjcxMzY3MTIxMzExMDUiLCJlbWFpbCI6ImhveW91bmcuaHVyM0BnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwibmJmIjoxNzA5NDg1NzkwLCJuYW1lIjoiaG95b3VuZyBodXIiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSU56UGRnRzkwbjRaRVI0X1k5Sk1UZ0NOZU9pNGZjd1BZSWJTS1lWZ25JUDE0dj1zOTYtYyIsImdpdmVuX25hbWUiOiJob3lvdW5nIiwiZmFtaWx5X25hbWUiOiJodXIiLCJsb2NhbGUiOiJlbiIsImlhdCI6MTcwOTQ4NjA5MCwiZXhwIjoxNzA5NDg5NjkwLCJqdGkiOiIzY2MzZjdjMzU1Nzg0NDRmMjhkN2YwMjlmMDVkMzQ3Yzc0ZmM1NDY1In0.B_lzol8gE1amYaZ0S-vhPwQaSWaBvji6Nc0xhPn1fMNXaHuckxFjotsgnmI5mKtJEA790IoWCU7uz1gzkL6Z2q7tviu6dksqttPg3nQPllgRl6xOyRBsjPirucPJFYNcY-VZZt0yRWh4aiWgzLtuFa9RnInQ1E6z89vsmaTnifAwkeMyFLWJ8QXnPzcpigBBS1wOVJY3pce1YiXWREQHq50_bMbxRWr-ilBTI1v0K9Sjy0UIgrRXUmdTrzVw6H5lm_mALUirnIylzm-u9nuQwnendgx2vz3-ovk61W7fqUFey2SsUhqgeYFZABl_7nBzuREPhcc5XHk6yEWmUqg7wg

import os
import json

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredXMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

# TODO : you need to change following line to your local path
# hsio : memory and high speed io
# mm : multimedia
# ec : ec
# typeC : typeC
DOC_ROOT = f"..\\..\\article\\"
VS_ROOT = f"..\\..\\persistent\\"
collection_name = "hsio"

persist_directory = os.path.abspath(VS_ROOT + collection_name + "\\chroma")
local_store = os.path.abspath(VS_ROOT + collection_name + "\\docstore")
article_directory = os.path.abspath(DOC_ROOT + collection_name)

# define a dictionary to map file extensions to document loaders
document_loaders = {
    "txt": TextLoader,
    "pdf": PyMuPDFLoader,
    "docx": Docx2txtLoader,
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

def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]

if __name__ == "__main__":

    #load OPENAI API key
    load_dotenv()
    if os.path.exists(persist_directory):
        print("already embedded articles. Please remove the directory first.")
        exit(0)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    all_documents = []

    # loop each file type and create DirectoryLoader
    for file_type in document_loaders.keys():
        loader = create_directory_loader(file_type, article_directory)
        raw_documents = loader.load()
        documents = parent_splitter.split_documents(raw_documents)
        all_documents.extend(documents)

    print(f"total {len(all_documents)} documents")

    # create embeddings for all the documents
    # https://python.langchain.com/docs/integrations/text_embedding/openai
    embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002")

    # add_documents returns error if all_documents is too large.
    # to workaround it, split list and add_documents in chunks
    # ValueError: Batch size 31875 exceeds maximum batch size 5461
    split_doc_chunked = split_list(all_documents, 1000)
    for split_doc in split_doc_chunked:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )

        # https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever
        local_store_instance = LocalFileStore(local_store)
        store_instance = create_kv_docstore(local_store_instance)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store_instance,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        retriever.add_documents(split_doc, ids=None)
        vectorstore.persist()
    #retriever.add_documents(all_documents, ids=None)
    
    print("chroma db created")

    exit(0)

    # create characterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

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

