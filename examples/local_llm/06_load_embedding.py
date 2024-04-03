import os

import json

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredXMLLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

import langchain

from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
import torch
import nncf
import logging
import shutil
import gc
import ipywidgets as widgets
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)

import dotenv

try:
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)
        
except Exception as e:
    print(e)
    print("Please create .env file")
    exit(0)

print(f"Selected LLM model {os.getenv('LLM_MODEL_ID')}")
print(f"Selected LLM Compression model {os.getenv('LLM_MODEL_COMPRESSION')}")
print(f"Selected LLM embedding model {os.getenv('LLM_EMBEDDING')}")
print(f"Selected LLM embedding device {os.getenv('EMBEDDING_DEVICE_ID')}")
print(f"Selected LLM device {os.getenv('LLM_DEVICE_ID')}")

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

from config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS
from ov_embedding_model import OVEmbeddings
from langchain_community.embeddings import OpenVINOEmbeddings

"""
# OVEmbedding is a class that wraps the OpenVINO model for use as an embedding model in Langchain.
try:
    llm_model_id = os.getenv('LLM_MODEL_ID')
    llm_model_configuration = SUPPORTED_LLM_MODELS['English'][llm_model_id]

    embedding_model_id = os.getenv('LLM_EMBEDDING')
    embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[embedding_model_id]

    embedding_device_id = os.getenv('EMBEDDING_DEVICE_ID')
    embedding_model_dir = Path(embedding_model_id)

    llm_device_id = os.getenv('LLM_DEVICE_ID')
    
    model_to_run = os.getenv('LLM_MODEL_COMPRESSION')
    fp16_model_dir = Path(llm_model_id) / "FP16"
    int8_model_dir = Path(llm_model_id) / "INT8_compressed_weights"
    int4_model_dir = Path(llm_model_id) / "INT4_compressed_weights"

    embedding = OVEmbeddings.from_model_id(
        embedding_model_dir,
        do_norm=embedding_model_configuration["do_norm"],
        ov_config={
            "device_name": embedding_device_id,
            "config": {"PERFORMANCE_HINT": "THROUGHPUT"},
        },
        model_kwargs={
            # HHUR. optimize later.  
            # This is the max length of the input sequence that the model can handle.        
            "model_max_length": 512,
            #"model_max_length": 4096,
        },
    )
except Exception as e:
    print(e)
    exit(0)
"""

# Langchain Embedding
try:    
    embedding_model_id = os.getenv('LLM_EMBEDDING')
    embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[embedding_model_id]

    embedding_model_name = embedding_model_configuration["model_id"]
    model_kwargs = {'device': 'CPU'}
    encode_kwargs = {'normalize_embeddings': True}
    ov_embedding = OpenVINOEmbeddings(
        model_name_or_path=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
except Exception as e:
    print(e)
    exit(0)


# TODO : you need to change following line to your local path
# hsio : memory and high speed io
# mm : multimedia
# ec : ec
# typeC : typeC
DOC_ROOT = f"..\\..\\article\\"
VS_ROOT = f"..\\..\\persistent\\"
MODEL_ROOT = f"..\\..\\model\\"

#DOC_ROOT = f"C:\\workspace\\llm\\Agents-for-Developer\\article\\"
#VS_ROOT = f"C:\\workspace\\llm\\Agents-for-Developer\\persistent\\"
#MODEL_ROOT = f"C:\\workspace\\llm\\Agents-for-Developer\\model\\"

#collection_name = "hsio"
collection_name = "platform"

persist_directory = os.path.abspath(VS_ROOT + collection_name + "\\chroma")
local_store = os.path.abspath(VS_ROOT + collection_name + "\\docstore")
article_directory = os.path.abspath(DOC_ROOT + collection_name)
model_directory = os.path.abspath(MODEL_ROOT)

# define a dictionary to map file extensions to document loaders
document_loaders = {
    "txt": TextLoader,
    "pdf": PyMuPDFLoader,
    "docx": UnstructuredWordDocumentLoader,
    "doc": UnstructuredWordDocumentLoader,
    "xml": UnstructuredXMLLoader,
    "md": UnstructuredMarkdownLoader,
    "mmd": UnstructuredMarkdownLoader,
    "csv": CSVLoader,
    "ppt": UnstructuredPowerPointLoader,
    "pptx": UnstructuredPowerPointLoader,
    "html": UnstructuredHTMLLoader,
}

document_loaders = {
    "html": UnstructuredHTMLLoader,
}



langchain.debug = True

try:

    if os.path.exists(persist_directory):
        print(persist_directory)
        print("already embedded articles. Please remove the directory first.  if not it'll be appended")

    # define function to create DirectroyLoader for file type
    def create_directory_loader(file_type, path):
        loader_cls = document_loaders.get(file_type)
        if loader_cls is None:
            raise ValueError(f"Unsupported file type: {file_type}")
        return DirectoryLoader(path, glob=f"**/*.{file_type}", loader_cls=loader_cls, show_progress=True)
        #return DirectoryLoader(path, glob=f"*.{file_type}", loader_cls=loader_cls, show_progress=True)

    def split_list(input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)

    all_documents = []

    # loop each file type and create DirectoryLoader
    for file_type in document_loaders.keys():
        loader = create_directory_loader(file_type, article_directory)
        raw_documents = loader.load()
        documents = parent_splitter.split_documents(raw_documents)
        all_documents.extend(documents)

    print(f"total {len(all_documents)} documents")

    # create embeddings for all the documents
    #embedding_model = embedding
    embedding_model = ov_embedding

    # add_documents returns error if all_documents is too large.
    # to workaround it, split list and add_documents in chunks
    # ValueError: Batch size 31875 exceeds maximum batch size 5461
    split_doc_chunked = split_list(all_documents, 512)
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

        # get medatada from the document and check if it's already embedded
        metadata = split_doc[0].metadata
        print(metadata["source"])
        """
        print(metadata["page"])
        # check source and page from vectorstore
        results = vectorstore.get( where={"$and": [{"source": metadata["source"]}, {"page": metadata["page"]}]}, include=["metadatas"],)
        if len(results["ids"]) > 0:
            print(f"Already embedded {metadata}")
        """

        retriever.add_documents(split_doc, ids=None)
        vectorstore.persist()
    #retriever.add_documents(all_documents, ids=None)
    
    vectorstore = None

    print("chroma db created")

except Exception as e:
    print(e)
    exit(0)
