import os
import json

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama

import chromadb
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

langchain.debug = True

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



# TODO : you need to change following line to your local path
# hsio : memory and high speed io
# mm : multimedia
# ec : ec
# typeC : typeC
DOC_ROOT = f"..\\..\\article\\"
VS_ROOT = f"..\\..\\persistent\\"
MODEL_ROOT = f"..\\..\\model\\"
collection_name = "platform"
#collection_name = "hsio"

persist_directory = os.path.abspath(VS_ROOT + collection_name + "\\chroma")
local_store = os.path.abspath(VS_ROOT + collection_name + "\\docstore")
article_directory = os.path.abspath(DOC_ROOT + collection_name)
model_directory = os.path.abspath(MODEL_ROOT)


from config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS
from ov_embedding_model import OVEmbeddings
from langchain_community.embeddings import OpenVINOEmbeddings


"""
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



from langchain_community.llms import HuggingFacePipeline
from ov_llm_model import model_classes

try:
    llm_model_id = os.getenv('LLM_MODEL_ID')
    llm_model_configuration = SUPPORTED_LLM_MODELS['English'][llm_model_id]

    llm_device_id = os.getenv('LLM_DEVICE_ID')
    
    model_to_run = os.getenv('LLM_MODEL_COMPRESSION')
    #fp16_model_dir = Path(llm_model_id) / "FP16"
    #int8_model_dir = Path(llm_model_id) / "INT8_compressed_weights"
    #int4_model_dir = Path(llm_model_id) / "INT4_compressed_weights"
    fp16_model_dir = Path(model_directory + '\\' + llm_model_id + '\\' + "FP16")
    int8_model_dir = Path(model_directory + '\\' + llm_model_id + '\\' + "INT8_compressed_weights")
    int4_model_dir = Path(model_directory + '\\' + llm_model_id + '\\' + "INT4_compressed_weights")

    if model_to_run == "INT4-model":
        model_dir = int4_model_dir
    elif model_to_run == "INT8-model":
        model_dir = int8_model_dir
    else:
        model_dir = fp16_model_dir
    print(f"Loading model from {model_dir}")

    model_cache_dir = os.path.join(model_dir, "cache")

    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer, pipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM

    #ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    ov_config = {
        "PERFORMANCE_HINT": "LATENCY", 
        "NUM_STREAMS": "1", 
        "CACHE_DIR": "",
    }

    ov_llm = HuggingFacePipeline.from_model_id(
        model_id=Path(model_dir).absolute().as_posix(),
        task="text-generation",
        backend="openvino",
        model_kwargs={"device": llm_device_id, "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 1000},
    )


    from langchain.prompts import PromptTemplate

    template = """{question}

    Let's think fast and summarize within three sentences."""
    prompt = PromptTemplate.from_template(template)

    #chain = prompt | ov_llm
    chain = LLMChain(llm=ov_llm, prompt=prompt)

    question = "what is earth?"

    print(chain.invoke({"question": question}))
    exit(0)

    core = ov.Core()

    # On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
    # issues caused by this, which we avoid by setting precision hint to "f32".
    if llm_model_id == "red-pajama-3b-chat" and "GPU" in core.available_devices and llm_device_id in ["GPU", "AUTO"]:
        ov_config["INFERENCE_PRECISION_HINT"] = "f32"

    model_name = llm_model_configuration["model_id"]
    stop_tokens = llm_model_configuration.get("stop_tokens")
    class_key = llm_model_id.split("-")[0]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    class StopOnTokens(StoppingCriteria):
        def __init__(self, token_ids):
            self.token_ids = token_ids

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            for stop_id in self.token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    if stop_tokens is not None:
        if isinstance(stop_tokens[0], str):
            stop_tokens = tok.convert_tokens_to_ids(stop_tokens)

        stop_tokens = [StopOnTokens(stop_tokens)]

    model_class = (
        OVModelForCausalLM
        if not llm_model_configuration["remote"]
        else model_classes[class_key]
    )
    ov_model = model_class.from_pretrained(
        model_dir,
        device=llm_device_id,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )

except Exception as e:
    print(e)
    exit(0)


def Vectorstore_backed_retriever(
vectorstore,search_type="similarity",k=4,score_threshold=None
):
    """create a vectorsore-backed retriever
    ref. https://medium.com/thedeephub/rag-chatbot-powered-by-langchain-openai-google-generative-ai-and-hugging-face-apis-6a9b9d7d59db
    Parameters: 
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4) 
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs={}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever



try:
    if not os.path.exists(persist_directory):
        print("embedded not found. Please run 06_load_embedding.py first.")
        exit(0)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)

    # create embeddings for all the documents
    #embedding_model = embedding
    #embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002")
    embedding_model = ov_embedding

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )

    # https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever
    local_store = LocalFileStore(local_store)
    store = create_kv_docstore(local_store)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type="mmr",
        k=10,
    )

    #retriever = Vectorstore_backed_retriever(vectorstore, search_type="similarity",k=8)
    #print(vectorstore.similarity_search("what are the Sanitize Command Restrictions"))

    print("total", vectorstore._collection.count(), "in the collection")


except Exception as e:
    print(e)
    exit(0)

try:

    """if os.getenv('USE_OLLAMA') == 'True':
        llm = ChatOllama(model="llama2")
    elif os.getenv('USE_OPENAI') == 'True':
        llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-0125-preview")
    else:
        print("ERROR: please choose LLM")
    """
    
    """
        temperature:  parameter for control the level of creativity in AI-generated text.
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus
      top_k:  Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      conversation_id: unique conversation identifier.
    """
    temperature = 0.0  # 0.0 means deterministic, 1.0 means maximum diversity
    top_p = 0.9  # 0.0 means no restrictions, 1.0 means only one token is considered
    top_k = 50  # 0 means no restrictions, 200 maximum
    repetition_penalty = 1.1  # 1.0 means no penalty, 2.0 means tokens are halved in probability
   
    generate_kwargs = dict(
        model=ov_model,
        tokenizer=tok,
        max_new_tokens=1024,
        temperature=temperature,
        do_sample=temperature > 0.0,
        #top_p=top_p,
        #top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    if stop_tokens is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

    pipe = pipeline("text-generation", **generate_kwargs)
    llm = HuggingFacePipeline(pipeline=pipe)

    #llm = ChatOllama(model="llama2")
    #llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-0125-preview")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    condense_question_prompt = """Given the following conversation and a follow up question, \
        rephrase the follow up question to be a standalone question, in its original language.\
        Make sure to avoid using any unclear pronouns.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    condense_question_prompt = PromptTemplate.from_template(condense_question_prompt)
    condense_question_chain = LLMChain(
        llm=llm,
        prompt=condense_question_prompt,
    )

    # Create a question answering chain that returns an answer with sources.
    qa_chain = create_qa_with_sources_chain(llm)

    doc_prompt = PromptTemplate(
        template= (
            "Please answer in a structured, clear, coherent, and comprehensive manner.\n"
            "If the content has sections, please summarize them in order and present them easy to read format.\n"
            "if you are using table or figure number then make sure to add reference at the footnote.\n\n"
            "please mention reference sources.\n\n"
            "Content: {page_content}\nSource: {source}"
        ),
        input_variables=["page_content"],
    )


    # create final_qa_chain with map reduce chain
    
    # StuffDocumentsChain is a chain that takes a list of documents and first combines them into a single string. 
    # It does this by formatting each document into a string with the document_prompt and then joining them together with document_separator. 
    # It then adds that new string to the inputs with the variable name set by document_variable_name. Those inputs are then passed to the llm_chain.
    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name="context",
        document_prompt=doc_prompt,
    )

    # This chain combines documents by iterative reducing them. 
    # It groups documents into chunks (less than some context length) then passes them into an LLM. 
    # It then takes the responses and continues to do this until it can fit everything into one final LLM call.
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=final_qa_chain,
    )

    
    retrieval_prompt = PromptTemplate.from_template(llm_model_configuration["rag_prompt_template"])
    
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        chain_type_kwargs={"prompt": retrieval_prompt},
    )
    ai_response = retrieval_qa.invoke({"query": "explain theory of NVMe operation"})
    print(ai_response)
    
    
    """
    from langchain import hub
    from langchain_core.runnables import RunnablePassthrough, RunnablePick

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_prompt = hub.pull("rlm/rag-prompt")

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    qa_chain.invoke("what are the Sanitize Command Restrictions")
    """

    """
    retrieval_qa = ConversationalRetrievalChain(
        question_generator=condense_question_chain,
        retriever=retriever,
        memory=memory,
        #combine_docs_chain=reduce_documents_chain,
        combine_docs_chain=final_qa_chain,
    )
    #ai_response = json.loads(retrieval_qa.invoke({"question": "PCIe packet efficiency based on the MPS size"})["answer"])
    ai_response = json.loads(retrieval_qa.invoke({"question": "explain theory of NVMe operation"})["answer"])
    
    print(ai_response)
    """

except Exception as e:
    print(e)
    exit(0)