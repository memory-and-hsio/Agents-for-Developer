
from pathlib import Path

import streamlit as st

import os
import time
import json
import openai
from openai import OpenAI
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.chains import LLMMathChain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent, Tool, create_react_agent
from langchain.agents import AgentOutputParser

from langchain import hub

from langchain_community.embeddings import OpenVINOEmbeddings


import chromadb

import langchain

from dotenv import load_dotenv



# TODO : you need to change following line to your local path
# hsio : memory and high speed io
# mm : multimedia
# ec : ec
# typeC : typeC
DOC_ROOT = f".\\article\\"
VS_ROOT = f".\\persistent\\"
MODEL_ROOT = f".\\model\\"

#collection_name = "hsio"
collection_name = "platform"


persist_directory = os.path.abspath(VS_ROOT + collection_name + "\\chroma")
local_store = os.path.abspath(VS_ROOT + collection_name + "\\docstore")
article_directory = os.path.abspath(DOC_ROOT + collection_name)
model_directory = os.path.abspath(MODEL_ROOT)

output_parser = StrOutputParser()

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



def GPT_demo(retriever, ov_llm):

    try:
        
        """
        # initialize model
        # https://platform.openai.com/docs/models
        if "model" not in st.session_state:
            #st.session_state.model = "gpt-3.5-turbo"
            #st.session_state.model="gpt-4"
            #st.session_state.model="gpt-4-32k"
            st.session_state.model="gpt-4-turbo-preview"


        llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model=st.session_state.model)
        """
        llm = ov_llm


        retrieval_prompt = PromptTemplate.from_template(llm_model_configuration["rag_prompt_template"])
        
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": retrieval_prompt},
        )
        #ai_response = retrieval_qa.invoke({"query": "explain theory of NVMe operation"})
        #print(ai_response)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


        # check for messages in session and create if not exists
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", 
                 "content": "Hi, I am Jane, an local expert on platform. feel free to ask questions."}
            ]

        # display all messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # user input
        if user_prompt := st.chat_input("Prompt"):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Loading..."):
                    ai_response = retrieval_qa.invoke({"query": user_prompt})['result']

                    # Define prompt
                    last_prompt_template = """You are an assistant. please rewrite content in a structured, clear, coherent, and comprehensive manner.
                    If the content has sections, please summarize them in order and present them easy to read format.
                    If the content has program code, please provide an easy way for developers to copy and run the code.
                    If the content has table or figure number then make sure to add reference at the footnote.
                    
                    Content: {page_content}
                    """
                    #last_prompt = PromptTemplate.from_template(last_prompt_template)

                    # Define LLM chain
                    #last_llm_chain = LLMChain(llm=llm, prompt=last_prompt)

                    #ai_response = last_llm_chain.predict(page_content=ai_response)

                    #ai_response = agent_executor.invoke({"input": user_prompt})['output']
                    st.markdown(ai_response)
            new_ai_message = {"role": "assistant", "content": ai_response}
            st.session_state.messages.append(new_ai_message)

    except Exception as e:
       st.error(
            """
            **got exception.**
            error: %s
            """
            % e
        )
       st.stop()


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


# Langchain Embedding
try:    

    SUPPORTED_EMBEDDING_MODELS = {
        # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        "all-mpnet-base-v2": {
            "model_id": "sentence-transformers/all-mpnet-base-v2",
            "do_norm": True,
        },
    }
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

try:

    DEFAULT_SYSTEM_PROMPT = """You are an assistant for question-answering task. 
        If the content has sections, please summarize them in order and present them easy to read format.
        if you are using table or figure number then make sure to add reference at the footnote.
        Please make sure to provide the answer in a structured, clear, coherent, and comprehensive manner.
    """

    DEFAULT_RAG_PROMPT = """You are an assistant for question-answering task. 
        Use the retrieved context to answer the question.
        If the content has sections, please summarize them in order and present them easy to read format.
        if you are using table or figure number then make sure to add reference at the footnote.
        please mention reference sources.
        Please make sure to provide the answer in a structured, clear, coherent, and comprehensive manner.
    """

    def llama_partial_text_processor(partial_text, new_text):
        new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
        partial_text += new_text
        return partial_text

    SUPPORTED_LLM_MODELS = {
        "English":{
            "llama-2-chat-7b": {
                # https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
                "model_id": "meta-llama/Llama-2-7b-chat-hf",
                "remote": False,
                "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
                "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
                "current_message_template": "{user} [/INST]{assistant}",
                "tokenizer_kwargs": {"add_special_tokens": False},
                "partial_text_processor": llama_partial_text_processor,
                "rag_prompt_template": f"""[INST]Human: <<SYS>> {DEFAULT_RAG_PROMPT }<</SYS>>"""
                + """
                Question: {question} 
                Context: {context} 
                Answer: [/INST]""",
            },
            "mistral-7b": {
                # https://huggingface.co/mistralai/Mistral-7B-v0.1
                "model_id": "mistralai/Mistral-7B-v0.1",
                "remote": False,
                "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
                "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
                "current_message_template": "{user} [/INST]{assistant}",
                "tokenizer_kwargs": {"add_special_tokens": False},
                "partial_text_processor": llama_partial_text_processor,
                "rag_prompt_template": f"""<s> [INST] {DEFAULT_RAG_PROMPT } [/INST] </s>"""
                + """ 
                [INST] Question: {question} 
                Context: {context} 
                Answer: [/INST]""",
            },
        },

    }
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

    
except Exception as e:
    print(e)
    exit(0)


# get icon from https://emojipedia.org/robot
st.set_page_config(
                   page_title="GPT Jane - Local Expert", 
                   page_icon="🙄",
                   layout="wide")
# st.markdown("# GPT Demo")


image_arrow = st.sidebar.image("Gifs/blue_grey_arrow.gif",)



# st.sidebar.header("GPT Demo")
st.markdown(
    """
        Jane is an expert on platform. feel free to ask questions.

        - platform specification
        - IP HAS, SAS, SWAS
        
        """
)

if not os.path.exists(persist_directory):
    st.write(persist_directory)
    st.write("ERROR: embedded database not found. Please run embedding first.")
else:
    st.write("INFO: embedded database found.")

    
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)

# create embeddings for all the documents
# https://python.langchain.com/docs/integrations/text_embedding/openai
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
    #search_type="mmr",
    search_type="similarity",
    k=1,
)
#retriever = Vectorstore_backed_retriever(vectorstore, search_type="similarity",k=8)
#retriever = Vectorstore_backed_retriever(vectorstore, search_type="mmr",k=8)

st.write("INFO : total", vectorstore._collection.count(), "in the collection")


GPT_demo(retriever, ov_llm)

