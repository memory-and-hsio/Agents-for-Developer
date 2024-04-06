

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
collection_name = "windows"


persist_directory = os.path.abspath(VS_ROOT + collection_name + "\\chroma")
local_store = os.path.abspath(VS_ROOT + collection_name + "\\docstore")
article_directory = os.path.abspath(DOC_ROOT + collection_name)


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



def GPT_demo(retriever):

    try:
        
        # initialize model
        # https://platform.openai.com/docs/models
        if "model" not in st.session_state:
            #st.session_state.model = "gpt-3.5-turbo"
            #st.session_state.model="gpt-4"
            #st.session_state.model="gpt-4-32k"
            st.session_state.model="gpt-4-turbo-preview"


        llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model=st.session_state.model)

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
                "if you are using table or figure number then please add reference source.\n\n"
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

        # ConversationalRetrievalChain is a chain that combines the question generator, retriever, memory, and combine_docs_chain
        retrieval_qa = ConversationalRetrievalChain(
            question_generator=condense_question_chain,
            retriever=retriever,
            memory=memory,
            combine_docs_chain=reduce_documents_chain,
        )

        # initialize the LLM tool
        retrieval_tool = Tool(
            name='OS, windows, kernel, system, device driver programming Expert',
            func=retrieval_qa.invoke,
            description='Use this tool for windows OS, system programming, device driver, filter driver queries'
        )


        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

        calculator_tool = Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="A tool that uses the LLM to perform math calculations. Input should be a math question."
        )

        tavily_tool = TavilySearchResults()
        
        tools = [retrieval_tool, tavily_tool, calculator_tool]


        instructions = """You are an assistant.
        Please answer in a structured, clear, coherent, and comprehensive manner.
        If the content has sections, please summarize them in order and present them easy to read format.
        if you are using table or figure number then make sure to add reference at the footnote.
        please mention reference sources.
        Please make sure to provide the answer in a structured, clear, coherent, and comprehensive manner.
        """

        react_base_prompt = hub.pull("hwchase17/react")
        react_prompt = react_base_prompt.partial(instructions=instructions)

        agent_executor = AgentExecutor(
            agent=create_react_agent(llm, tools, react_prompt), 
            tools=tools, 
            verbose=False,
            max_iterations = 10,
            memory=memory,
        )



        # check for messages in session and create if not exists
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", 
                 "content": "Hi, I am Winnie, an expert on windows OS and system programming. feel free to ask questions."}
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
                    ai_response = agent_executor.invoke({"input": user_prompt})['output']

                    # Define prompt
                    last_prompt_template = """You are an assistant. please rewrite content in a structured, clear, coherent, and comprehensive manner.
                    If the content has sections, please summarize them in order and present them easy to read format.
                    If the content has program code, please provide an easy way for developers to copy and run the code.
                    If the content has table or figure number then make sure to add reference at the footnote.
                    
                    Content: {page_content}
                    """
                    last_prompt = PromptTemplate.from_template(last_prompt_template)

                    # Define LLM chain
                    last_llm_chain = LLMChain(llm=llm, prompt=last_prompt)

                    ai_response = last_llm_chain.predict(page_content=ai_response)

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


load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# get icon from https://emojipedia.org/robot
st.set_page_config(
                   page_title="GPT Winnie", 
                   page_icon="🙄",
                   layout="wide")
# st.markdown("# GPT Demo")

# let the use enter key
key_choice = st.sidebar.radio(
    "OpenAI API Key",
    (
        "Your Key",
        "Free Key (capped)",
    ),
    horizontal=True,
)

if key_choice == "Your Key":
    openai.api_key = st.sidebar.text_input( "First, enter your OpenAI API key", type="password")
elif key_choice == "Free Key (capped)":
    openai.api_key = os.environ.get("OPENAI_API_KEY")

image_arrow = st.sidebar.image("Gifs/blue_grey_arrow.gif",)

if key_choice == "Free Key (capped)":
    image_arrow.empty()
else:
    st.write("")
    st.sidebar.caption(
        "No OpenAI API key? Get yours [here!](https://openai.com/blog/api-no-waitlist/)"
    )
    pass
st.write("")



# st.sidebar.header("GPT Demo")
st.markdown(
    """
        Winnie is an expert on windows OS and system programming.  feel free to ask questions.

        - windows OS
        - device driver
        - system programming
        
        """
)

if not os.path.exists(persist_directory):
    st.write(persist_directory)
    st.write("ERROR: embedded database not found. Please run embedding first.")
else:
    st.write("INFO: embedded database found.")

    
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

# create embeddings for all the documents
# https://python.langchain.com/docs/integrations/text_embedding/openai
embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002")
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
    k=8,
)
#retriever = Vectorstore_backed_retriever(vectorstore, search_type="similarity",k=8)
#retriever = Vectorstore_backed_retriever(vectorstore, search_type="mmr",k=8)

st.write("INFO : total", vectorstore._collection.count(), "in the collection")

GPT_demo(retriever)

