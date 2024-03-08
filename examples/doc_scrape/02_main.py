

import os
import json

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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

import chromadb

import langchain
#langchain.debug = True

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


if __name__ == "__main__":

    if not os.path.exists(persist_directory):
        print("embedded not found. Please run 01_load_embed.py first.")
        exit(0)

    #load OPENAI API key
    load_dotenv()
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

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

    #print(vectorstore.similarity_search("what is FunnyIO?"))
    #print(vectorstore.max_marginal_relevance_search("what is FunnyIO?", k = 3, fetch_k = 30))

    print("total", vectorstore._collection.count(), "in the collection")

    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], model="gpt-4-0125-preview")


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

    # ConversationalRetrievalChain is a chain that combines the question generator, retriever, memory, and combine_docs_chain
    retrieval_qa = ConversationalRetrievalChain(
        question_generator=condense_question_chain,
        retriever=retriever,
        memory=memory,
        combine_docs_chain=reduce_documents_chain,
        #combine_docs_chain=final_qa_chain,
    )

    output_parser = StrOutputParser()

    #invoke question
    ai_response = json.loads(retrieval_qa.invoke({"question": "PCIe packet efficiency based on the MPS size"})["answer"])
    print(ai_response["answer"])


    #from pprint import pprint
    #pprint(output_parser.parse(out))

