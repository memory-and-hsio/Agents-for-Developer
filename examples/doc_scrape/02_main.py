

import os
import json

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.map_reduce import ReduceDocumentsChain
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


if __name__ == "__main__":

    if not os.path.exists("./chroma"):
        print("embedded not found. Please run 01_load_embed.py first.")
        exit(0)

    #load OPENAI API key
    load_dotenv()

    
    #vectorstore
    vectorstore = Chroma(
        persist_directory="./chroma",
        embedding_function=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-ada-002"),
    )
    retriever = Vectorstore_backed_retriever(vectorstore)

    #print(vectorstore.similarity_search("what is FunnyIO?"))
    #print(vectorstore.max_marginal_relevance_search("what is FunnyIO?", k = 3, fetch_k = 30))

    print("total", vectorstore._collection.count(), "in the collection")

    
    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-4-0125-preview")

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
        template="Content: {page_content}\nSource: {source}",
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

    output_parser = StrOutputParser()

    #invoke question
    out = retrieval_qa.invoke({"question": "what is L0s?"})
    print(out)

    #from pprint import pprint
    #pprint(output_parser.parse(out))

