# https://github.com/jasonrobwebster/langchain-webscraper-demo/blob/master/main.py

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
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

import os
os.environ["OPEN_API_KEY"] = "your key"

#vectorstore
vectorstore = Chroma(
    persist_directory="./chroma",
    embedding_function=OpenAIEmbeddings(openai_api_key=os.environ["OPEN_API_KEY"], model="text-embedding-ada-002"),
)

llm = ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"], temperature=0, model="gpt-4-0125-preview")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

condense_question_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\
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

qa_chain = create_qa_with_sources_chain(llm)

doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

final_qa_chain = StuffDocumentsChain(
    llm_chain=qa_chain,
    document_variable_name="context",
    document_prompt=doc_prompt,
)

# ConversationalRetrievalChain is a chain that combines the question generator, retriever, memory, and combine_docs_chain
retrieval_qa = ConversationalRetrievalChain(
    question_generator=condense_question_chain,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain=final_qa_chain,
)


#out = retrieval_qa.invoke({"question": "what are the latest news from Folson Times?"}) 
out = retrieval_qa.invoke({"question": "what happens to the folsom garden club?"}) 

print(out)
