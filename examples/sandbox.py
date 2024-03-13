from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
#llm = Ollama(model="llama2")
llm = ChatOllama(model="llama2", verbose=True)

output_parser = StrOutputParser()


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are technical documentation writer."), 
    ("user", "{input}")
])

#print(prompt.format_messages(input = "what is your job?"))

chain = prompt | llm | output_parser

print(chain.invoke({"input": "what three plus three?"}))
