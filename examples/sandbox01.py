from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama


from huggingface_hub import notebook_login, whoami

try:
    whoami()
    print('Authorization token already provided')
except OSError:
    notebook_login()

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

ov_llm = HuggingFacePipeline.from_model_id(
    #model_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_id="openchat/openchat-3.5-0106",
    task="text-generation",
    backend="openvino",
    model_kwargs={"device": "GPU", "ov_config": ov_config},
    pipeline_kwargs={"max_new_tokens": 10},
)

from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | ov_llm

question = "i can fly and i can swim, what am i?"

from pprint import pprint

pprint(chain.invoke({"question": question}))


exit()

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
