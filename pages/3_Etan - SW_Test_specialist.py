

import streamlit as st

import os
import time
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory

from dotenv import load_dotenv


def GPT_demo():

    try:
        
        prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="""Answer the question based on the context below. 

            Context:  You are software test specialist who has a high level of experience in
            software quality assurance and software testing for the C language or C++ language or Python language.  

            Software testing encompasses various crucial aspects such as usability, functionality, performance, 
            security, and many more. Thus, rigorous testing protocols enable testers to assess and remove vulnerabilities 
            and shortcomings which may affect the end users.

            As a software test specialist, you are responsible writing test plans and test cases and test code.
            For each test plan, you can write multiple test cases.
            
            you should write test plan based on the code asked by the user that meets the following constraints and requirements.            
                Test plan Introduction section provides an overview of the entire test plan. 
                Test scope section outlines the scope of the testing process.
                Test environment section describes the environment in which the testing will be conducted.
                Test approach section outlines the approach that will be used to conduct the testing.
                Test scenarios and tasks section outlines the scenarios and tasks that will be tested.
                Test data section describes the data that will be used to conduct the testing.
                Test execution section outlines the execution of the testing process.
                Data analysis and reporting section outlines the analysis and reporting of the testing process.
                Test deliverables section outlines the deliverables of the testing process.

            you should write test plan based on the code asked by the user that meets the following constraints and requirements.                
                Test case id section provides a unique identifier for the test case.
                Test case description section provides a description of the test case.
                Test case steps section outlines the steps that will be taken to conduct the test case.
                you should implement test code  to call the function with the provided input for each test step. The code follows industry's style guide and is easy to understand and maintain.
                Test case expected results section outlines the expected results of the test case.
                Test case actual results section outlines the actual results of the test case.
                Test case status section outlines the status of the test case.

            you should write test code based on the code asked by the user that meets the following constraints and requirements.                
                The code should be written in a way that is well-documented and tested.
                The code should be written in a way that is well-organized and follows best practices.
                Test code provides a functional code that will be used to conduct the testing.
                Provide an easy way for developers to copy and run the code. If the code example demonstrates interactive and animated features, consider providing a way for the developer to run the example directly from your content page.

            you need to write test code based on the code asked by the user. Provide an easy way for developers to copy and run the code. 
            
            chat_history: {chat_history}

            Question: {question}

            Answer:"""
        )


        llm = ChatOpenAI(openai_api_key=openai.api_key)
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)
        llm_chain = LLMChain(
            llm=llm,
            memory=memory,
            prompt=prompt
        )

        # check for messages in session and create if not exists
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello there"}
            ]

        # display all messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # initialize model
        # https://platform.openai.com/docs/models
        if "model" not in st.session_state:
            #st.session_state.model = "gpt-3.5-turbo"
            #st.session_state.model="gpt-4"
            #st.session_state.model="gpt-4-32k"
            st.session_state.model="gpt-4-0125-preview"

        # user input
        if user_prompt := st.chat_input("Prompt"):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Loading..."):
                    ai_response = llm_chain.predict(question=user_prompt)
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
                   page_title="GPT Etan", 
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



#st.sidebar.header("GPT Demo")

st.markdown(
    """
    - Etan can help you in many ways.

        - Test Generation: GPT can be used to automatically generate test cases
        
        - Documentation: GPT can be used to automatically generate comments or 
        documentation based on the code, saving developers time.
    """
)


GPT_demo()

