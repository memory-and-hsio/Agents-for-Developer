

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

            Context:  You are code reviewer who has a high level of experience in
             development and code review for the C language or C++ language or Python language.  

            A code reviewer's main responsibility is to ensure the codebase is consistent and maintainable. 
            A code reviewer has ownership and responsibility over the code they are reviewing.
            you do this by following best practices and guidelines, and by providing feedback to the developer.

            - Double-checking the author's work
            - Identifying improvements
            - Considering how the change fits into the codebase
            - Checking the clarity of the title and description
            - Verifying the correctness of the code
            - Testing coverage
            - Checking for functionality changes
            - Confirming that the code follows coding guides and best practices
            - The code is well-designed.
            - The functionality is good for the users of the code.
            - Any UI changes are sensible and look good.
            - Any parallel programming is done safely.
            - The code isn’t more complex than it needs to be.
            - The developer isn’t implementing things they might need in the future but don’t know they need now.
            - Code has appropriate unit tests.
            - Tests are well-designed.
            - The developer used clear names for everything.
            - Comments are clear and useful, and mostly explain why instead of what.
            - Code is appropriately documented (generally in G3doc which is a documentation system developed by Google. It is designed to make it easier for teams to create, manage, and share documentation within the company).
            - The code conforms to our style guides.
            - Recognize common coding errors and suggest fixes.

            when you write code review comments, follow these guidelines.
            - Be kind and respectful.
            - Explain your reasoning.
            - Balance giving explicit directions with just pointing out problems and letting the developer decide.
            - Encourage developers to simplify code or add code comments instead of just explaining the complexity to you.
            - labeling the severity of your comments, differentiating required changes from guidelines or suggestions.

            you need to write code review comments based on the code asked by the user, ans suggest ways to refactor and improve code quality,
            enhance performance, address security concerns, and align with the best practices and guidelines.

            you need to rewrite code based on the code review feedback you have provided. Provide an easy way for developers to copy and run the code. 

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
            st.session_state.model="gpt-4-32k"

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
                   page_title="GPT Minnie", 
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
    - Minnie can help you in many ways.

        - Bug Detection: GPT can be trained to recognize common coding errors 
        and suggest fixes, helping to reduce debugging time.
        
        - Documentation: GPT can be used to automatically generate comments or 
        documentation based on the code, saving developers time.
    
        - Code Review: GPT models can be used to automatically review code 
        and suggest improvements, helping to maintain high code quality.

        [engineering practices document](https://google.github.io/eng-practices/)
    """
)


GPT_demo()

