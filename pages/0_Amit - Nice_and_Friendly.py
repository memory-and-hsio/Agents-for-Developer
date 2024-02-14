

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

            Context:  You are currently having a conversation with a human. 
            Answer the questions asked by the user.
            
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
                   page_title="GPT Amit", 
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
st.write(
    """Amit is very nice and friendly. feel free to ask him anything.
    """
)

GPT_demo()

