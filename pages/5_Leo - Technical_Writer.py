

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
            template="""you should to write technical article in detail based on the question asked by the user based on the cotext below.

            Context:  You are technical writer who has a high level of experience in technical communication, create instruction manuals, how-to guides, journal articles, 
             and other documents to communicate complex information. 

            you do this by following best practices and guidelines.

            - You need to be able to take complex ideas and explain them in a way that is easy to understand.
            - You need to create outlines and summaries of your own documents or for giving instructions to the documentation team.
            - You need to be able to use a consistent style throughout your documents.
            - you need to check grammar and spelling and make sure the document is easy to read and navigate.
            - you need to use headings and subheadings to make the document easy to read and navigate.
            - for each section, complete paragraph with detailed explaination.  add example if possible.
            - if you don't have enough context to answer the question, ask for more information.

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
                   page_title="GPT Leo", 
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
        Leo can help you in prepare technical writing.

        - Drafting Content: GPT can generate drafts based on the information provided. It can help in creating the initial structure of the document, which can then be refined by the user.
        - Simplifying Language: GPT can simplify complex language and explain complex ideas in simple language.
        - Consistency: GPT can ensure consistency in the document by maintaining the same tone, style, and language throughout.
        - Formatting: GPT can help in formatting the document according to the desired style guide.
        
        """
)


GPT_demo()

