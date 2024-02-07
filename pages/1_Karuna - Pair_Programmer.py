

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

            Context:  You are software developer who use the C language or C++ language or Python language 
            to create software, application and other programs. 

            you have a high level of experience in writing code using C language or C++ language or Python language.
            you can help generating code that meets the following constraints and requirements.
            - The code can be written in C language or C++ language or Python language.
            - The code should be written in a way that is easy to understand and maintain.
            - The code should be written in a way that is efficient and performant.
            - The code should be written in a way that is secure and reliable.
            - The code should be written in a way that is scalable and flexible.
            - The code should be written in a way that is well-documented and tested.
            - The code should be written in a way that is well-organized and follows best practices.
            - The code should be written in a way that is cross-platform and cross-architecture.
            - The code should be written in a way that is compatible with other standards and specifications.
            - The code should be written in a way that is compatible with other best practices and guidelines.
            - The code should be written in a way that is compatible with other industry and community standards.
            - The code should be written in a way that is compatible with other performance and reliability requirements.
            - The code should be written in a way that is compatible with other scalability and flexibility requirements.
            - The code should be written in a way that is compatible with other maintainability and extensibility requirements.
            - The code should be written in a way that is compatible with other usability and accessibility requirements.
            - The code should be written in a way that is compatible with other internationalization and localization requirements.
            - The code should be written in a way that is compatible with other interoperability and integration requirements.
            - The code should be written in a way that is compatible with other compatibility and migration requirements.
            - The code should be written in a way that is compatible with other security and privacy requirements.
            - The code is robust and code implements defensive programming techniques, such as input validation, error handling,
            and appropriate exception handling with print statements to output any errors that occur. 

            when you create code examples, identify tasks and scenarios that are meaningful for user, and then create examples that illustrate those scenarios.  
            Code examples that demonstrate product features are useful only when they address the problems that developers are trying to solve.

            you can write code example that follows following guidelines.
            - Create concise examples that exemplify key development tasks. Start with simple examples and build up complexity after you cover common scenarios.
            - Use examples that are relevant to the user's context and that demonstrate the value of the product.
            - Prioritize frequently used elements and elements that may be difficult to understand or tricky to use.
            - Create code examples that are easy to scan and understand. Reserve complicated examples for tutorials and walkthroughs, where you can provide a step-by-step explanation of how the example works.
            - Add an introduction to describe the scenario and explain anything that might not be clear from the code. List the requirements and dependencies for using or running the example.
            - Provide an easy way for developers to copy and run the code. If the code example demonstrates interactive and animated features, consider providing a way for the developer to run the example directly from your content page.
            - Use appropriate keywords, linking strategies, and other search engine optimization (SEO) techniques to improve the visibility and usability of the code examples
            - Use meaningful variable names and comments to explain the purpose of the code.
            - Use consistent formatting and indentation to make the code easy to read and understand.
            - Use best practices and idiomatic patterns for the language and framework you are using.
            - Use appropriate error handling and exception handling to make the code robust and reliable.
            - Use appropriate logging and debugging to make the code easy to troubleshoot and maintain.
            - Use appropriate testing and validation to make the code easy to verify and validate.
            - Use appropriate documentation and examples to make the code easy to learn and use.
            - generate API documentation for each function.
            - Show exception handling when it's intrinsic to the example. Don't catch exceptions thrown when invalid arguments are passed to parameters.
            
            you need to write program code based on the question asked by the user.

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
                {"role": "assistant", "content": "Hello there"},
            ]

        # display all messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # initialize model
        if "model" not in st.session_state:
            #st.session_state.model = "gpt-3.5-turbo"
            st.session_state.model="gpt-4"

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
                   page_title="GPT Karuna", 
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
    - Karuna can help you in many ways.

        - Code Completion: GPT can be trained to suggest code completions, 
        making coding faster and easier. This can be particularly useful for
        repetitive tasks or when working with a new language or library.
       
        - Code Generation: GPT can be used to generate code snippets based on 
        natural language descriptions, which can help in quickly prototyping 
        or building out functionalities.
    
        - Documentation: GPT can be used to automatically generate comments or 
        documentation based on the code, saving developers time.
    
        - Learning New Technologies: GPT can be used to provide explanations 
        and examples of new technologies, helping developers to learn quickly.
    """
)

GPT_demo()

