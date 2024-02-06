# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="GPT Agents",
        page_icon="😀",
    )

    st.write("# Welcome to GPT Agents! 😀")

    st.sidebar.success("Select a Agent above.")
    image_arrow = st.sidebar.image("Gifs/blue_grey_arrow.gif",)

    # markdown syntax: https://www.markdownguide.org/basic-syntax/
    st.markdown(
        """
        - GPT can be used to improve developer productivity in several ways.

            - Code Completion: GPT can be trained to suggest code completions, 
            making coding faster and easier. This can be particularly useful for
            repetitive tasks or when working with a new language or library.
	    
            - Bug Detection: GPT can be trained to recognize common coding errors 
            and suggest fixes, helping to reduce debugging time.
	        
            - Code Generation: GPT can be used to generate code snippets based on 
            natural language descriptions, which can help in quickly prototyping 
            or building out functionalities.
	    
            - Documentation: GPT can be used to automatically generate comments or 
            documentation based on the code, saving developers time.
	    
            - Code Review: GPT models can be used to automatically review code 
            and suggest improvements, helping to maintain high code quality.

            - Learning New Technologies: GPT can be used to provide explanations 
            and examples of new technologies, helping developers to learn quickly.
	    
            - Automating Tasks: GPT can be used to automate routine tasks, 
            such as generating boilerplate code or setting up project structures.

            (date : 2024-02-05.  rev 0.3)

    """
    )



if __name__ == "__main__":
    run()
