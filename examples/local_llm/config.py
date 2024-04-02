DEFAULT_SYSTEM_PROMPT = """You are an assistant for question-answering task. 
    If the content has sections, please summarize them in order and present them easy to read format.
    if you are using table or figure number then make sure to add reference at the footnote.
    please mention reference sources.
    Please make sure to provide the answer in a structured, clear, coherent, and comprehensive manner.
"""

DEFAULT_RAG_PROMPT = """You are an assistant for question-answering task. 
    Use the retrieved context to answer the question.
    If the content has sections, please summarize them in order and present them easy to read format.
    if you are using table or figure number then make sure to add reference at the footnote.
    please mention reference sources.
    Please make sure to provide the answer in a structured, clear, coherent, and comprehensive manner.
"""

def red_pijama_partial_text_processor(partial_text, new_text):
    if new_text == "<":
        return partial_text

    partial_text += new_text
    return partial_text.split("<bot>:")[-1]


def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text


def internlm_partial_text_processor(partial_text, new_text):
    partial_text += new_text
    return partial_text.split("<|im_end|>")[0]


SUPPORTED_LLM_MODELS = {
    "English":{
        "tiny-llama-1b-chat": {
            # https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "remote": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """
            <|user|>
            Question: {question} 
            Context: {context} 
            Answer: </s>
            <|assistant|>""",
        },
        "gemma-2b-it": {
            # https://huggingface.co/google/gemma-2b-it
            "model_id": "google/gemma-2b-it",
            "remote": True,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""+"""<start_of_turn>user{question}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model"""
        },
        "red-pajama-3b-chat": {
            # https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
            "model_id": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            "remote": False,
            "start_message": "",
            "history_template": "\n<human>:{user}\n<bot>:{assistant}",
            "stop_tokens": [29, 0],
            "partial_text_processor": red_pijama_partial_text_processor,
            "current_message_template": "\n<human>:{user}\n<bot>:{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT }"""
            + """
            <human>: Question: {question} 
            Context: {context} 
            Answer: <bot>""",
        },
        "gemma-7b-it": {
            # https://huggingface.co/google/gemma-7b-it
            "model_id": "google/gemma-7b-it",
            "remote": True,
            "start_message": DEFAULT_SYSTEM_PROMPT + ", ",
            "history_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}<end_of_turn>",
            "current_message_template": "<start_of_turn>user{user}<end_of_turn><start_of_turn>model{assistant}",
            "rag_prompt_template": f"""{DEFAULT_RAG_PROMPT},"""+"""<start_of_turn>user{question}<end_of_turn><start_of_turn>context{context}<end_of_turn><start_of_turn>model"""
        },
        "llama-2-chat-7b": {
            # https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "remote": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""[INST]Human: <<SYS>> {DEFAULT_RAG_PROMPT }<</SYS>>"""
            + """
            Question: {question} 
            Context: {context} 
            Answer: [/INST]""",
        },
        "mpt-7b-chat": {
            # https://huggingface.co/mosaicml/mpt-7b-chat
            "model_id": "mosaicml/mpt-7b-chat",
            "remote": True,
            "start_message": f"<|im_start|>system\n {DEFAULT_SYSTEM_PROMPT }<|im_end|>",
            "history_template": "<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}<|im_end|>",
            "current_message_template": '"<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}',
            "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
            "rag_prompt_template": f"""<|im_start|>system 
            {DEFAULT_RAG_PROMPT }<|im_end|>"""
            + """
            <|im_start|>user
            Question: {question} 
            Context: {context} 
            Answer: <im_end><|im_start|>assistant""",
        },
        "mistral-7b": {
            # https://huggingface.co/mistralai/Mistral-7B-v0.1
            "model_id": "mistralai/Mistral-7B-v0.1",
            "remote": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""<s> [INST] {DEFAULT_RAG_PROMPT } [/INST] </s>"""
            + """ 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]""",
        },
        "zephyr-7b-beta": {
            # https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
            "model_id": "HuggingFaceH4/zephyr-7b-beta",
            "remote": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """ 
            <|user|>
            Question: {question} 
            Context: {context} 
            Answer: </s>
            <|assistant|>""",
        },
        "notus-7b-v1": {
            # https://huggingface.co/argilla/notus-7b-v1
            "model_id": "argilla/notus-7b-v1",
            "remote": False,
            "start_message": f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}</s>\n",
            "history_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}</s> \n",
            "current_message_template": "<|user|>\n{user}</s> \n<|assistant|>\n{assistant}",
            "rag_prompt_template": f"""<|system|> {DEFAULT_RAG_PROMPT }</s>"""
            + """
            <|user|>
            Question: {question} 
            Context: {context} 
            Answer: </s>
            <|assistant|>""",
        },
        "neural-chat-7b-v3-1": {
            # https://huggingface.co/Intel/neural-chat-7b-v3-3
            "model_id": "Intel/neural-chat-7b-v3-3",
            "remote": False,
            "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT }\n<</SYS>>\n\n",
            "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
            "current_message_template": "{user} [/INST]{assistant}",
            "tokenizer_kwargs": {"add_special_tokens": False},
            "partial_text_processor": llama_partial_text_processor,
            "rag_prompt_template": f"""<s> [INST] {DEFAULT_RAG_PROMPT } [/INST] </s>"""
            + """
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]""",
        },
    },

}

SUPPORTED_EMBEDDING_MODELS = {
    # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    "all-mpnet-base-v2": {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "do_norm": True,
    },
}