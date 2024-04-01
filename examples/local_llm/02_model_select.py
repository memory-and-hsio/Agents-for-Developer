import os
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov
import torch
import nncf
import logging
import shutil
import gc
import ipywidgets as widgets
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TextIteratorStreamer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)

import dotenv

try:
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)
        
except Exception as e:
    print(e)
    print("Please create .env file")
    exit(0)


from config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS, SUPPORTED_EMBEDDING_MODELS

try:
    llm_model_ids = [model_id for model_id, model_config in SUPPORTED_LLM_MODELS['English'].items() if model_config.get("rag_prompt_template")]
    import inquirer
    questions = [
        inquirer.List("selected_id", message="Select LLM model", choices=llm_model_ids), 
    ]
    answer = inquirer.prompt(questions)

    llm_model_id = answer["selected_id"]
    
    os.environ["LLM_MODEL_ID"] = llm_model_id
    dotenv.set_key(dotenv_file, "LLM_MODEL_ID", llm_model_id)
except Exception as e:
    print(e)
    exit(0)

print(f"Selected LLM model {llm_model_id}")

llm_model_configuration = SUPPORTED_LLM_MODELS['English'][llm_model_id]

try:
    import inquirer
    questions = [
        inquirer.List("selected_id", \
                    message="Select Compression model", \
                    choices=['INT4-model', 'INT8-model', 'FP16-model']),
    ]
    answer = inquirer.prompt(questions)

    compression_id = answer["selected_id"]
    os.environ["LLM_MODEL_COMPRESSION"] = compression_id
    dotenv.set_key(dotenv_file, "LLM_MODEL_COMPRESSION", compression_id)
except Exception as e:
    print(e)
    exit(0)

print(f"Selected Compression model {compression_id}")



try:
    embedding_model_ids = list(SUPPORTED_EMBEDDING_MODELS)
    import inquirer
    questions = [
        inquirer.List("selected_id", message="Select embedding model", choices=embedding_model_ids), 
    ]
    answer = inquirer.prompt(questions)

    embedding_model_id = answer["selected_id"]
    
    os.environ["LLM_EMBEDDING"] = embedding_model_id
    dotenv.set_key(dotenv_file, "LLM_EMBEDDING", embedding_model_id)
except Exception as e:
    print(e)
    exit(0)

print(f"Selected LLM embedding model {embedding_model_id}")
