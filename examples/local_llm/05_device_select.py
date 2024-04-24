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
    embedding_device_ids = ['CPU', 'GPU', 'NPU']
    import inquirer
    questions = [
        inquirer.List("selected_id", message="Select embedding device", choices=embedding_device_ids), 
    ]
    answer = inquirer.prompt(questions)

    embedding_device_id = answer["selected_id"]
    
    os.environ["EMBEDDING_DEVICE_ID"] = embedding_device_id
    dotenv.set_key(dotenv_file, "EMBEDDING_DEVICE_ID", embedding_device_id)
except Exception as e:
    print(e)
    exit(0)

print(f"Selected embedding device {embedding_device_id}\n\n")


try:
    llm_device_ids = ['CPU', 'GPU', 'NPU']
    import inquirer
    questions = [
        inquirer.List("selected_id", message="Select LLM device", choices=llm_device_ids), 
    ]
    answer = inquirer.prompt(questions)

    llm_device_id = answer["selected_id"]
    
    os.environ["LLM_DEVICE_ID"] = llm_device_id
    dotenv.set_key(dotenv_file, "LLM_DEVICE_ID", llm_device_id)
except Exception as e:
    print(e)
    exit(0)

print(f"Selected LLM device {llm_device_id}\n\n")

