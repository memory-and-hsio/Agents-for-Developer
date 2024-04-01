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

# load env file
import dotenv

try:
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)
        
except Exception as e:
    print(e)
    print("Please create .env file")
    exit(0)

print(f"Selected LLM model {os.getenv('LLM_MODEL_ID')}")
print(f"Selected Compression model {os.getenv('LLM_MODEL_COMPRESSION')}")
print(f"Selected LLM embedding model {os.getenv('LLM_EMBEDDING')}")


# local LLM model configuration
from config import SUPPORTED_EMBEDDING_MODELS, SUPPORTED_LLM_MODELS
from optimum.intel import OVWeightQuantizationConfig
from converter import converters, register_configs

from pprint import pprint


# TODO : you need to change following line to your local path
# hsio : memory and high speed io
# mm : multimedia
# ec : ec
# typeC : typeC
DOC_ROOT = f"..\\..\\article\\"
VS_ROOT = f"..\\..\\persistent\\"
MODEL_ROOT = f"..\\..\\model\\"

model_directory = os.path.abspath(MODEL_ROOT)


llm_model_id = os.getenv('LLM_MODEL_ID')
llm_model_configuration = SUPPORTED_LLM_MODELS['English'][llm_model_id]

embedding_model_id = os.getenv('LLM_EMBEDDING')
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[embedding_model_id]

#embedding_model_dir = Path(embedding_model_id)
embedding_model_dir = Path(model_directory + '\\' + embedding_model_id)

if not (embedding_model_dir / "openvino_model.xml").exists():
    model = AutoModel.from_pretrained(embedding_model_configuration["model_id"])
    converters[embedding_model_id](model, embedding_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_configuration["model_id"])
    tokenizer.save_pretrained(embedding_model_dir)
    del model

