
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
from pprint import pprint

llm_model_id = os.getenv('LLM_MODEL_ID')
model_to_compress = os.getenv('LLM_MODEL_COMPRESSION')

llm_model_configuration = SUPPORTED_LLM_MODELS['English'][llm_model_id]
pprint(llm_model_configuration)


# TODO : you need to change following line to your local path
# hsio : memory and high speed io
# mm : multimedia
# ec : ec
# typeC : typeC
DOC_ROOT = f"..\\..\\article\\"
VS_ROOT = f"..\\..\\persistent\\"
MODEL_ROOT = f"..\\..\\model\\"

model_directory = os.path.abspath(MODEL_ROOT)


from optimum.intel import OVWeightQuantizationConfig
from converter import converters, register_configs

register_configs()

nncf.set_log_level(logging.ERROR)

#Instantiate one of the configuration classes of the library from a pretrained model configuration.
#The configuration class is used to create the model and tokenizer.
try:
    pt_model_id = llm_model_configuration["model_id"]
    pt_model_name = llm_model_id.split("-")[0]

    model_type = AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True).model_type
    
    #fp16_model_dir = Path(llm_model_id) / "FP16"
    #int8_model_dir = Path(llm_model_id) / "INT8_compressed_weights"
    #int4_model_dir = Path(llm_model_id) / "INT4_compressed_weights"
    fp16_model_dir = Path(model_directory + '\\' + llm_model_id + '\\' + "FP16")
    int8_model_dir = Path(model_directory + '\\' + llm_model_id + '\\' + "INT8_compressed_weights")
    int4_model_dir = Path(model_directory + '\\' + llm_model_id + '\\' + "INT4_compressed_weights")
    if model_to_compress == "INT4-model":
        model_dir = int4_model_dir
        weight_format = "int4"
    elif model_to_compress == "INT8-model":
        model_dir = int8_model_dir
        weight_format = "int8"
    else:
        model_dir = fp16_model_dir
        weight_format = "fp16"
    print(f"model dir  {model_dir}")

except Exception as e:
    print(e)
    exit(0)

# optimum-cli export openvino --model "meta-llama/Llama-2-7b-chat-hf" ov_model_dir_llama
    
try:
    #call 'optimum-cli' command to export the model to OpenVINO format
    import subprocess
    subprocess.call([
            'optimum-cli.exe', 
            'export', 
            'openvino',
            '--model',
            pt_model_id,
            '--weight-format',
            weight_format,
            model_dir
    ])

 
except Exception as e:
    print(e)
    exit(0)

exit(0)

try:
    def convert_to_fp16():
        if (fp16_model_dir / "openvino_model.xml").exists():
            return
        if not llm_model_configuration["remote"]:
            remote_code = llm_model_configuration.get("remote_code", False)
            model_kwargs = {}
            if remote_code:
                model_kwargs = {
                    "trust_remote_code": True,
                    "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
                }
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=False, **model_kwargs
            )
            ov_model.half()
            ov_model.save_pretrained(fp16_model_dir)
            del ov_model
        else:
            model_kwargs = {}
            if "revision" in llm_model_configuration:
                model_kwargs["revision"] = llm_model_configuration["revision"]
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_configuration["model_id"],
                torch_dtype=torch.float32,
                trust_remote_code=True,
                **model_kwargs
            )
            converters[pt_model_name](model, fp16_model_dir)
            del model
        #gc.collect()

    def convert_to_int8():
        if (int8_model_dir / "openvino_model.xml").exists():
            return
        int8_model_dir.mkdir(parents=True, exist_ok=True)
        if not llm_model_configuration["remote"]:
            remote_code = llm_model_configuration.get("remote_code", False)
            model_kwargs = {}
            if remote_code:
                model_kwargs = {
                    "trust_remote_code": True,
                    "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
                }
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False, load_in_8bit=True, **model_kwargs
            )
            ov_model.save_pretrained(int8_model_dir)
            del ov_model
        else:
            convert_to_fp16()
            ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int8_model_dir / "config.json")
            configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
            if configuration_file.exists():
                shutil.copy(
                    configuration_file, int8_model_dir / f"configuration_{model_type}.py"
                )
            compressed_model = nncf.compress_weights(ov_model)
            ov.save_model(compressed_model, int8_model_dir / "openvino_model.xml")
            del ov_model
            del compressed_model
        #gc.collect()

    def convert_to_int4():
        compression_configs = {
            "zephyr-7b-beta": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "mistral-7b": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "minicpm-2b-dpo": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "notus-7b-v1": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "neural-chat-7b-v3-1": {
                "sym": True,
                "group_size": 64,
                "ratio": 0.6,
            },
            "llama-2-chat-7b": {
                "sym": True,
                "group_size": 128,
                "ratio": 0.8,
            },
            "chatglm2-6b": {
                "sym": True,
                "group_size": 128,
                "ratio": 0.72,
            },
            "qwen-7b-chat": {
                "sym": True, 
                "group_size": 128, 
                "ratio": 0.6
            },
            'red-pajama-3b-chat': {
                "sym": False,
                "group_size": 128,
                "ratio": 0.5,
            },
            "default": {
                "sym": False,
                "group_size": 128,
                "ratio": 0.8,
            },
        }

        model_compression_params = compression_configs.get(
            llm_model_id, compression_configs["default"]
        )
        if (int4_model_dir / "openvino_model.xml").exists():
            return
        int4_model_dir.mkdir(parents=True, exist_ok=True)
        if not llm_model_configuration["remote"]:
            remote_code = llm_model_configuration.get("remote_code", False)
            model_kwargs = {}
            if remote_code:
                model_kwargs = {
                    "trust_remote_code" : True,
                    "config": AutoConfig.from_pretrained(pt_model_id, trust_remote_code=True)
                }
            ov_model = OVModelForCausalLM.from_pretrained(
                pt_model_id, export=True, compile=False,
                quantization_config=OVWeightQuantizationConfig(bits=4, **model_compression_params),
                **model_kwargs
            )
            ov_model.save_pretrained(int4_model_dir)
            del ov_model

        else:
            convert_to_fp16()
            ov_model = ov.Core().read_model(fp16_model_dir / "openvino_model.xml")
            shutil.copy(fp16_model_dir / "config.json", int4_model_dir / "config.json")
            configuration_file = fp16_model_dir / f"configuration_{model_type}.py"
            if configuration_file.exists():
                shutil.copy(
                    configuration_file, int4_model_dir / f"configuration_{model_type}.py"
                )
            mode = nncf.CompressWeightsMode.INT4_SYM if model_compression_params["sym"] else \
                nncf.CompressWeightsMode.INT4_ASYM
            del model_compression_params["sym"]
            compressed_model = nncf.compress_weights(ov_model, mode=mode, **model_compression_params)
            ov.save_model(compressed_model, int4_model_dir / "openvino_model.xml")
            del ov_model
            del compressed_model
        #gc.collect()

    def model_compression(id):
        if id == "FP16-model":
            return convert_to_fp16()
        elif id == "INT8-model":
            return convert_to_int8()
        elif id == "INT4-model":
            return convert_to_int4()
        
    model_compression(os.getenv('LLM_MODEL_COMPRESSION'))

except Exception as e:
    print(e)
    exit(0)

print("\n\nSuccess. Model conversion completed.   you can ignore temp file error message above.")

try:
    fp16_weights = fp16_model_dir / "openvino_model.bin"
    int8_weights = int8_model_dir / "openvino_model.bin"
    int4_weights = int4_model_dir / "openvino_model.bin"

    if fp16_weights.exists():
        print(f"Size of FP16 model is {fp16_weights.stat().st_size / 1024 / 1024:.2f} MB")
    for precision, compressed_weights in zip([8, 4], [int8_weights, int4_weights]):
        if compressed_weights.exists():
            print(
                f"Size of model with INT{precision} compressed weights is {compressed_weights.stat().st_size / 1024 / 1024:.2f} MB"
            )
        if compressed_weights.exists() and fp16_weights.exists():
            print(
                f"Compression rate for INT{precision} model: {fp16_weights.stat().st_size / compressed_weights.stat().st_size:.3f}"
            )
except Exception as e:
    print(e)
    exit(0)
