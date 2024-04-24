
"""
to use model from HuggingFace, you need to login to Huggingface and create access token.

pease refer to this link.  https://huggingface.co/docs/hub/security-tokens
"""

import os
import dotenv

## locate .env file
try:
    dotenv_file = dotenv.find_dotenv()
    print("dotenv file found", dotenv_file)
    # Print all of the environment variables
    for key, value in os.environ.items():
        print(f'{key}={value}')
except Exception as e:
    print(e)
    print("Please create .env file")
    exit(0)


## login to huggingfacehub to get access to pretrained model 

from huggingface_hub import notebook_login, whoami

try:
    whoami()
    print('Authorization token already provided')
except OSError:
    notebook_login()

