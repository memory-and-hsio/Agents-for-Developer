
pip uninstall -y openvino-dev openvino openvino-nightly optimum optimum-intel
pip install --extra-index-url https://download.pytorch.org/whl/cpu "git+https://github.com/huggingface/optimum-intel.git" "git+https://github.com/openvinotoolkit/nncf.git" "datasets" "accelerate" "openvino-nightly" "gradio" "onnx" "einops" "transformers_stream_generator" "tiktoken" "transformers>=4.38.1" "bitsandbytes" "chromadb" "sentence_transformers" "langchain>=0.1.7" "langchainhub" "unstructured" "scikit-learn" "python-docx" "pdfminer.six"
pip install --upgrade-strategy eager "optimum[openvino,nncf]" --quiet
pip install --upgrade-strategy eager "langchain" --quiet
pip install --upgrade-strategy eager "chromadb" --quiet
rem pip freeze -l | Out-File -Encoding UTF8 requirements.txt
