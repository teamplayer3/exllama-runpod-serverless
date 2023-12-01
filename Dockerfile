FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir data
WORKDIR /data

COPY requirements.txt requirements.txt

# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip && \
    pip install ninja && \
    pip install -r requirements.txt

COPY handler.py /data/handler.py
COPY __init.py__ /data/__init__.py

ENV MODEL_REPO=""
ENV PROMPT_PREFIX=""
ENV PROMPT_SUFFIX=""
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

CMD [ "python", "-m", "handler" ]