FROM pytorch/pytorch:latest

WORKDIR /workspace

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git

WORKDIR /workspace

RUN git clone https://github.com/kauevestena/open_llama.git

RUN pip install -r requirements.txt

RUN python open_llama/run_test.py
