FROM pytorch/pytorch:latest

WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt

RUN python run_test.py
