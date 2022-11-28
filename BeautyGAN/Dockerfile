FROM tensorflow/tensorflow:1.9.0-devel-py3

WORKDIR /workspace

RUN apt update && \
    apt install -y \
        cmake \
        python3-tk

COPY requirements.txt requirements.txt

RUN pip install -U pip && \
    pip install -r requirements.txt
