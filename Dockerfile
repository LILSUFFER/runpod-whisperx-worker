FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /

RUN apt-get update -y && \
    apt-get install --yes --no-install-recommends \
    sudo ca-certificates git wget curl bash libgl1 libx11-6 \
    software-properties-common ffmpeg build-essential -y && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY builder/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

COPY src /

CMD ["python", "-u", "/rp_handler.py"]
