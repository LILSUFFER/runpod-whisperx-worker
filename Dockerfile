FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]
WORKDIR /

RUN apt-get update && \
    apt-get install -y ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY builder/requirements.txt /builder/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /builder/requirements.txt

COPY src/rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
