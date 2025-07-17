FROM continuumio/miniconda3

COPY environment.yml .
COPY requirements.txt .

RUN conda env create -f environment.yml

# 激活环境并安装 pip 包
SHELL ["conda", "run", "-n", "llasa", "/bin/bash", "-c"]
RUN pip install -r requirements.txt

WORKDIR /workspace

