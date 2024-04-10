FROM quay.io/jupyter/pytorch-notebook:python-3.11.8

COPY requirements.txt requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:CubiCasa5k:/home/jovyan"

RUN pip install -r requirements.txt

RUN mkdir /tmp/models && gdown 1d7cb80hRT2aJG7cKIyDiUq5roEWpgbyc -O /tmp/models/model_6_566.00565.pt
