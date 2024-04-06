FROM quay.io/jupyter/pytorch-notebook:python-3.11.8

COPY requirements.txt requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:CubiCasa5k:/home/jovyan"

RUN pip install -r requirements.txt
