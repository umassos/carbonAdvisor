# syntax=docker/dockerfile:1

FROM amd64/python:3.9-bullseye

RUN mkdir /workspace

ADD *.py /workspace/
ADD requirements.txt /workspace/requirements.txt

ADD scale_profile.yaml /workspace/scale_profile.yaml
ADD pages /workspace/pages
ADD traces /workspace/traces

RUN pip3 install -r /workspace/requirements.txt

WORKDIR /workspace

ENTRYPOINT ["streamlit", "run", "Policy_Model.py"]

