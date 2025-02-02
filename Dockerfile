FROM python:3.11.11-slim-bullseye

WORKDIR /openwebui

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["open-webui serve"]