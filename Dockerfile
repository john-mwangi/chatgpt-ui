FROM python:3.10-slim

ENV APP_HOME /app

WORKDIR ${APP_HOME}

COPY . .

# CMD ls -la && pwd
RUN pip install -r requirements.txt
CMD bash -c "streamlit run login.py --server.address=0.0.0.0 --server.port=8501"

# DOCKER_BUILDKIT=1 docker build -f Dockerfile -t chatgpt-ui .
# docker run -p 8501:8501 chatgpt-ui