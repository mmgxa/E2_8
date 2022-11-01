FROM python:3.9.12-slim

WORKDIR /app

COPY "test_serve" "./test_serve/"

COPY "requirements.txt" "."

RUN apt-get update && apt-get install curl -y

RUN pip install --no-cache-dir -r requirements.txt

# RUN curl -X POST  http://$serve_host/models?url=https://emlo-v2-s8.s3.us-east-2.amazonaws.com/cifar10s3.mar

ENTRYPOINT ["bash"]
