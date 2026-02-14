FROM python:3.12-slim-buster

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir awscli \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python3","app.py"]