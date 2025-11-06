FROM python:3.11-bullseye

COPY requirements.txt /app/
WORKDIR /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8000



