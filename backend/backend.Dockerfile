FROM python:3.11-bullseye

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8000
