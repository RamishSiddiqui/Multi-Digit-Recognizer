FROM python:3.8-slim-buster

ENV PYTHONUNBUFFERED 1

RUN apt-get update

RUN apt-get install libgl1-mesa-glx -y

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN mkdir /DigitRecognizer

WORKDIR /DigitRecognizer

COPY requirements.txt requirements.txt

COPY . .

RUN pip install -r requirements.txt