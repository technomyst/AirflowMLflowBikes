FROM continuumio/miniconda3:23.9.0-0

RUN pip install mlflow boto3 pymysql

#ADD ./app
#WORKDIR /app

RUN mkdir -p /app
RUN chown root /app
USER root
WORKDIR /app