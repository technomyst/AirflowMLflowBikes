FROM apache/airflow:2.7.1

COPY ./docker/airflow/requirements.txt ./
COPY ./.env ./
COPY ./bashrc_generate.sh ./
RUN pip install pip==23.1.2
RUN pip install --no-cache-dir -r requirements.txt
COPY ./dags ./dags

ENV PYTHONPATH "${PYTHONPATH}:/opt/airflow/dags"
CMD ['bashrc_generate.sh']
