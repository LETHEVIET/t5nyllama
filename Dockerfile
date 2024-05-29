FROM python:3.10

WORKDIR /app

USER root 
RUN mkdir -p /root/.cache 
RUN chown -R $USER:$USER /root/.cache 
USER $USER 

RUN mkdir ./texteditor-model
COPY . .

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 download_dependencies.py

ENTRYPOINT ["python", "app.py"]