FROM python:3.10

USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN mkdir ./texteditor-model
COPY . .

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 download_dependencies.py

ENTRYPOINT ["python", "app.py"]