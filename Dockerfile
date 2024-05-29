FROM python:3.10

RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

RUN mkdir ./texteditor-model
COPY --chown=user . .

# RUN apt-get update && apt-get -y upgrade
# RUN apt-get install -y build-essential
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN python3 download_dependencies.py

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

ENTRYPOINT ["python", "app.py"]