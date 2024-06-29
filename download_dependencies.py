import os

import gdown
import nltk

id = "1TnPssg0CkWQ_thuAH8cY3hdB2J18A0Kl"

if not os.path.exists("./texteditor-model"):
    os.mkdir("./texteditor-model")

output = "texteditor-model/coedit-tinyllama-chat-bnb-4bit-unsloth.Q4_K_M.gguf"
gdown.download(id=id, output=output)

nltk.download("punkt", download_dir="./nltk_data")
