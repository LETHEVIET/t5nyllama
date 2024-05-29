import gdown
import nltk

id = "1TnPssg0CkWQ_thuAH8cY3hdB2J18A0Kl"
output = "texteditor-model/coedit-tinyllama-chat-bnb-4bit-unsloth.Q4_K_M.gguf"
gdown.download(id=id, output=output)

nltk.download('punkt', download_dir="./nltk_data")