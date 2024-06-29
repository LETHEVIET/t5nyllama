---
title: T5nyllama
emoji: 🐳
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
---

## Lightweight English Text Editing Assistant (t5nyllama)

This repository houses the source code for t5nyllama, a lightweight English text editing assistant designed to provide a simple and efficient way to enhance your writing.

**Huggingface Spaces:**
https://huggingface.co/spaces/letheviet/t5nyllama

**How it Works:**

t5nyllama uses a two-step approach:

1. **Text Generation:** The core of the assistant is a TinyLlama model, specifically fine-tuned for text editing. This model is designed to improve the flow and clarity of your text, making it more polished and engaging. However, TinyLlama is **relatively small and not particularly adept at complex grammar correction.**

2. **Grammar Correction:** To address this limitation, we employ a powerful Flan-T5 model for a second pass. This model takes the output of the TinyLlama model and carefully analyzes it for grammatical errors. It then suggests corrections, ensuring your final text is grammatically sound and ready for publication.

**Key Features:**

- **Lightweight and Efficient:** The TinyLlama model is quantized to 4-bit precision, minimizing memory usage and computational demands, making it suitable for resource-constrained environments.
- **Focused on Text Improvement:** TinyLlama excels at refining the overall quality of your writing, making it more readable and engaging.
- **Enhanced Grammar Accuracy:** The Flan-T5 model provides a robust final check for grammatical errors, ensuring your text is free from mistakes.

**Design Principles:**

- **Local Application:** Prioritizes offline functionality, allowing you to edit text without requiring an internet connection.
- **Lightweight Design:** Minimizes resource consumption, making the application suitable for a wide range of devices and systems.

## Installation

**1. Clone the Repository:**

```shell
git clone https://github.com/LETHEVIET/t5nyllama.git
```

**2. Install Dependencies:**

```shell
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
mkdir ./texteditor-model
python3 download_dependencies.py
```

**3. Run the Application:**

```shell
python3 app.py
```

## Docker Deployment

**1. Build Docker Image:**

```shell
docker build . -t t5nyllama
```

**2. Run Docker Image:**

```shell
docker run -p 7860:7860 t5nyllama
```

## Fine-Tuning TinyLlama

The fine-tuning script follows the UnslothAI example for fine-tuning Tiny Llama. Please install dependencies from [unsloth](https://github.com/unslothai/unsloth) before running the script.

```shell
python finetuning_tinyllama.py
```

## References

- **Unsloth Fast Fine-Tuning LLM:** https://github.com/unslothai/unsloth
- **Dataset Card for CoEdIT: Text Editing via Instruction Tuning :** https://huggingface.co/datasets/grammarly/coedit
- **Grammar-Synthesis-Large: FLAN-t5:** https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis
- **ALLECS: A Lightweight Language Error Correction System:** https://github.com/nusnlp/ALLECS
- **Python Bindings for llama.cpp:** https://github.com/abetlen/llama-cpp-python
- **Gradio: Build Machine Learning Web Apps — in Python:** https://github.com/gradio-app/gradio
