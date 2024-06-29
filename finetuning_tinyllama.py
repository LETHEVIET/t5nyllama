import datasets
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Define model parameters
max_seq_length = 4096  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-chat-bnb-4bit",  # "unsloth/tinyllama" for 16bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Apply PEFT (Parameter-Efficient Fine-Tuning)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Currently only supports dropout = 0
    bias="none",  # Currently only supports bias = "none"
    use_gradient_checkpointing=False,  # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


# Load the dataset
train = datasets.load_dataset("grammarly/coedit", split="train").to_pandas()
val = datasets.load_dataset("grammarly/coedit", split="validation").to_pandas()

# Data cleaning and preparation
data = pd.concat([train, val])
data[["instruction", "input"]] = data["src"].str.split(": ", n=1, expand=True)
data = data.rename(columns={"tgt": "output"})
data = data.drop(columns=["_id", "src"])

# Stratify based on task for balanced splits
stratify_col = data["task"]

# Split the data into train and test sets
train_df, test_df = train_test_split(
    data, test_size=0.2, random_state=42, stratify=stratify_col
)


def formatting_prompts_func(examples, tokenizer):
    """
    Formats the examples into the desired chat format for training.

    Args:
        examples: A dictionary of examples from the dataset.
        tokenizer: The tokenizer used for processing text.

    Returns:
        A dictionary containing the formatted text for each example.
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        message = [
            {"role": "user", "content": instruction + ": " + input},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {
        "text": texts,
    }


# Create datasets from pandas DataFrames
train_ds = datasets.Dataset.from_pandas(train_df)
test_ds = datasets.Dataset.from_pandas(test_df)

# Map the formatting function to the datasets for chat format conversion
train_ds = train_ds.map(
    formatting_prompts_func,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
)
test_ds = test_ds.map(
    formatting_prompts_func,
    fn_kwargs={"tokenizer": tokenizer},
    batched=True,
)

print(train_ds[0]["text"])
# Define training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=10,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_steps=100,
        save_total_limit=4,  # Limit the total number of checkpoints
        evaluation_strategy="steps",
        eval_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        load_best_model_at_end=True,
        save_strategy="steps",
    ),
)

# Print GPU information
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Train the model
trainer_stats = trainer.train()

# Print memory usage statistics
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save the trained model and tokenizer
print("Saving model to local")
model.save_pretrained("coedit-tinyllama-chat-bnb-4bit")  # Local saving
tokenizer.save_pretrained("coedit-tinyllama-chat-bnb-4bit")

# Evaluate the model (Optional)
# trainer.evaluate()
