# Llama-2 Fine-Tuning using QLora

## Table of Contents

- [Setup](#setup)
- [Dataset](#dataset)
- [Loading the Model](#loading-the-model)
- [Loading the Trainer](#loading-the-trainer)
- [Train the Model](#train-the-model)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Setup

Before getting started, you need to set up the required libraries and dependencies. Follow the steps below to install them:

```bash
!pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
!pip install -q datasets bitsandbytes einops wandb
## Dataset
For this project, we use the "NLPC-UOM/Student_feedback_analysis_dataset" dataset. You can replace it with your own dataset if needed. Here's how to load the dataset:
from datasets import load_dataset

dataset_name = 'NLPC-UOM/Student_feedback_analysis_dataset'  # Replace with your dataset name
dataset = load_dataset(dataset_name, split="train")
Loading the Model
We load the Llama-2-7B model for fine-tuning. Additionally, we quantize the model into 4-bit for memory efficiency. Here's how to load the model:
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "TinyPixel/Llama-2-7B-bf16-sharded"

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
Loading the Trainer
We use the SFTTrainer from the TRL library to fine-tune the model using PEFT adapters. Here's how to set up the trainer:
from transformers import TrainingArguments
from trl import SFTTrainer

# Configure training arguments
output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

# Create the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
Train the Model
Now, it's time to train the model. Simply call trainer.train() to start the training process.
trainer.train()
Usage
You can use the fine-tuned model for various natural language processing tasks, including chatbot development. Here's an example of how to generate text using the model:
text = "Student : Courses : Feedbacks"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

