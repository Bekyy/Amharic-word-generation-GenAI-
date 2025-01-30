# Amharic-word-generation-GenAI-

## 📌 Project Overview

This project focuses on generating antonyms for Amharic words using a fine-tuned pretrained transformer model. The model is trained to learn word relationships, specifically antonyms, leveraging T5 (Text-to-Text Transfer Transformer) for sequence generation.

## 🚀 Features

* Fine-tuned CodeT5 (or a similar pretrained model) for Amharic antonym generation.

* Training on a structured dataset containing Amharic word pairs labeled as antonyms.

* Evaluation using BLEU score and accuracy metrics.

* Deployment-ready implementation using Hugging Face Transformers.

## 📂 Dataset

* The dataset consists of three columns:

- word1 - The target Amharic word.

- word2 - The corresponding antonym.

* The data is preprocessed and tokenized using Hugging Face's Tokenizer.

## 🏗 Model Training

* The model is trained using T5ForConditionalGeneration with the following setup:
```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")

training_args = Seq2SeqTrainingArguments(
    output_dir="./m2m-amharic-antonym-augmented",  # Directory to save the model
    evaluation_strategy="epoch",        # Evaluate at the end of each epoch
    learning_rate=2e-5,                 # Learning rate
    per_device_train_batch_size=16,     # Training batch size
    per_device_eval_batch_size=16,      # Evaluation batch size
    num_train_epochs=100,               # Number of epochs
    weight_decay=0.01,                  # Weight decay for regularization
    predict_with_generate=True,         # Allow prediction generation
    # logging_dir="./logs",             # Directory for logs
    logging_steps=10,                   # Log every 10 steps
    push_to_hub=True,                   # Enable pushing to Hugging Face hub
    report_to="tensorboard",
    save_strategy="epoch",  
    save_total_limit=1,                 # Limit on the number of saved checkpoints
    load_best_model_at_end=True,
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)
```
## 🔥 Inference Example

To generate an antonym for a given Amharic word:
```python
from transformers import pipeline

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./m2m-amharic-antonym-augmented")
tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")

# Define a pipeline
antonym_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Generate an antonym
output = antonym_generator("ላይ")
print(output[0]["generated_text"])
## 📦 Installation & Dependencies
```

To install the required dependencies:
```bash
pip install transformers torch datasets
```

## ⭐ Acknowledgment

Special thanks to Hugging Face and the Amharic NLP community for their contributions to low-resource language modeling.
