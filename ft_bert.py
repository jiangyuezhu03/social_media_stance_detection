import torch
import torch.nn as nn
torch.set_num_threads(12)
import os
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
from transformers import DataCollatorWithPadding

from transformers import (
    BertTokenizer,
    BertConfig,
    BertModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,

)
from custom_encoder import BertForStance3Way
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

model_path = "models/stance_ch"
tokenizer = BertTokenizer.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)
config.num_labels = 3

model = BertForStance3Way(config)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

## Load binary SD model
print("Loading weights from FutureMa/stance_ch")
from transformers import AutoModelForSequenceClassification
old_model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.bert.load_state_dict(old_model.bert.state_dict())

# Load data
from datasets import load_from_disk
dataset = load_from_disk("c-stance_entertainment")

# Process dataset
# def tokenize_fn(batch):
#     return tokenizer(
#         batch["target"],
#         batch["text"],
#         truncation=True,
#         padding=False,
#         max_length=512
#     )
#
# encoded_dataset = dataset.map(tokenize_fn, batched=True)
# encoded_dataset = encoded_dataset.rename_column("label", "labels")
# encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

MAX_TARGET_LEN = 128
MAX_TEXT_LEN = 196

def tokenize_fn(batch):
    tokenized = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }

    for target, text in zip(batch["target"], batch["text"]):
        target_tokens = tokenizer.tokenize(target)[:MAX_TARGET_LEN]
        text_tokens = tokenizer.tokenize(text)[:MAX_TEXT_LEN]

        tokens = (
            [tokenizer.cls_token]
            + target_tokens
            + [tokenizer.sep_token]
            + text_tokens
            + [tokenizer.sep_token]
        )

        encoded = tokenizer.convert_tokens_to_ids(tokens)
        attn_mask = [1] * len(encoded)
        token_type_ids = [0] * (len(target_tokens) + 2) + [1] * (len(text_tokens) + 1)

        tokenized["input_ids"].append(encoded)
        tokenized["attention_mask"].append(attn_mask)
        tokenized["token_type_ids"].append(token_type_ids)

    return tokenized

encoded_dataset = dataset.map(tokenize_fn, batched=True)

import os
output_dir = "models/stance_model_3way_sep_pad"
log_dir = "./logs"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir=log_dir,
    report_to=["tensorboard"],
    logging_steps=50,
    logging_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


trainer.train()

print(f"Saved mode to {output_dir}")
trainer.save_model(output_dir)
