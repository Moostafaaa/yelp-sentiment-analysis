import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset

def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds) 
    return {"accuracy": acc, "f1": f1}

def normalize_text(text: str) -> str:
    return " ".join(text.split())

def load_text_dataset(dataset_name: str):
    # Mapping logic for datasets
    if dataset_name == "yelp":
        dataset = load_dataset("yelp_polarity")
        text_column = "text"
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb")
        text_column = "text"
    else:
        # Defaulting to SST2 structure
        dataset = load_dataset("glue", "sst2")
        text_column = "sentence"
    return dataset, text_column

def tokenize_function(examples, tokenizer, text_column, max_length: int):
    # Normalize input text to reduce noise
    texts = [normalize_text(t) for t in examples[text_column]]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
