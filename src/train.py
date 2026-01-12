import argparse
import torch
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Ensure the drive directory is in the path so we can import utils
sys.path.append('/content/drive/MyDrive/YELP')
from utils import compute_metrics, normalize_text, load_text_dataset, tokenize_function

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT or Train TF-IDF Baseline")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "tfidf"])
    parser.add_argument("--dataset", type=str, default="yelp", choices=["imdb", "sst2", "yelp"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/YELP/saved_model")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--subset_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42) # Added seed argument
    return parser.parse_args()

def main():
    args = parse_args()
    dataset, text_column = load_text_dataset(args.dataset)
    log_file = '/content/drive/MyDrive/YELP/experiments_log.csv'

    # --- UNIFIED DATA SPLITTING (Ensures reproducibility) ---
    # We shuffle and select indices the SAME way for both models
    train_ds = dataset["train"].shuffle(seed=args.seed).select(range(args.subset_size))
    test_ds = dataset["test"].shuffle(seed=args.seed).select(range(int(args.subset_size * 0.3)))

    # --- PATH 1: TF-IDF + Logistic Regression ---
    if args.model_type == "tfidf":
        print(f"--- Starting TF-IDF Baseline on {args.dataset} (Seed: {args.seed}) ---")

        # Convert the Hugging Face dataset subsets to lists for Sklearn
        train_texts = [normalize_text(t) for t in train_ds[text_column]]
        train_labels = train_ds['label']
        test_texts = [normalize_text(t) for t in test_ds[text_column]]
        test_labels = test_ds['label']

        tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3))
        X_train = tfidf.fit_transform(train_texts)
        X_test = tfidf.transform(test_texts)

        # random_state in LogisticRegression ensures the internal solver is deterministic
        model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=args.seed)
        model.fit(X_train, train_labels)

        y_pred = model.predict(X_test)
        f1_val = f1_score(test_labels, y_pred, average='binary')

        log_data = {
            "experiment_name": f"LR_{args.dataset}_{args.subset_size}_seed{args.seed}",
            "model_type": "Logistic Regression",
            "dataset": args.dataset,
            "f1_score": round(f1_val, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
        print(f"LR Complete. F1: {f1_val}")

    # --- PATH 2: Transformer Fine-Tuning ---
    else:
        print(f"--- Starting Transformer Fine-tuning on {args.dataset} (Seed: {args.seed}) ---")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        tokenized = train_ds.map(lambda x: tokenize_function(x, tokenizer, text_column, args.max_length), batched=True)
        eval_tokenized = test_ds.map(lambda x: tokenize_function(x, tokenizer, text_column, args.max_length), batched=True)

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            seed=args.seed, # Ensures Transformer training is reproducible
            fp16=torch.cuda.is_available()
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        
        # Save weights and Evaluate
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        metrics = trainer.evaluate()
        f1_val = metrics.get("eval_f1")

        log_data = {
            "experiment_name": f"Transformer_{args.dataset}_{args.subset_size}_seed{args.seed}",
            "model_type": "DistilBERT",
            "dataset": args.dataset,
            "f1_score": round(f1_val, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
        print(f"Transformer Complete. F1 logged.")

if __name__ == "__main__":
    main()
