import argparse
import torch
import sys
import os

# Ensure the drive directory is in the path to import your utils
sys.path.append('/content/drive/MyDrive/YELP')
from utils import normalize_text

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_inference(text, model_path):
    # 1. Load Model and Tokenizer
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 2. Preprocess (using your modular utility)
    clean_text = normalize_text(text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    
    # 3. Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. Process Output
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    
    # Yelp Mapping: 0 -> Negative, 1 -> Positive
    label = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[0][prediction].item()
    
    print(f"\nResult:")
    print(f"  Input: {text}")
    print(f"  Label: {label}")
    print(f"  Confidence: {confidence:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single sentence")
    parser.add_argument("--text", type=str, required=True, help="The sentence to analyze")
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/YELP/saved_model", 
                        help="Path to the saved transformer model")
    
    args = parser.parse_args()
    run_inference(args.text, args.model_path)
