# Yelp Polarity Sentiment Analysis

A professional NLP pipeline comparing TF-IDF/Logistic Regression and DistilBERT for sentiment classification.

* Dataset Used: Yelp Polarity dataset.
* Training Samples: 50,000 for Logistic Regression; 50,000 for DistilBERT.
* TF‑IDF Baseline F1 Score: 0.9194.
* DistilBERT F1 Score: 0.9561 (at 50,000 samples).
* Best Performing Model: DistilBERT.
* Recommendation for Deployment: * DistilBERT if high accuracy is critical, as it captures complex semantic relationships better than keywords alone.
    * Logistic Regression if the environment is resource-constrained (CPU-only), as it is much faster and more lightweight with a respectable 91.94% F1 score.



## Results
| Model Type | Training Samples | F1 Score |
| :--- | :--- | :--- |
| Logistic Regression | 50,000 | 91.94% |
| DistilBERT | 50,000 | 95.61% |

## Installation
```bash
pip install -r requirements.txt
