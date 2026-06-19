"""
RoBERTa Sentiment Classifier Training Pipeline
==============================================
Fine-tunes a pre-trained RoBERTa model on preprocessed Amazon review texts
using Hugging Face's transformers Trainer API.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ReviewSentimentDataset(Dataset):
    """
    Custom PyTorch Dataset for Tokenized Amazon Reviews
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def map_score_to_sentiment(score):
    """
    Maps 1-5 star ratings to standard 3-class sentiment:
    1-2 Stars -> 0 (Negative)
    3 Stars   -> 1 (Neutral)
    4-5 Stars -> 2 (Positive)
    """
    if score in [1, 2]:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    elif score in [4, 5]:
        return 2  # Positive
    else:
        # Fallback if float scores are present
        if score < 3.0:
            return 0
        elif score == 3.0:
            return 1
        else:
            return 2

def compute_metrics(eval_pred):
    """
    Calculates evaluation metrics (Accuracy & Macro F1) for the Trainer evaluations
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate macro average F1 and accuracy
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1_macro': f1
    }

def train_sentiment_model(
    data_path: str = "data/processed/clean_reviews.csv",
    model_name: str = "roberta-base",
    output_dir: str = "models/sentiment/weights",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    test_size: float = 0.15,
    random_seed: int = 42
):
    """
    Loads, processes, splits, tokenizes, and fine-tunes RoBERTa
    """
    logger.info(f"Loading preprocessed reviews from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Please run preprocess.py first.")
        
    df = pd.read_csv(data_path)
    
    # Keep only target columns and drop rows with empty values
    df = df.dropna(subset=["clean_review_text", "Score"])
    
    logger.info("Mapping review ratings to sentiment classes...")
    df["label"] = df["Score"].apply(map_score_to_sentiment)
    
    # Print label distribution
    label_counts = df["label"].value_counts().to_dict()
    logger.info(f"Label distribution: Negative (0): {label_counts.get(0, 0)}, Neutral (1): {label_counts.get(1, 0)}, Positive (2): {label_counts.get(2, 0)}")
    
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["clean_review_text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label"].tolist()
    )
    
    logger.info(f"Split data size: Train={len(train_texts)}, Validation={len(val_texts)}")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize texts
    logger.info("Tokenizing texts...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    
    # Create Datasets
    train_dataset = ReviewSentimentDataset(train_encodings, train_labels)
    val_dataset = ReviewSentimentDataset(val_encodings, val_labels)
    
    # Load model
    logger.info(f"Loading base classification model '{model_name}' (num_labels=3)...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using training device: {device.upper()}")
    model.to(device)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=learning_rate,
        seed=random_seed,
        report_to="none"  # Set to "mlflow" or "tensorboard" in production
    )
    
    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Run training
    logger.info("Starting model fine-tuning...")
    trainer.train()
    
    # Evaluate final model
    logger.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Final Validation Results: Accuracy={eval_results['eval_accuracy']:.4f}, Macro F1={eval_results['eval_f1_macro']:.4f}")
    
    # Save the model and tokenizer
    best_model_dir = os.path.join(output_dir, "best_model")
    logger.info(f"Saving best model and tokenizer to {best_model_dir}...")
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    # Generate final classification report on validation split
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    
    report = classification_report(
        val_labels,
        pred_labels,
        target_names=["Negative", "Neutral", "Positive"]
    )
    logger.info("Classification Report:\n" + report)
    
    # Write report summary to file
    with open(os.path.join(output_dir, "eval_report.txt"), "w") as f:
        f.write(report)
        
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa Sentiment Classifier")
    parser.add_argument("--data_path", type=str, default="data/processed/clean_reviews.csv", help="Path to clean CSV reviews file")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Hugging Face model checkpoint to fine-tune")
    parser.add_argument("--output_dir", type=str, default="models/sentiment/weights", help="Directory to save checkpoints and best model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=256, help="Max token length")
    
    args = parser.parse_args()
    
    train_sentiment_model(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_len
    )
