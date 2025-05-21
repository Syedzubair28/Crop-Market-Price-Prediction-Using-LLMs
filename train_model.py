import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from transformers import EarlyStoppingCallback
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.metrics import mean_squared_error

# Custom Dataset Class (unchanged)
class CropCostDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"mse": mean_squared_error(labels, predictions)}

def main():
    # Load data with sampling if too large
    df = pd.read_csv("crop_csv_file_price.csv").dropna()
    
    # Sample 20% data if you have >50k rows (optional)
    if len(df) > 50000:
        df = df.sample(frac=0.2, random_state=42)
    
    texts = df.apply(
        lambda row: f"State: {row['State_Name']}, District: {row['District_Name']}, "
                   f"Year: {row['Crop_Year']}, Season: {row['Season']}, Crop: {row['Crop']}",
        axis=1
    ).tolist()
    labels = df["Cost"].tolist()

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Initialize model
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Create datasets with shorter max_length
    train_dataset = CropCostDataset(train_texts, train_labels, tokenizer, max_length=32)  # Reduced from 64
    val_dataset = CropCostDataset(val_texts, val_labels, tokenizer, max_length=32)

    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir="./fast_training_output",
        evaluation_strategy="steps",          # Check metrics every X steps
        eval_steps=200,                      # Evaluate every 200 steps
        save_strategy="steps",
        save_steps=200,
        learning_rate=3e-5,                  # Slightly higher learning rate
        per_device_train_batch_size=32,      # Increased batch size
        per_device_eval_batch_size=32,
        num_train_epochs=15,                  # Reduced from 10
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        logging_steps=100,
        report_to="none",                   # Disable WandB/MLflow reporting
        optim="adamw_torch",                 # Better optimizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Sooner stopping
    )

    # Train
    print("Starting optimized training...")
    trainer.train()
    
    # Save model
    save_path = "./fast_crop_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nOptimized model saved to {save_path}")

if __name__ == "__main__":
    main()
