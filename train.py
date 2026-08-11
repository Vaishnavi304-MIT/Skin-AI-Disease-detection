import os
import certifi
import numpy as np
import torch
import evaluate
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

# Fix potential SSL certificate loading issues
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

try:
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

# Path where your dataset split is located
DATASET_DIR = "./augmented dataset"

def main():
    print("Loading skin disease dataset...")
    # Load dataset using Hugging Face datasets ImageFolder loader
    dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)

    # Extract class labels
    labels = dataset["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    print(f"Detected {len(labels)} classes: {labels}")

    # Load Base DINOv2 Image Processor
    model_checkpoint = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_checkpoint)

    # Image Transformation pipeline for DINOv2
    def transform_batch(example_batch):
        images = [x.convert("RGB") for x in example_batch["image"]]
        inputs = processor(images, return_tensors="pt")
        inputs["label"] = example_batch["label"]
        return inputs

    prepared_ds = dataset.with_transform(transform_batch)

    # Load Base DINOv2 Model for Image Classification
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Accuracy Metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # Training Configuration
    training_args = TrainingArguments(
        output_dir="./dinov2-finetuned-skin-disease",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
        hub_model_id="shindevaishnavi304/dinov2-finetuned-skin-disease",
    )

    # Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"] if "validation" in prepared_ds else prepared_ds["test"],
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    print("Starting DINOv2 Fine-Tuning...")
    trainer.train()

    print("Pushing fine-tuned model to Hugging Face Hub...")
    trainer.push_to_hub()
    

if __name__ == "__main__":
    main()