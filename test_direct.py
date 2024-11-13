# evaluate_concepts.py

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from utils_direct_test import (
    ModelConfig,
    ContentClassifier,
    MetricsLogger
)

def main():
    """Main execution function."""
    # Configuration
    config = ModelConfig()
    concept_labels = ["offensive"]
    
    # Initialize components
    race = "latinx"  # Change to the desired race group
    classifier = ContentClassifier(config, concept_labels)
    classifier.load_model_weights(f"direct_weights/direct_roberta_weights_full_dataset_best.pt")
    data_loader = classifier.load_dataset(f"preprocessed_datasets/preprocessed_{race}_test_dataset.pt")

    # Initialize metrics logger
    metrics_logger = MetricsLogger(base_dir="experiment_logs")

    additional_info = {
        "config": config.__dict__,
        "num_samples": len(data_loader.dataset),
        "num_concepts": len(concept_labels),
        "device": str(config.device)
    }

    metrics = classifier.evaluate(data_loader)
    model_name = f"direct_roberta"
    dataset_name = f"{race}_test"

    # Save metrics
    saved_path = metrics_logger.save_metrics(
        metrics=metrics,
        model_name=model_name,
        dataset_name=dataset_name,
        additional_info=additional_info
    )

    # Evaluate and print results
    print("\nEvaluation Results:")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Coverage: {metrics['coverage']:.4f}")
    print(f"\nMetrics saved to: {saved_path}")

if __name__ == "__main__":
    main()
