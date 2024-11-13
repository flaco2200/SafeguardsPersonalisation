# utils.py

import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score

@dataclass
class ModelConfig:
    """Configuration settings for the classification model."""
    batch_size: int = 32
    decision_threshold: float = 0.5
    run: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ContentClassifier:
    """Main class for content classification using RoBERTa."""

    def __init__(self, config: ModelConfig, concept_labels: List[str]):
        self.config = config
        self.concept_labels = concept_labels
        self.device = config.device
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the RoBERTa model and tokenizer."""
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=len(self.concept_labels),
            problem_type="multi_label_classification"
        ).to(self.device)

    def load_model_weights(self, weights_path: str):
        """Load trained model weights."""
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def load_dataset(self, dataset_path: str) -> DataLoader:
        """Load and prepare the dataset for evaluation."""
        data = torch.load(dataset_path)
        dataset = TensorDataset(
            data['input_ids'].to(self.device),
            data['attention_mask'].to(self.device),
            data['labels'].to(self.device).float()
        )
        return DataLoader(dataset, batch_size=self.config.batch_size)

    def _uncertainty_propagation(self, offensive_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate final probability using uncertainty propagation.
        Uses the complement rule: P(any concept is true) = 1 - P(all concepts are false)
        """
        # Keep computation on GPU by using torch operations
        return 1 - torch.prod(1 - offensive_probs)

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model and calculate metrics.
        Optimized to minimize CPU-GPU transfers.
        """
        true_positives = false_positives = false_negatives = true_negatives = 0
        coverage = abstain = 0
        total_samples = len(data_loader.dataset)
        
        print(f"Starting evaluation on {total_samples} samples...")
        
        with torch.no_grad():
            for batch_idx, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(data_loader):
                inputs = {
                    'input_ids': batch_input_ids,  # Already on correct device from DataLoader
                    'attention_mask': batch_attention_mask
                }
                
                outputs = self.model(**inputs)
                offensive_probs = torch.sigmoid(outputs.logits)  # Keep as tensor, don't convert to numpy
                
                for j in range(len(batch_input_ids)):
                    # Perform comparisons on CPU only when necessary
                    prob_offensive = self._uncertainty_propagation(offensive_probs[j])
                    if prob_offensive >= self.config.decision_threshold:
                        predicted_label = 1
                        coverage += 1
                    elif (1 - prob_offensive) >= self.config.decision_threshold:
                        predicted_label = 0
                        coverage +=1
                    else:
                        abstain += 1
                        continue
                    
                    actual_label = batch_labels[j].item()  # Single item conversion to CPU
                    
                    if predicted_label == 1 and actual_label == 1:
                        true_positives += 1
                    elif predicted_label == 1 and actual_label == 0:
                        false_positives += 1
                    elif predicted_label == 0 and actual_label == 1:
                        false_negatives += 1
                    elif predicted_label == 0 and actual_label == 0:
                        true_negatives += 1
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Processed {(batch_idx + 1) * self.config.batch_size}/{total_samples} samples")
                    print(f"Current stats: TP={true_positives}, FP={false_positives}, FN={false_negatives}, TN={true_negatives}")
                    print(f"Predicted samples:{coverage}, abstained samples={abstain}")
        
        coverage = coverage / total_samples
        metrics = self._calculate_metrics(true_positives, false_positives, false_negatives, true_negatives, coverage)
        return metrics

    def _calculate_metrics(self, tp: int, fp: int, fn: int, tn: int, coverage: int) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "coverage": coverage,
            "true positive": tp,
            "true negative": tn,
            "false positive": fp,
            "false negative": fn
        }

class MetricsLogger:
    """Handles saving and loading of model evaluation metrics."""

    def __init__(self, base_dir: str = "metrics_logs"):
        """
        Initialize metrics logger.

        Args:
            base_dir: Directory to store metrics logs
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Create subdirectories for different log types
        self.metrics_dir = self.base_dir / "performance_metrics"
        self.metrics_dir.mkdir(exist_ok=True)

    def save_metrics(self, 
                    metrics: Dict[str, float],
                    model_name: str,
                    dataset_name: str,
                    additional_info: Optional[Dict] = None) -> str:
        """
        Save metrics with metadata to JSON file.
        
        Args:
            metrics: Dictionary of metric names and values
            model_name: Name of the model used
            dataset_name: Name of the dataset evaluated
            additional_info: Optional additional metadata
            
        Returns:
            str: Path to saved metrics file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_data = {
            "timestamp": timestamp,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metrics": metrics
        }
        
        if additional_info:
            metrics_data.update(additional_info)
            threshold = additional_info.get("config", {}).get("decision_threshold", "")
            threshold_str = f"{threshold:g}"
            run = additional_info.get("config", {}).get("run", "")
            run_str = f"{run:g}"
        
        filename = f"{model_name}_{dataset_name}_{threshold_str}_{run_str}.json"
        filepath = self.metrics_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=4)
            
        return str(filepath)

    def load_metrics(self, model_name: Optional[str] = None,
                     dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load metrics into DataFrame for analysis.

        Args:
            model_name: Optional filter for specific model
            dataset_name: Optional filter for specific dataset

        Returns:
            pd.DataFrame: DataFrame containing metrics history
        """
        import pandas as pd  # Import here to avoid unnecessary dependency in other scripts
        metrics_list = []

        for filepath in self.metrics_dir.glob("*.json"):
            with open(filepath) as f:
                data = json.load(f)

                # Apply filters if specified
                if model_name and data["model_name"] != model_name:
                    continue
                if dataset_name and data["dataset_name"] != dataset_name:
                    continue

                # Flatten metrics dictionary for DataFrame
                flat_data = {
                    "timestamp": data["timestamp"],
                    "model_name": data["model_name"],
                    "dataset_name": data["dataset_name"]
                }
                flat_data.update({f"metric_{k}": v for k, v in data["metrics"].items()})
                metrics_list.append(flat_data)

        return pd.DataFrame(metrics_list)
