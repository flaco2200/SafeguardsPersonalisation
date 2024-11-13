# compare_models.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(metrics_dir: str, dataset_name: str) -> pd.DataFrame:
    """
    Load metrics from JSON files into a DataFrame.

    Args:
        metrics_dir (str): Directory containing the metrics JSON files.
        model_name (str): Name of the model to filter metrics.

    Returns:
        pd.DataFrame: DataFrame containing all metrics for the specified model.
    """
    metrics_list = []

    # Iterate over all JSON files in the directory
    for filepath in Path(metrics_dir).glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)

            # Filter by model name
            if data.get("dataset_name") != dataset_name:
                continue

            # Extract metrics
            metrics = data.get("metrics", {})
            additional_info = data.get("config", {})
            threshold = additional_info.get("decision_threshold", None)
            run = additional_info.get("run", None)

            # Include threshold and run in the data
            flat_data = {
                "threshold": float(threshold) if threshold is not None else None,
                "run": int(run) if run is not None else None,
                "model_name": data.get("model_name"),
                "dataset_name": data.get("dataset_name"),
                "timestamp": data.get("timestamp"),
            }

            # Add all metrics
            flat_data.update(metrics)

            # Add to list
            metrics_list.append(flat_data)

    # Create DataFrame
    df = pd.DataFrame(metrics_list)

    # Sort by threshold or coverage for plotting
    df.sort_values(by=["coverage"], inplace=True)

    return df

def plot_metrics(df_concepts: pd.DataFrame, df_direct: pd.DataFrame, save_dir: str, race: str):
    """
    Generate and save plots comparing the two models.

    Args:
        df_concepts (pd.DataFrame): DataFrame for the concepts model.
        df_direct (pd.DataFrame): DataFrame for the direct model.
        save_dir (str): Directory to save the plots.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Plot F1 Score vs. Coverage
    plt.figure(figsize=(8, 6))
    plt.plot(df_concepts["coverage"], df_concepts["f1_score"], label="Concepts Model", marker='o')
    plt.plot(df_direct["coverage"], df_direct["f1_score"], label="Direct Model", marker='o')
    plt.xlabel("Coverage")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Coverage")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"f1_score_vs_coverage_{race}.png"))
    plt.close()

    # Plot Precision vs. Coverage
    plt.figure(figsize=(8, 6))
    plt.plot(df_concepts["coverage"], df_concepts["precision"], label="Concepts Model", marker='o')
    plt.plot(df_direct["coverage"], df_direct["precision"], label="Direct Model", marker='o')
    plt.xlabel("Coverage")
    plt.ylabel("Precision")
    plt.title("Precision vs. Coverage")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"precision_vs_coverage_{race}.png"))
    plt.close()

    # Plot Recall vs. Coverage
    plt.figure(figsize=(8, 6))
    plt.plot(df_concepts["coverage"], df_concepts["recall"], label="Concepts Model", marker='o')
    plt.plot(df_direct["coverage"], df_direct["recall"], label="Direct Model", marker='o')
    plt.xlabel("Coverage")
    plt.ylabel("Recall")
    plt.title("Recall vs. Coverage")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"recall_vs_coverage_{race}.png"))
    plt.close()

def main():
    # Directories containing metrics
    concepts_metrics_dir = "experiment_logs/performance_metrics"
    direct_metrics_dir = "experiment_logs/direct_performance_metrics"

    race = "latinx"
    dataset_name = f"{race}_test"

    # Load metrics
    df_concepts = load_metrics(concepts_metrics_dir, dataset_name)
    df_direct = load_metrics(direct_metrics_dir, dataset_name)

    # Ensure necessary fields are present
    required_fields = ["precision", "recall", "f1_score", "coverage"]
    for field in required_fields:
        if field not in df_concepts.columns or field not in df_direct.columns:
            print(f"Field '{field}' not found in metrics. Please ensure that evaluation metrics include precision, recall, f1_score, and coverage.")
            return

    # Plot and save metrics
    save_dir = "comparison_plots"
    plot_metrics(df_concepts, df_direct, save_dir, race)

    print(f"Plots saved in directory: {save_dir}")

if __name__ == "__main__":
    main()