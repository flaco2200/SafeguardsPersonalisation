# Conceptual Safeguards for Personalization in Toxic Content Detection

This repository contains code and data for training, fine-tuning, and evaluating classifiers for detecting concepts and predicting labels across different demographic groups. Below is an organized summary of the files and folders within the project.

## Folder Structure

- **comparison_plots**: Contains plots comparing various metrics of different models across racial groups and coverage levels.
- **dataset_scripts**: Scripts for preparing CSV datasets used to train and test a RoBERTa classifier.
- **experiment_logs**: Logs of test results across demographic groups, documenting classifier performance.
- **preprocessed_datasets**: Ready-to-use datasets for classifier training and testing.

## File Overview

- **Concept Detection Classifiers**
  - **train_concepts_full_dataset.py**: Trains classifiers to detect specific concepts in text using the full dataset.
  - **fine_tune_concepts_race.py**: Fine-tunes concept detection classifiers on different racial groups.

- **Direct Prediction Classifiers**
  - **train_direct_full_dataset.py**: Trains classifiers to predict labels directly from the full dataset.
  - **fine_tune_direct_race.py**: Fine-tunes direct prediction classifiers on different racial groups.

- **Testing and Comparison**
  - **test_concepts.py**: Tests concept detection classifiers to evaluate their performance.
  - **test_direct.py**: Tests direct prediction classifiers to assess their accuracy.
  - **compare_models.py**: Produces comparative performance plots for both model types across demographic groups.
