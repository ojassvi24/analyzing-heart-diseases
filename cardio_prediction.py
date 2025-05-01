#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cardiovascular Disease Prediction Model

This script builds and evaluates machine learning models to predict cardiovascular disease
based on patient data. It implements both Logistic Regression and Random Forest classifiers.
"""

# Standard library imports
import numpy as np
import pandas as pd

# Data visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
# - Model selection
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# - Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# - Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# - Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)


def load_and_explore_data(file_path='cardio_train.csv'):
    """
    Load the dataset and perform initial exploratory data analysis.

    Args:
        file_path (str): Path to the dataset file

    Returns:
        pandas.DataFrame: The loaded dataset
    """
    # Load the dataset
    data = pd.read_csv(file_path, sep=';')

    # Inspect the first few rows
    print("Dataset columns:")
    print(data.columns)
    print("\nFirst 5 rows:")
    print(data.head())

    # Check for data types and missing values
    print("\nDataset info:")
    print(data.info())

    # Summary statistics for numerical columns
    print("\nSummary statistics:")
    print(data.describe())

    return data


def preprocess_data(data):
    """
    Preprocess the data by creating a pipeline for imputation and scaling.

    Args:
        data (pandas.DataFrame): The dataset to preprocess

    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessing_pipeline
    """
    # Define categorical and continuous features
    categorical_features = ['cholesterol', 'gluc', 'gender', 'smoke', 'alco', 'active']
    numeric_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
        ])

    # Create final preprocessing pipeline with scaling
    preprocessing_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])

    # Split data into features and target
    X = data.drop(columns=['cardio'])
    y = data['cardio']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply preprocessing
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessing_pipeline


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with hyperparameter tuning.

    Args:
        X_train (numpy.ndarray): Processed training features
        y_train (pandas.Series): Training target variable

    Returns:
        sklearn.model_selection.GridSearchCV: Trained model with best parameters
    """
    # Initialize logistic regression model
    logreg = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)

    # Define hyperparameter grid
    param_grid = {'C': [0.01, 0.1, 1, 10]}

    # Perform grid search
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print best parameters
    print(f"Best hyperparameters for Logistic Regression: {grid_search.best_params_}")

    return grid_search


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning.

    Args:
        X_train (numpy.ndarray): Processed training features
        y_train (pandas.Series): Training target variable

    Returns:
        sklearn.model_selection.RandomizedSearchCV: Trained model with best parameters
    """
    print("Starting Random Forest model training...")

    # Initialize Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': np.arange(100, 501, 100),
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 5]
    }

    # Perform randomized search
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_grid, n_iter=10,
        cv=5, scoring='accuracy', random_state=42
    )
    random_search.fit(X_train, y_train)

    print("Random Forest model fitting done.")
    print(f"Best hyperparameters for Random Forest: {random_search.best_params_}")

    return random_search


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and display performance metrics.

    Args:
        model: Trained model
        X_test (numpy.ndarray): Processed test features
        y_test (pandas.Series): Test target variable
        model_name (str): Name of the model for display purposes

    Returns:
        dict: Dictionary containing performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }

    # Print metrics
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.2f}")

    # Create and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)

    return metrics


def plot_confusion_matrix(cm, model_name):
    """
    Plot a confusion matrix for model evaluation.

    Args:
        cm (numpy.ndarray): Confusion matrix
        model_name (str): Name of the model for the plot title
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()


def main():
    """
    Main function to orchestrate the cardiovascular disease prediction workflow.
    """
    # Load and explore data
    data = load_and_explore_data()

    # Preprocess data
    X_train, X_test, y_train, y_test, _ = preprocess_data(data)

    # Train and evaluate Logistic Regression model
    logreg_model = train_logistic_regression(X_train, y_train)
    logreg_metrics = evaluate_model(logreg_model, X_test, y_test, "Logistic Regression")

    # Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Compare models
    print("\nModel Comparison:")
    print(f"Logistic Regression F1: {logreg_metrics['f1']:.2f}")
    print(f"Random Forest F1: {rf_metrics['f1']:.2f}")

    if rf_metrics['f1'] > logreg_metrics['f1']:
        print("Random Forest performs better based on F1 score.")
    else:
        print("Logistic Regression performs better based on F1 score.")


if __name__ == "__main__":
    main()
