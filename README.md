# Xernian Soldier Betrayal Prediction System

## Overview
![alt text](<Screenshot 2024-09-23 221647.png>)
![alt text](<Screenshot 2024-09-23 221706.png>)

This project aims to predict the likelihood of a soldier betraying the Xernian clan based on various factors. The system is built using machine learning models and provides a user-friendly interface for predictions.

## Contents

1. [Model Training](#model-training)
2. [Streamlit Application](#streamlit-application)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [License](#license)

## Model Training

The model training script utilizes:
- **Pandas** for data manipulation
- **Scikit-learn** for machine learning
- **Imbalanced-learn (SMOTE)** for handling class imbalance
- **Joblib** for saving the trained model

### Script Overview

1. **Data Loading**: Loads the training data from a CSV file.
2. **Preprocessing**: Standardizes numerical features and one-hot encodes categorical features.
3. **Modeling**: Trains two classifiers (Random Forest and Gradient Boosting) and performs hyperparameter tuning using GridSearchCV.
4. **Evaluation**: Evaluates models using metrics like accuracy, precision, recall, F1-score, and ROC AUC.
5. **Feature Importance**: Computes and saves feature importance for interpretation.

### Dependencies

Ensure you have the following Python packages installed:

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib
```

### Running the Training Script

Run the training script as follows:

```bash
python train_model.py
```

This will generate the trained model file `best_model.joblib` and feature importance file `feature_importance.csv`.

## Streamlit Application

The Streamlit application provides an interactive interface for users to input soldier information and get predictions about betrayal likelihood.

### Features

- User input for various soldier attributes
- Prediction display including betrayal likelihood and risk level
- Visualization of feature importance affecting the prediction

### Dependencies

Ensure you have the following Python packages installed:

```bash
pip install streamlit matplotlib joblib pandas scikit-learn
```

### Running the Streamlit App

Run the Streamlit app using the command:

```bash
streamlit run app.py
```

Open your browser and go to `http://localhost:8501` to access the application.

## Usage

1. Input soldier attributes such as wealth level, rank, reputation, etc.
2. Click on "Predict Betrayal Likelihood" to get predictions.
3. View the top factors influencing the prediction in a bar graph.

## Results

The system provides a likelihood of betrayal and classifies the risk level into three categories: Low, Moderate, and High. The feature importance visualization helps in understanding the key factors affecting predictions.

