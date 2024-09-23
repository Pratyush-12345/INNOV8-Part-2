import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib

# Load the data
data = pd.read_csv('soldiers_data.csv')

# Separate features and target
X = data.drop(['soldier_id', 'past_betrayals'], axis=1)
y = data['past_betrayals']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical columns
numeric_features = ['wealth_level', 'promise_of_wealth', 'reputation', 'years_of_service',
                    'respect_from_peers', 'temptation_level', 'influence_of_others',
                    'mental_resilience', 'stress_levels']
categorical_features = ['rank']

# Create preprocessing steps
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Apply SMOTE to the preprocessed training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Define models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Define hyperparameters for grid search
param_grid = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

# Train and evaluate models
best_model = None
best_score = 0

for name, model in models.items():
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Predict on test set
    y_pred = grid_search.predict(X_test_processed)
    y_pred_proba = grid_search.predict_proba(X_test_processed)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print("\n")

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = grid_search

# Save the best model
joblib.dump(best_model, 'best_model.joblib')

print("Best model saved.")

# Feature importance for Random Forest (if applicable)
if hasattr(best_model.best_estimator_, 'feature_importances_'):
    importances = best_model.best_estimator_.feature_importances_
    feature_names = numeric_features + [f'rank_{cat}' for cat in preprocessor.transformers_[1][1].categories_[0][1:]]
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_imp)

    # Save feature importance
    feature_imp.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to feature_importance.csv")

