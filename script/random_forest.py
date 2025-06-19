import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tabulate import tabulate

# Hide warnings (for clean CLI output)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
cleaned_dir = '../data/cleaned_datasets'
output_dir = '../data/rf_tuned_results'
os.makedirs(output_dir, exist_ok=True)

# Get all cleaned CSV files
cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

# Main loop per dataset
for file in cleaned_files:
    print(f"\nProcessing file: {file}")

    df = pd.read_csv(os.path.join(cleaned_dir, file))

    if 'Label' not in df.columns:
        print(f"Skipping {file}: No Label column.")
        continue

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest hyperparameter search space
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 15, 20, 25],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }

    # Use stratified 3-fold CV for faster training
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=cv_strategy,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Train and select best model
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best RF Parameters:", search.best_params_)

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    unique_attacks = np.unique(y_test)
    results = []

    for attack in unique_attacks:
        attack_indices = (y_test == attack)
        attack_y_true = y_test[attack_indices]
        attack_y_pred = y_pred[attack_indices]

        acc = accuracy_score(attack_y_true, attack_y_pred)
        recall = recall_score(attack_y_true, attack_y_pred, average='macro', zero_division=0)
        precision = precision_score(attack_y_true, attack_y_pred, average='macro', zero_division=0)
        f1 = f1_score(attack_y_true, attack_y_pred, average='macro', zero_division=0)

        results.append((attack, round(acc, 4), round(recall, 4), round(precision, 4), round(f1, 4)))

    # Save and display results
    results_df = pd.DataFrame(results, columns=['Attack Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))

    results_df.to_csv(
        os.path.join(output_dir, f"rf_optimized_{file.replace('.csv', '')}_metrics.csv"),
        index=False
    )

