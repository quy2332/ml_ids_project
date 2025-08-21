import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tabulate import tabulate
from scipy.stats import loguniform

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
cleaned_dir = '../data/cleaned_datasets'
output_dir = '../data/mlp_tuned_results'
os.makedirs(output_dir, exist_ok=True)

# Load cleaned CSVs
cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

for file in cleaned_files:
    print(f"\nProcessing file (MLP + Light CV): {file}")
    df = pd.read_csv(os.path.join(cleaned_dir, file))

    if 'Label' not in df.columns:
        print(f"Skipping {file}: No Label column.")
        continue

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Encode string labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split for final eval (CV happens on train)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Define low-cost hyperparameter search space
    param_dist = {
        "hidden_layer_sizes": [(128, 64), (64, 32)],
        "alpha": [1e-4, 1e-3],
        "batch_size": [128, 256],
        "learning_rate_init": [0.001, 0.005],
    }

    mlp = MLPClassifier(
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        early_stopping=True,
        max_iter=100,
        n_iter_no_change=10,
        random_state=42
    )

    # 2-fold CV, 5 hyperparameter samples → 10 fits total
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_dist,
        n_iter=5,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print("Best MLP Parameters:", search.best_params_)

    # Predict on test set
    y_pred = best_model.predict(X_test)

    # Per-class metrics
    results = []
    for attack_code in np.unique(y_test):
        attack_name = label_encoder.inverse_transform([attack_code])[0]
        indices = (y_test == attack_code)
        acc = accuracy_score(y_test[indices], y_pred[indices])
        recall = recall_score(y_test[indices], y_pred[indices], average='macro', zero_division=0)
        precision = precision_score(y_test[indices], y_pred[indices], average='macro', zero_division=0)
        f1 = f1_score(y_test[indices], y_pred[indices], average='macro', zero_division=0)
        results.append((attack_name, round(acc, 4), round(recall, 4), round(precision, 4), round(f1, 4)))

    # Save + print
    results_df = pd.DataFrame(results, columns=['Attack Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))
    results_df.to_csv(
        os.path.join(output_dir, f"mlp_tuned_{file.replace('.csv', '')}_metrics.csv"),
        index=False
    )
