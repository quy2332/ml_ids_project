import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tabulate import tabulate

# Hide warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths
cleaned_dir = '../data/cleaned_datasets'
output_dir = '../data/knn_tuned_results'
os.makedirs(output_dir, exist_ok=True)

# Get all cleaned CSV files
cleaned_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]

for file in cleaned_files:
    print(f"\nProcessing file (Optimized KNN - 40 features): {file}")

    df = pd.read_csv(os.path.join(cleaned_dir, file))

    if 'Label' not in df.columns:
        print(f"Skipping {file}: No Label column.")
        continue

    X = df.drop('Label', axis=1)
    y = df['Label']

    # Encode attack labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize + feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=40)
    X_selected = selector.fit_transform(X_scaled, y_encoded)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Optimized KNN model
    knn_model = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        p=1,  # Manhattan distance
        n_jobs=-1
    )
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    # Per-class evaluation
    unique_attacks = np.unique(y_test)
    results = []

    for attack_code in unique_attacks:
        attack_name = label_encoder.inverse_transform([attack_code])[0]
        indices = (y_test == attack_code)
        acc = accuracy_score(y_test[indices], y_pred[indices])
        recall = recall_score(y_test[indices], y_pred[indices], average='macro', zero_division=0)
        precision = precision_score(y_test[indices], y_pred[indices], average='macro', zero_division=0)
        f1 = f1_score(y_test[indices], y_pred[indices], average='macro', zero_division=0)
        results.append((attack_name, round(acc, 4), round(recall, 4), round(precision, 4), round(f1, 4)))

    # Save results
    results_df = pd.DataFrame(results, columns=['Attack Type', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))
    results_df.to_csv(os.path.join(output_dir, f"knn_tuned_{file.replace('.csv', '')}_metrics.csv"), index=False)

