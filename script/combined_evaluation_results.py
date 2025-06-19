import pandas as pd
import os
from tabulate import tabulate

# Folder mappings (easy to extend later for other models if wanted)
folders_to_check = {
    'Random Forest': '../data/rf_tuned_results',
}

# Helper function to combine and print results for a given model
def process_results(model_name, results_dir):
    if not os.path.exists(results_dir):
        print(f"\nDirectory {results_dir} does not exist. Skipping {model_name}.")
        return

    result_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

    if not result_files:
        print(f"\nNo evaluation files found in {results_dir}. Skipping {model_name}.")
        return

    all_results = []
    for file in result_files:
        df = pd.read_csv(os.path.join(results_dir, file), encoding='utf-8', on_bad_lines='skip')
        df['Attack Type'] = df['Attack Type'].astype(str).str.strip().str.replace(' ', '-', regex=False)
        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Normalize attack labels: fix weird characters and spacing
    combined_df['Attack Type'] = combined_df['Attack Type'].str.strip()

    # Group and average
    grouped_df = combined_df.groupby('Attack Type', as_index=False).mean(numeric_only=True)

    # Round for cleaner output
    grouped_df[['Accuracy', 'Recall', 'Precision', 'F1-Score']] = grouped_df[
        ['Accuracy', 'Recall', 'Precision', 'F1-Score']
    ].round(4)

    grouped_df = grouped_df.sort_values('Attack Type').reset_index(drop=True)

    # Print final combined results
    print(f"\nCombined Evaluation for {model_name}:\n")
    print(tabulate(grouped_df, headers='keys', tablefmt='pretty'))


    # Can be saved into a .csv file
    # final_combined_df.to_csv(f'../data/{model_name.replace(' ', '_').lower()}_combined.csv', index=False)

# Main
if __name__ == "__main__":
    for model_name, results_dir in folders_to_check.items():
        process_results(model_name, results_dir)
