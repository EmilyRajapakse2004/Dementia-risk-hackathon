# modules/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df, output_dir="plots"):
    # Create folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Feature distributions (histograms)
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(hist_path)
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(corr_path)
    plt.close()

    # 3. Class balance
    if 'dementia_label' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='dementia_label', data=df)
        class_path = os.path.join(output_dir, "class_balance.png")
        plt.savefig(class_path)
        plt.close()
    else:
        print("Warning: 'dementia_label' column not found. Skipping class balance plot.")

    print(f"EDA plots saved in '{output_dir}' folder.")


if __name__ == "__main__":
    # Get project root dynamically
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "dementia_data.csv")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        visualize_data(df, output_dir=os.path.join(BASE_DIR, "plots"))