import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    # Feature distributions
    df.hist(figsize=(10,8))
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()

    # Class balance
    sns.countplot(x='dementia_label', data=df)
    plt.show()