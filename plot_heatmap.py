import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(data):
    # Move 'Price' to the last column
    cols = [col for col in data.columns if col != 'Price'] + ['Price']
    data_reordered = data[cols]

    plt.figure(figsize=(10, 8))
    correlation_matrix = data_reordered.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()
