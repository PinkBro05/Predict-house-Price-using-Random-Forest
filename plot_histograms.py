import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(data, features):
    # Calculate the number of rows and columns needed for the subplots
    num_features = len(features)
    num_rows = (num_features // 4) + (1 if num_features % 4 != 0 else 0)  # Create rows dynamically

    # Create a single figure with multiple subplots
    plt.figure(figsize=(16, num_rows * 4))  # Adjust height based on the number of rows

    for i, feature in enumerate(features):
        plt.subplot(num_rows, 4, i + 1)  # Create subplots dynamically based on feature count
        plt.hist(data[feature], bins=30, edgecolor='black')
        plt.title(feature)

    # Adjust layout to prevent overlapping
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)

    plt.show()