import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(data):
    # Define the features for plotting
    features = ['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Suburb', 'RE Agency', 'Status', 'Price']

    # Create a single figure with multiple subplots
    plt.figure(figsize=(16, 14))

    # Plot histograms manually without using a loop
    plt.subplot(4, 4, 1)
    plt.hist(data['CBD Distance'], bins=30, edgecolor='black')
    plt.title('CBD Distance')

    plt.subplot(4, 4, 2)
    plt.hist(data['Bedroom'], bins=30, edgecolor='black')
    plt.title('Bedroom')

    plt.subplot(4, 4, 3)
    plt.hist(data['Bathroom'], bins=30, edgecolor='black')
    plt.title('Bathroom')

    plt.subplot(4, 4, 4)
    plt.hist(data['Car-Garage'], bins=30, edgecolor='black')
    plt.title('Car-Garage')

    plt.subplot(4, 4, 5)
    plt.hist(data['Landsize'], bins=30, edgecolor='black')
    plt.title('Landsize')

    plt.subplot(4, 4, 6)
    plt.hist(data['Building Area'], bins=30, edgecolor='black')
    plt.title('Building Area')

    plt.subplot(4, 4, 7)
    plt.hist(data['Property Age'], bins=30, edgecolor='black')
    plt.title('Property Age')

    plt.subplot(4, 4, 8)
    plt.hist(data['Suburb'], bins=30, edgecolor='black')
    plt.title('Suburb')

    plt.subplot(4, 4, 9)
    plt.hist(data['RE Agency'], bins=30, edgecolor='black')
    plt.title('RE Agency')

    plt.subplot(4, 4, 10)
    plt.hist(data['Status'], bins=30, edgecolor='black')
    plt.title('Status')

    plt.subplot(4, 4, 11)
    plt.hist(data['Price'], bins=30, edgecolor='black')
    plt.title('Price')

    # Adjust layout to prevent overlapping
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)

    plt.show()
