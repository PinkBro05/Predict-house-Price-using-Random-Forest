import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Import all the plotting file
from plot_histograms import plot_histograms
from plot_heatmap import plot_heatmap

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("housing_feature_dataset.csv")

# Data preprocessing
# Drop the 'Listing ID', PropType, Address column as it's not necessary for analysis
data = data.drop(['Address', 'Listing ID', 'PropType'], axis=1)

# Initialize LabelEncoders
suburb_encoder = LabelEncoder()
status_encoder = LabelEncoder()
re_agency_encoder = LabelEncoder()

# Convert categorical columns to numerical values
data['Suburb'] = suburb_encoder.fit_transform(data['Suburb'])
data['Status'] = status_encoder.fit_transform(data['Status'])
data['RE Agency'] = re_agency_encoder.fit_transform(data['RE Agency'])

data['Price'] = np.log(data['Price'])
# Calculate the property age
data['Property Age'] = 2024 - data['Built Year']  # Assuming data is from 2024
# Convert CBD Distance into bands (Near, Moderate, Far)
#data['Distance Band'] = pd.cut(data['CBD Distance'], bins=[0, 5, 15, np.inf], labels=['Near', 'Moderate', 'Far'])

# Define the features and target
features = ['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Property Age', 'Suburb', 'RE Agency', 'Status']
X = data[features]
y = data['Price']

# Scaling for data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Plot graph
#plot_histograms(data)
# plot_heatmap(data)

# Initialize the base model
model = RandomForestRegressor()

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate the performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print performance metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R^2 Score:", r2)

# Prepare new data point (example)
suburb_code = suburb_encoder.transform(['Doncaster'])[0]
re_agency_code = re_agency_encoder.transform(['Barry Plant'])[0]
status_code = status_encoder.transform(['S'])[0]

new_data = pd.DataFrame({
    'CBD Distance': [13.9],        # Replace with your value
    'Bedroom': [3],              # Replace with your value
    'Bathroom': [2],             # Replace with your value
    'Car-Garage': [2],           # Replace with your value
    'Landsize': [606],           # Replace with your value
    'Building Area': [252.5],      # Replace with your value
    'Property Age': [54],         # Replace with your value
    'Suburb': [suburb_code],      # Encode the suburb
    'RE Agency': [re_agency_code],  # Encode the RE Agency
    'Status': [status_code]
})

# Apply the same scaling
new_data_scaled = scaler.transform(new_data)

# Predict
predicted_price = model.predict(new_data_scaled)

# Convert back from log scale if necessary
predicted_price = np.exp(predicted_price)

print(f"Predicted Price: ${predicted_price[0]:,.2f}")