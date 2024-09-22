import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
#plot graph
from plot_heatmap import plot_heatmap
from plot_histograms import plot_histograms
# Load the data
data = pd.read_csv("house_price_prediction/housing_feature_dataset.csv")

# Data preprocessing
data = data.drop(['Address', 'Listing ID', 'PropType', 'RE Agency', 'Status'], axis=1)

# Initialize LabelEncoders
suburb_encoder = LabelEncoder()

# Convert categorical columns to numerical values
data['Suburb'] = suburb_encoder.fit_transform(data['Suburb'])

# Calculate property age
data['Property Age'] = 2024 - data['Built Year']

# Log transform the target variable to reduce the effect of outliers
data['Price'] = np.log(data['Price'])

# ---- Feature Engineering ---- #

# Add new feature: Building Area to Landsize Ratio
data['Area-to-Landsize Ratio'] = data['Building Area'] / data['Landsize']
data['Area-to-Landsize Ratio'] = data['Area-to-Landsize Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Add new feature: Total Rooms (Bedroom + Bathroom)
data['Total Rooms'] = data['Bedroom'] + data['Bathroom']

# Add interaction feature: Bedroom * Bathroom (capturing the combined effect)
data['Bedroom*Bathroom'] = data['Bedroom'] * data['Bathroom']

# Handle outliers by capping Landsize and Building Area (remove extremely high values)
data['Landsize'] = np.clip(data['Landsize'], 0, data['Landsize'].quantile(0.99))
data['Building Area'] = np.clip(data['Building Area'], 0, data['Building Area'].quantile(0.99))

#data visualization
#plot_heatmap(data)
features_to_plot = ['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize', 'Building Area', 'Price']
plot_histograms(data, features_to_plot)
# ---- Feature Selection ---- #
features = ['CBD Distance', 'Bedroom', 'Bathroom', 'Car-Garage', 'Landsize',
            'Building Area', 'Property Age', 'Suburb', 'Area-to-Landsize Ratio',
            'Total Rooms', 'Bedroom*Bathroom']

X = data[features]
y = data['Price']

# Scaling for data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=1)

# Set up the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 150],  # Number of trees in the forest
    'min_samples_split': [5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [3, 5],  # Minimum number of samples required at each leaf node
    'max_features': ['sqrt', 'log2']
}

# Initialize the Random Forest Regressor model
model = RandomForestRegressor()

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)

# Fit the model with GridSearchCV
grid_search.fit(X_train, y_train)

# Retrieve the best parameters from the grid search
best_params = grid_search.best_params_

# Initialize the Random Forest Regressor with the best parameters
best_model = grid_search.best_estimator_

# Predict on the test and train data with the best model
y_test_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)

# Evaluate on the test data
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

# Evaluate on the training data
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Print the best parameters from GridSearchCV
print("Best Parameters from Grid Search:", best_params)

# Print performance metrics for test data
print("\nTest Data Evaluation:")
print("Mean Absolute Error (MAE):", test_mae)
print("Mean Squared Error (MSE):", test_mse)
print("Root Mean Squared Error (RMSE):", test_rmse)
print("R^2 Score:", test_r2)

# Print performance metrics for training data
print("\nTraining Data Evaluation:")
print("Mean Absolute Error (MAE):", train_mae)
print("Mean Squared Error (MSE):", train_mse)
print("Root Mean Squared Error (RMSE):", train_rmse)
print("R^2 Score:", train_r2)

# Function to receive new data input from the user
def get_user_input():
    cbd_distance = float(input("Enter CBD Distance (km): "))
    bedroom = int(input("Enter number of Bedrooms: "))
    bathroom = int(input("Enter number of Bathrooms: "))
    car_garage = int(input("Enter number of Car Garages: "))
    landsize = float(input("Enter Landsize (sqm): "))
    building_area = float(input("Enter Building Area (sqm): "))
    property_age = int(input("Enter Property Age (years): "))
    suburb_name = input("Enter Suburb Name: ")

    # Convert the suburb name to code using the encoder
    suburb_code = suburb_encoder.transform([suburb_name])[0]

    # Feature engineering for new input data
    area_to_landsize_ratio = building_area / landsize if landsize != 0 else 0
    total_rooms = bedroom + bathroom
    bedroom_bathroom_interaction = bedroom * bathroom

    # Prepare the input data as a DataFrame
    new_data = pd.DataFrame({
        'CBD Distance': [cbd_distance],
        'Bedroom': [bedroom],
        'Bathroom': [bathroom],
        'Car-Garage': [car_garage],
        'Landsize': [landsize],
        'Building Area': [building_area],
        'Property Age': [property_age],
        'Suburb': [suburb_code],
        'Area-to-Landsize Ratio': [area_to_landsize_ratio],
        'Total Rooms': [total_rooms],
        'Bedroom*Bathroom': [bedroom_bathroom_interaction]
    })

    return new_data

# Call the function to get user input
new_data = get_user_input()

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Predict the price based on user input
predicted_price = best_model.predict(new_data_scaled)

# Convert back from log scale
predicted_price = np.exp(predicted_price)

print(f"\nPredicted Price: ${predicted_price[0]:,.2f}")

# Show plots if needed
plt.show()