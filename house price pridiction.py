import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create a Synthetic Dataset
# Generating synthetic data for house prices
np.random.seed(42)  # For reproducibility
size = np.random.randint(800, 4000, size=100)  # Size in square feet
num_rooms = np.random.randint(1, 6, size=100)  # Number of rooms
num_bathrooms = np.random.randint(1, 4, size=100)  # Number of bathrooms
location = np.random.choice(['Downtown', 'Suburbs', 'Uptown'], size=100)  # Locations

# Create a DataFrame
data = pd.DataFrame({
    'size': size,
    'num_rooms': num_rooms,
    'num_bathrooms': num_bathrooms,
    'location': location
})

# Generate prices based on size, number of rooms, and bathrooms with some noise
data['price'] = (data['size'] * 150) + (data['num_rooms'] * 20000) + (data['num_bathrooms'] * 10000) + np.random.normal(0, 20000, size=100)

# Step 3: Prepare the Data
# Convert categorical variable 'location' to dummy variables
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Define features (X) and target variable (y)
X = data.drop('price', axis=1)  # Features
y = data['price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Visualize Results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()