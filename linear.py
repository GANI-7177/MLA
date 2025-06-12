# Step 1: Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Load the dataset
csv_path = r"C:\Users\GANESH\Downloads\house_prices_sample.csv"
df = pd.read_csv(csv_path)

# Step 3: Prepare features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# Step 4: Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print("✅ Coefficients:", model.coef_)
print("✅ Intercept:", model.intercept_)
print("✅ Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("✅ R² Score:", r2_score(y_test, y_pred))

# Step 7: Visualize predictions vs actual
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
