import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Sort by date
df = df.sort_values("Order Date")

# Create monthly sales summary
monthly_sales = (
    df.set_index("Order Date")
      .resample("ME")["Sales"]
      .sum()
      .reset_index()
)

# Create numeric feature
monthly_sales["Month_Number"] = range(len(monthly_sales))

X = monthly_sales[["Month_Number"]]
y = monthly_sales["Sales"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
monthly_sales["Predicted_Sales"] = model.predict(X)

# Evaluation
mae = mean_absolute_error(y, monthly_sales["Predicted_Sales"])
mse = mean_squared_error(y, monthly_sales["Predicted_Sales"])
r2 = r2_score(y, monthly_sales["Predicted_Sales"])

print("\nModel Evaluation")
print("MAE =", round(mae, 2))
print("MSE =", round(mse, 2))
print("R2 Score =", round(r2, 4))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    label="Actual Sales"
)

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Predicted_Sales"],
    label="Predicted Sales"
)

plt.title("Sales Forecasting - Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig("forecast_graph.png")

# Show graph
plt.show()

print("\nForecast completed successfully!")