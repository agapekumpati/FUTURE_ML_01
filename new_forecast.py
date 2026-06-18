import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1. CREATE TIME SERIES DATA
# ==============================
months = pd.date_range(start="2024-01-01", periods=30, freq="ME")

sales = np.array([
    120,130,125,140,150,160,
    170,165,180,190,200,210,
    220,235,230,245,260,270,
    285,280,295,310,320,340,
    350,360,375,390,405,420
])

df = pd.DataFrame({"date": months, "sales": sales})

# ==============================
# 2. FEATURE ENGINEERING
# ==============================
df["time_index"] = np.arange(len(df))

df["month"] = df["date"].dt.month

df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

# ==============================
# 3. TRAIN / TEST SPLIT
# ==============================
train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

features = ["time_index", "sin_month", "cos_month"]

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]

# ==============================
# 4. MODEL TRAINING
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 5. PREDICTIONS
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 6. EVALUATION
# ==============================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMODEL PERFORMANCE")
print("------------------")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))

# ==============================
# 7. FUTURE FORECAST (12 MONTHS)
# ==============================
future_dates = pd.date_range(start=df["date"].iloc[-1], periods=13, freq="ME")[1:]

future_df = pd.DataFrame({"date": future_dates})

future_df["time_index"] = np.arange(len(df), len(df) + 12)
future_df["month"] = future_df["date"].dt.month

future_df["sin_month"] = np.sin(2 * np.pi * future_df["month"] / 12)
future_df["cos_month"] = np.cos(2 * np.pi * future_df["month"] / 12)

future_X = future_df[features]
future_pred = model.predict(future_X)

# ==============================
# 8. CONFIDENCE INTERVAL
# ==============================
residuals = y_train - model.predict(X_train)
std_dev = np.std(residuals)

upper = future_pred + 1.96 * std_dev
lower = future_pred - 1.96 * std_dev

# ==============================
# 9. VISUALIZATION + SAVE GRAPH
# ==============================
plt.figure(figsize=(14,6))

# Actual sales
plt.plot(df["date"], df["sales"], label="Actual Sales", marker="o")

# Test predictions
plt.plot(test["date"], y_pred, label="Test Predictions", marker="o")

# Future forecast
plt.plot(future_dates, future_pred, label="Forecast", linestyle="dashed", marker="o")

# Confidence interval
plt.fill_between(future_dates, lower, upper, color="gray", alpha=0.2, label="Confidence Interval")

plt.title("Industry-Level Sales Forecasting (Trend + Seasonality Model)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# ✅ SAVE IMAGE FILE
plt.savefig("forecast_graph.png", dpi=300, bbox_inches="tight")

# SHOW PLOT
plt.show()