import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Sort by date
df = df.sort_values('Order Date')

# Group sales by date
sales_data = df.groupby('Order Date')['Sales'].sum()

# Plot sales trend
plt.figure(figsize=(12,6))
plt.plot(sales_data)
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()

# Save graph
plt.savefig("forecast_graph.png")

# Show graph
plt.show()

print("Graph created successfully!")