# House-price-predicition
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Create sample dataset
data = {
    "area": [1200, 1500, 1800, 2000, 2500, 3000],
    "bedrooms": [2, 3, 3, 4, 4, 5],
    "bathrooms": [1, 2, 2, 3, 3, 4],
    "price": [3000000, 4500000, 5000000, 6500000, 8000000, 10000000]
}
df = pd.DataFrame(data)
# Split features and target
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Test prediction
y_pred = model.predict(X_test)
# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))
# New house prediction
new_house = [[2200, 3, 2]]
predicted_price = model.predict(new_house)
print("Predicted House Price:", int(predicted_price[0]))

