import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# 1. Load dataset
df = pd.read_csv("measures_v2.csv")

# 2. Reduce dataset size for speed
df = df.sample(n=1000, random_state=42)

# 3. Target column (change if needed)
target_column = "profile_id"
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset columns: {df.columns.tolist()}")

X = df.drop(columns=[target_column])
y = df[target_column]

# 4. Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Train model
model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

# 8. Save model, scaler, and feature names
joblib.dump(model, "model.save")
joblib.dump(scaler, "transform.save")
joblib.dump(list(X.columns), "features.save")
print("✅ Model, scaler, and features saved successfully!")
