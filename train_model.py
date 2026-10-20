"""Melatih model regresi linear dan menyimpan model dengan Joblib."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parent
data = pd.read_csv(ROOT / "data_belajar.csv")
X = data[["jam_belajar"]]
y = data["nilai_ujian"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Intercept: {model.intercept_:.3f}")
print(f"Koefisien jam belajar: {model.coef_[0]:.3f}")
print(f"R-squared (test): {r2_score(y_test, pred):.3f}")
print(f"RMSE (test): {mean_squared_error(y_test, pred) ** 0.5:.3f}")
joblib.dump(model, ROOT / "model_regresi.joblib")
print("Model tersimpan di model_regresi.joblib")
