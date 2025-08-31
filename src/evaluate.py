import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from preprocessing import load_data

def evaluate():
    X, y, _, _ = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("../models/best_model.pkl")
    y_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Validation Performance -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    evaluate()
