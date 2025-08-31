import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import load_data, create_preprocessor

def train_and_evaluate():
    # Load
    X, y, test, test_ids = load_data()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessor
    preprocessor = create_preprocessor(X)

    # Models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        # "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                   ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

    import os
    os.makedirs("models", exist_ok=True)  # ✅ create folder if missing
    joblib.dump(best_model, "../models/best_model.pkl")
    print("✅ Best model saved to models/best_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
