import joblib
import pandas as pd
from preprocessing import load_data

def predict_submission():
    _, _, test, test_ids = load_data()
    model = joblib.load("../models/best_model.pkl")

    preds = model.predict(test)

    submission = pd.DataFrame({"id": test_ids, "Premium Amount": preds})
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission file saved as submission.csv")

if __name__ == "__main__":
    predict_submission()
