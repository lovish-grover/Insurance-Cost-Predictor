import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def extract_date_features(df):
    """Extract features from Policy Start Date"""
    df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")
    df["Policy_Year"] = df["Policy Start Date"].dt.year
    df["Policy_Month"] = df["Policy Start Date"].dt.month
    df["Policy_DayOfWeek"] = df["Policy Start Date"].dt.dayofweek
    df.drop(columns=["Policy Start Date"], inplace=True)
    return df

def load_data(train_path="../data/train.csv", test_path="../data/test.csv"):
    """Load train and test datasets"""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Save test ids
    test_ids = test["id"]

    # Drop ID
    train = train.drop(columns=["id"])
    test = test.drop(columns=["id"])

    # Extract datetime features
    train = extract_date_features(train)
    test = extract_date_features(test)

    # Split X, y
    X = train.drop(columns=["Premium Amount"])
    y = train["Premium Amount"]

    return X, y, test, test_ids

def create_preprocessor(X):
    """Build preprocessing pipeline"""
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )
    return preprocessor
