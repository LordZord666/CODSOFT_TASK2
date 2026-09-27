import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


TRAIN_PATH = "data/fraudTrain.csv"
TEST_PATH = "data/fraudTest.csv"
MODEL_PATH = "fraud_classifier.joblib"

# Columns that are just identifiers or raw text with no direct predictive
# value once we've engineered better features from them (below).
COLUMNS_TO_DROP = [
    "Unnamed: 0", "cc_num", "first", "last", "street", "city", "state",
    "zip", "job", "merchant", "trans_num", "unix_time", "dob",
    "trans_date_trans_time", "lat", "long", "merch_lat", "merch_long",
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two lat/long points.

    Fraudulent transactions often happen far from the cardholder's home
    location, so this distance is one of the most useful engineered
    features for this dataset.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction/customer columns into model-ready features."""
    df = df.copy()

    trans_time = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])

    df["age"] = ((trans_time - dob).dt.days / 365.25).astype(int)
    df["trans_hour"] = trans_time.dt.hour
    df["trans_day_of_week"] = trans_time.dt.dayofweek  # 0=Monday

    df["distance_km"] = haversine_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load a raw fraud CSV and apply feature engineering."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    df = df.dropna()
    return df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Encode categoricals (fit on train, apply to test) and scale numerics."""
    y_train = train_df["is_fraud"]
    y_test = test_df["is_fraud"]
    X_train = train_df.drop(columns=["is_fraud"])
    X_test = test_df.drop(columns=["is_fraud"])

    # category and gender are text - encode them as numbers.
    # Fit the encoder on TRAIN only, then reuse it on test, so test never
    # "leaks" information into how categories are numbered.
    for col in X_train.select_dtypes(exclude=["number", "bool"]).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Any category in test not seen during training gets mapped to -1
        # instead of crashing.
        known = set(le.classes_)
        X_test[col] = X_test[col].astype(str).apply(lambda v: v if v in known else "unknown")
        if "unknown" not in known:
            le.classes_ = np.append(le.classes_, "unknown")
        X_test[col] = le.transform(X_test[col])

    numeric_cols = ["amt", "city_pop", "age", "distance_km"]
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test


def build_and_evaluate(X_train, X_test, y_train, y_test, model_name: str):
    """Train a classifier and print evaluation metrics focused on fraud detection."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }
    model = models[model_name]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    print(f"\n=== Results: {model_name} ===")
    print("Classification Report (0=legitimate, 1=fraud):")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "fraud"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    if model_name == "random_forest":
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        print("\nTop 5 most important features:")
        print(importances.sort_values(ascending=False).head(5))

    return model


if __name__ == "__main__":
    print("Loading and engineering features for training data...")
    train_df = load_data(TRAIN_PATH)
    print("Loading and engineering features for test data...")
    test_df = load_data(TEST_PATH)

    fraud_count = train_df["is_fraud"].sum()
    total = len(train_df)
    print(f"\nTraining set: {total} transactions ({fraud_count} fraud, {total - fraud_count} legitimate)")
    print(f"Fraud rate: {fraud_count / total * 100:.3f}%")

    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df)

    model = None
    for name in ["logistic_regression", "random_forest"]:
        model = build_and_evaluate(X_train, X_test, y_train, y_test, model_name=name)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
