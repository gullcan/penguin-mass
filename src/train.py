from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data import load_penguins, prepare_split


FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "species",
]
NUMERIC = FEATURES[:3]


def build_model():
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        (
            "species",
            OneHotEncoder(
                drop="first",
                handle_unknown="error",
                sparse_output=False,
            ),
            ["species"],
        ),
    ])

    return Pipeline([
        ("preprocess", preprocessing),
        ("regression", LinearRegression()),
    ])


def error_metrics(actual, predicted):
    predicted = pd.Series(predicted, index=actual.index)
    residual = actual - predicted
    mae = residual.abs().mean()
    rmse = residual.pow(2).mean() ** 0.5
    return mae, rmse


def main():
    _, train, test = prepare_split(load_penguins())

    x_train = train[FEATURES]
    y_train = train["body_mass_g"]
    x_test = test[FEATURES]
    y_test = test["body_mass_g"]

    baseline_g = y_train.mean()
    baseline_mae, baseline_rmse = error_metrics(y_test, baseline_g)

    model = build_model()
    model.fit(x_train, y_train)

    train_mae, train_rmse = error_metrics(
        y_train, model.predict(x_train)
    )
    test_mae, test_rmse = error_metrics(
        y_test, model.predict(x_test)
    )

    print(f"Egitim kaydi: {len(train)}, test kaydi: {len(test)}")
    print(f"Baseline test MAE: {baseline_mae:.1f} g")
    print(f"Baseline test RMSE: {baseline_rmse:.1f} g")
    print(f"Pipeline egitim MAE: {train_mae:.1f} g")
    print(f"Pipeline egitim RMSE: {train_rmse:.1f} g")
    print(f"Pipeline test MAE: {test_mae:.1f} g")
    print(f"Pipeline test RMSE: {test_rmse:.1f} g")

    model_path = (
        Path(__file__).resolve().parent.parent
        / "model"
        / "model.joblib"
    )
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)

    original = model.predict(x_test.head(3))
    restored = joblib.load(model_path).predict(x_test.head(3))
    print("Kaydedilen modelin tahmin farki:",
          max(abs(original - restored)))


if __name__ == "__main__":
    main()