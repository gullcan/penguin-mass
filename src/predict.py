from math import isfinite
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"
VALID_SPECIES = {"Adelie", "Chinstrap", "Gentoo"}


def predict_mass(model, species, bill_length_mm, bill_depth_mm, flipper_length_mm):
    if species not in VALID_SPECIES:
        raise ValueError("Tur Adelie, Chinstrap veya Gentoo olmali.")

    measurements = (bill_length_mm, bill_depth_mm, flipper_length_mm)
    if any(not isfinite(value) or value <= 0 for value in measurements):
        raise ValueError("Olcumler sonlu ve sifirdan buyuk sayilar olmali.")

    penguin = pd.DataFrame([{
        "species": species,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
    }])
    penguin = penguin[list(model.feature_names_in_)]

    return float(model.predict(penguin)[0])


def read_measurement(label):
    try:
        return float(input(f"{label} (mm): "))
    except ValueError as error:
        raise ValueError(f"{label} icin sayisal bir deger girin.") from error

if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)

    try:
        species = input("Tur (Adelie/Chinstrap/Gentoo): ").strip()
        bill_length = read_measurement("Gaga uzunlugu")
        bill_depth = read_measurement("Gaga derinligi")
        flipper_length = read_measurement("Yuzgec uzunlugu")

        predicted_g = predict_mass(
            model, species, bill_length, bill_depth, flipper_length
        )
    except ValueError as error:
        print(f"Girdi hatasi: {error}")
        raise SystemExit(1) from None

    print(f"Tahmini vucut kutlesi: {predicted_g:.1f} g")
