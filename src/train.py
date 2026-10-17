from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from joblib import dump, load
from data import load_penguins, prepare_split

penguins = load_penguins()
known, train, test = prepare_split(penguins)

baseline_mass = train["body_mass_g"].mean()
baseline_predictions = pd.Series(
    baseline_mass,
    index=test.index,
    name="predicted_body_mass_g",
)

print("Egitim kaydi:", len(train))
print("Test kaydi:", len(test))
print("Ortak satir:", len(train.index.intersection(test.index)))
print("Ogrenilen ortalama (g):", round(baseline_mass, 1))
print("Tahmin sayisi:", len(baseline_predictions))
print(baseline_predictions.head(3))


def linear_prediction(flipper_mm, intercept_g, slope_g_per_mm):
    return intercept_g + slope_g_per_mm * flipper_mm


first_flipper = test["flipper_length_mm"].iloc[0]

trial_intercept = -6000.0
trial_slope = 50.0

trial_mass = linear_prediction(
    first_flipper,
    trial_intercept,
    trial_slope,
)

print("Ilk test kaydinin indeksi:", test.index[0])
print("Yuzgec uzunlugu (mm):", first_flipper)
print("Baseline tahmini (g):", round(baseline_predictions.iloc[0], 1))
print("Dogrusal deneme tahmini (g):", round(trial_mass, 1))

actual_train = train["body_mass_g"]

baseline_train_predictions = pd.Series(
    baseline_mass,
    index=train.index,
)

trial_train_predictions = linear_prediction(
    train["flipper_length_mm"],
    trial_intercept,
    trial_slope,
)

x_train = train["flipper_length_mm"]
y_train = actual_train

x_offsets = x_train - x_train.mean()
y_offsets = y_train - y_train.mean()

numerator = (x_offsets * y_offsets).sum()
denominator = (x_offsets ** 2).sum()

if denominator == 0:
    raise ValueError("Egitimdeki yuzgec uzunluklari ayni; egim hesaplanamaz.")

learned_slope = numerator / denominator
learned_intercept = y_train.mean() - learned_slope * x_train.mean()

learned_train_predictions = linear_prediction(
    x_train, learned_intercept, learned_slope
)
learned_test_predictions = linear_prediction(
    test["flipper_length_mm"], learned_intercept, learned_slope
)

print("Ogrenilen egim (g/mm):", round(learned_slope, 2))
print("Ogrenilen sabit terim (g):", round(learned_intercept, 1))
print("Ilk uc test tahmini:")
print(learned_test_predictions.head(3))

for model_name, predicted_train in [
    ("Baseline", baseline_train_predictions),
    ("Deneme dogrusu", trial_train_predictions),
    ("Ogrenilen dogru", learned_train_predictions),
]:
    residuals = actual_train - predicted_train
    absolute_errors = residuals.abs()
    squared_errors = residuals ** 2

    mae = absolute_errors.mean()
    mse = squared_errors.mean()
    rmse = mse ** 0.5

    print(model_name)
    print("  MAE (g):", round(mae, 1))
    print("  MSE (g^2):", round(mse, 1))
    print("  RMSE (g):", round(rmse, 1))


actual_test = test["body_mass_g"]

for model_name, predicted_test in [
    ("Baseline", baseline_predictions),
    ("Ogrenilen dogru", learned_test_predictions),
]:
    if not actual_test.index.equals(predicted_test.index):
        raise ValueError("Gercek ve tahmin satirlari eslesmiyor.")

    residuals = actual_test - predicted_test
    mae = residuals.abs().mean()
    rmse = ((residuals ** 2).mean()) ** 0.5

    print(model_name, "test MAE (g):", round(mae, 1))
    print(model_name, "test RMSE (g):", round(rmse, 1))

comparison = pd.DataFrame({
    "gercek_g": actual_test,
    "tahmin_g": learned_test_predictions,
    "artik_g": actual_test - learned_test_predictions,
})

print("Ilk uc test kaydinin sonucu:")
print(comparison.head(3))


estimate_g = 0.0
learning_rate = 0.25

for step in range(1, 9):
    residuals = y_train - estimate_g
    gradient = (-2 * residuals).mean()

    estimate_g = estimate_g - learning_rate * gradient
    mse = ((y_train - estimate_g) ** 2).mean()

    print(
        "Adim:", step,
        "Tahmin (g):", round(estimate_g, 1),
        "Egitim MSE:", round(mse, 1),
    )

print("Dogrudan hesaplanan ortalama (g):", round(baseline_mass, 1))


x_mean = x_train.mean()
x_std = (((x_train - x_mean) ** 2).mean()) ** 0.5
z_train = (x_train - x_mean) / x_std

a_g = 0.0
b_g = 0.0
step_size = 0.25

for step in range(1, 21):
    predictions_g = a_g + b_g * z_train
    errors_g = predictions_g - y_train

    gradient_a = 2 * errors_g.mean()
    gradient_b = 2 * (errors_g * z_train).mean()

    a_g = a_g - step_size * gradient_a
    b_g = b_g - step_size * gradient_b

    if step in (1, 2, 5, 10, 20):
        new_predictions_g = a_g + b_g * z_train
        mse = ((y_train - new_predictions_g) ** 2).mean()
        print("Adim:", step, "a:", round(a_g, 1),
              "b:", round(b_g, 1), "MSE:", round(mse, 1))

gd_slope = b_g / x_std
gd_intercept = a_g - gd_slope * x_mean

print("Gradient descent egimi (g/mm):", round(gd_slope, 2))
print("Dogrudan hesaplanan egim (g/mm):", round(learned_slope, 2))
print("Gradient descent sabiti (g):", round(gd_intercept, 1))
print("Dogrudan hesaplanan sabit (g):", round(learned_intercept, 1))


bill_train = train["bill_length_mm"]
bill_mean = bill_train.mean()
bill_std = (((bill_train - bill_mean) ** 2).mean()) ** 0.5
z_bill_train = (bill_train - bill_mean) / bill_std

a_two = 0.0
b_flipper = 0.0
b_bill = 0.0
step_size_two = 0.1

for step in range(1, 501):
    predictions = a_two + b_flipper * z_train + b_bill * z_bill_train
    errors = predictions - y_train

    gradient_a = 2 * errors.mean()
    gradient_flipper = 2 * (errors * z_train).mean()
    gradient_bill = 2 * (errors * z_bill_train).mean()

    a_two = a_two - step_size_two * gradient_a
    b_flipper = b_flipper - step_size_two * gradient_flipper
    b_bill = b_bill - step_size_two * gradient_bill

    if step in (1, 10, 100, 500):
        updated = a_two + b_flipper * z_train + b_bill * z_bill_train
        mse = ((y_train - updated) ** 2).mean()
        print("Iki girdi, adim:", step, "egitim MSE:", round(mse, 1))

two_train_predictions = (
    a_two + b_flipper * z_train + b_bill * z_bill_train
)
two_train_rmse = (((y_train - two_train_predictions) ** 2).mean()) ** 0.5

print("Iki girdi, egitim RMSE (g):", round(two_train_rmse, 1))
print("Yuzgec katsayisi (g/mm):", round(b_flipper / x_std, 2))
print("Gaga katsayisi (g/mm):", round(b_bill / bill_std, 2))


species_flags = pd.get_dummies(train["species"], dtype=int)
species_flags = species_flags[["Chinstrap", "Gentoo"]]

print("Turlerin sayisal temsili:")
print(species_flags.groupby(train["species"]).first())

species_means = train.groupby("species")["body_mass_g"].mean()

adelie_g = species_means["Adelie"]
chinstrap_extra_g = species_means["Chinstrap"] - adelie_g
gentoo_extra_g = species_means["Gentoo"] - adelie_g

species_train_predictions = (
    adelie_g
    + chinstrap_extra_g * species_flags["Chinstrap"]
    + gentoo_extra_g * species_flags["Gentoo"]
)

species_train_rmse = (
    ((y_train - species_train_predictions) ** 2).mean()
) ** 0.5

print("Egitim tur ortalamalari (g):")
print(species_means.round(1))
print("Chinstrap farki (g):", round(chinstrap_extra_g, 1))
print("Gentoo farki (g):", round(gentoo_extra_g, 1))
print("Yalnizca tur, egitim RMSE (g):", round(species_train_rmse, 1))


numeric_columns = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
]
feature_columns = numeric_columns + ["species"]

X_train = train[feature_columns]
X_test = test[feature_columns]

preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_columns),
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

model = Pipeline([
    ("preprocess", preprocessor),
    ("regression", LinearRegression()),
])

model.fit(X_train, y_train)

train_predictions = pd.Series(
    model.predict(X_train),
    index=X_train.index,
)
train_residuals = y_train - train_predictions

print("Pipeline egitim MAE (g):",
      round(train_residuals.abs().mean(), 1))
print("Pipeline egitim RMSE (g):",
      round(((train_residuals ** 2).mean()) ** 0.5, 1))
print("Modele ulasan sutunlar:")
print(model.named_steps["preprocess"].get_feature_names_out())

first_test_predictions = pd.Series(
    model.predict(X_test.head(3)),
    index=X_test.head(3).index,
)
print("Ilk uc test pengueni icin tahmin:")
print(first_test_predictions)

y_test = test["body_mass_g"]

pipeline_test_predictions = pd.Series(
    model.predict(X_test),
    index=X_test.index,
    name="pipeline_prediction_g",
)

pipeline_test_residuals = y_test - pipeline_test_predictions
pipeline_test_mae = pipeline_test_residuals.abs().mean()
pipeline_test_rmse = (
    (pipeline_test_residuals ** 2).mean()
) ** 0.5

print("Pipeline test MAE (g):", round(pipeline_test_mae, 1))
print("Pipeline test RMSE (g):", round(pipeline_test_rmse, 1))
print("Egitim RMSE (g):", 320.9)

comparison = pd.DataFrame({
    "species": test["species"],
    "actual_g": y_test,
    "predicted_g": pipeline_test_predictions,
    "residual_g": pipeline_test_residuals,
})

print("Ilk uc test kaydi:")
print(comparison.head(3))


comparison["absolute_error_g"] = comparison["residual_g"].abs()
comparison["squared_error_g2"] = comparison["residual_g"] ** 2

by_species = comparison.groupby("species").agg(
    n=("residual_g", "size"),
    mean_residual_g=("residual_g", "mean"),
    mae_g=("absolute_error_g", "mean"),
    mse_g2=("squared_error_g2", "mean"),
)
by_species["rmse_g"] = by_species["mse_g2"] ** 0.5

print("Ture gore test hatalari:")
print(
    by_species[
        ["n", "mean_residual_g", "mae_g", "rmse_g"]
    ].round(1)
)

print("Ture gore MAE: her # yaklasik 20 g")
for species, row in by_species.iterrows():
    bar = "#" * round(row["mae_g"] / 20)
    print(
        f"{species:10} | {bar} "
        f"{row['mae_g']:.1f} g (n={int(row['n'])})"
    )


model_path = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "model.joblib"
)
model_path.parent.mkdir(exist_ok=True)

dump(model, model_path)
loaded_model = load(model_path)

original = model.predict(X_test.head(3))
restored = loaded_model.predict(X_test.head(3))
largest_difference = max(abs(original - restored))

print("Model kaydedildi:", model_path)
print("Yeniden yukleme sonrasi en buyuk tahmin farki:",
      largest_difference)