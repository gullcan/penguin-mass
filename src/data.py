from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "penguins.csv"

REQUIRED_COLUMNS = {
    "species",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
}


def load_penguins() -> pd.DataFrame:
    table = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS.difference(table.columns)
    if missing:
        raise ValueError(f"Veride eksik sutunlar: {sorted(missing)}")

    if table.empty:
        raise ValueError("Penguen veri dosyasi bos.")

    return table

def prepare_split(table):
    known = table.dropna(subset=["body_mass_g"]).copy()
    train = known.sample(frac=0.8, random_state=42)
    test = known.drop(index=train.index)
    return known, train, test

if __name__ == "__main__":
    table = load_penguins()
    print("Kaynak tablo boyutu:", table.shape)
    print("Agirligi bilinen kayit:", table["body_mass_g"].notna().sum())