import sys
import unittest
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import load_penguins, prepare_split
from predict import MODEL_PATH, predict_mass


class DataFlowTests(unittest.TestCase):
    def test_split_has_no_shared_rows(self):
        known, train, test = prepare_split(load_penguins())

        self.assertEqual(len(known), 342)
        self.assertEqual((len(train), len(test)), (274, 68))
        self.assertFalse(set(train.index) & set(test.index))
        self.assertEqual(
            set(known.index),
            set(train.index) | set(test.index),
        )

    def test_saved_model_reproduces_known_prediction(self):
        model = joblib.load(MODEL_PATH)

        predicted_g = predict_mass(
            model, "Adelie", 39.5, 17.4, 186.0
        )

        self.assertAlmostEqual(
            predicted_g, 3509.178016, delta=0.1
        )

    def test_unknown_species_is_rejected(self):
        model = joblib.load(MODEL_PATH)

        with self.assertRaises(ValueError):
            predict_mass(model, "Unknown", 39.5, 17.4, 186.0)


if __name__ == "__main__":
    unittest.main()