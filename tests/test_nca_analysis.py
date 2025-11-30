import os
import sys
import unittest

import pandas as pd

# Add the parent directory to sys.path to import nca
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nca.nca import nca_analysis
from nca.nca_random import nca_random


class TestNCAAnalysis(unittest.TestCase):
    def test_basic_analysis(self):
        # Generate random data
        data = nca_random(100, 0, 1)

        # Run analysis
        model = nca_analysis(data, "X", "Y", ceilings=["ce_fdh", "ce_vrs"])

        self.assertIsNotNone(model)
        self.assertIn("summaries", model)
        self.assertIn("X", model["summaries"])
        self.assertIn("plots", model)

        summary = model["summaries"]["X"]
        self.assertIn("params", summary)
        self.assertIn("peers", model)

        # Check effect size
        # params index 1 is effect size (0-based)
        # summary['params'] is a DataFrame, columns are methods
        effect_size = summary["params"].iloc[1, 0]
        self.assertTrue(0 <= effect_size <= 1)

    def test_outliers(self):
        data = nca_random(50, 0, 1)
        # Introduce an outlier?
        # nca_random generates valid data respecting the ceiling.
        # To test outliers, we might need data that violates it?
        # But nca_outliers detects points that *cause* the ceiling to be lower (i.e. peers).

        from nca.nca_outliers import nca_outliers

        outliers = nca_outliers(data, "X", "Y", ceiling="ce_fdh")

        # It might return None if no outliers found (k=1 default)
        # But usually peers are outliers in the sense of defining the ceiling.
        # nca_outliers detects points that if removed, increase the ceiling fit?
        # No, it detects points that determine the ceiling.

        # If outliers is not None, check structure
        if outliers is not None:
            self.assertIsInstance(outliers, pd.DataFrame)
            self.assertIn("outliers", outliers.columns)


if __name__ == "__main__":
    unittest.main()
