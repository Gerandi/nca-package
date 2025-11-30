"""
NCA - Necessary Condition Analysis

A Python implementation of Necessary Condition Analysis (NCA),
a methodology for identifying necessary conditions in datasets.

Author: Gerandi Matraku
Based on: The original R package (v4.0.4) by Jan Dul

Main Functions
--------------
nca_analysis : Perform NCA analysis on data
nca_output : Display analysis results
nca_outliers : Identify outliers in NCA analysis
nca_power : Power analysis for NCA
nca_random : Generate random data for testing

Example
-------
>>> import pandas as pd
>>> from nca import nca_analysis, nca_output
>>> data = pd.DataFrame({'X': [1,2,3,4,5], 'Y': [2,3,4,5,6]})
>>> model = nca_analysis(data, 'X', 'Y')
>>> nca_output(model)
"""

__version__ = "1.0.0"
__author__ = "Gerandi Matraku"

# Core analysis functions
from .nca import nca_analysis
from .nca_output import nca_output
from .nca_outliers import nca_outliers
from .nca_power import nca_power
from .nca_random import nca_random

# Public API
__all__ = [
    # Core functions
    "nca_analysis",
    "nca_output",
    "nca_outliers",
    "nca_power",
    "nca_random",
    # Metadata
    "__version__",
    "__author__",
]
