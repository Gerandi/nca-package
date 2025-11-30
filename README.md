# NCA - Necessary Condition Analysis

[![PyPI version](https://badge.fury.io/py/nca.svg)](https://badge.fury.io/py/nca)
[![Python Versions](https://img.shields.io/pypi/pyversions/nca.svg)](https://pypi.org/project/nca/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python implementation of Necessary Condition Analysis (NCA), a methodology for identifying necessary conditions in datasets.

**Author:** Gerandi Matraku  
**Based on:** The original R package by Jan Dul

## Overview

Necessary Condition Analysis (NCA) is a data analysis approach that can identify necessary conditions in datasets. Unlike traditional correlation or regression analysis which identifies conditions that contribute to an outcome (sufficiency), NCA identifies conditions that must be present for an outcome to occur (necessity).

This package is a complete Python port of the [NCA R package](https://cran.r-project.org/package=NCA) (v4.0.4) by Jan Dul.

## Installation

```bash
pip install nca
```

## Quick Start

```python
import pandas as pd
from nca import nca_analysis, nca_output

# Load your data
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
})

# Run NCA analysis
model = nca_analysis(data, 'X', 'Y')

# Display results
nca_output(model)
```

## Features

- **Multiple ceiling techniques**: CE-FDH, CR-FDH, CE-VRS, CR-VRS, and more
- **Effect size calculation**: Quantify the size of the necessity effect
- **Statistical significance testing**: Permutation tests for p-values
- **Bottleneck analysis**: Identify minimum necessary levels
- **Visualization**: Plot ceiling lines and data points
- **Confidence intervals**: Bootstrap confidence intervals for ceiling lines

## Main Functions

### `nca_analysis()`

Performs the core NCA analysis:

```python
model = nca_analysis(
    data,           # DataFrame with X and Y variables
    x='X',          # Independent variable(s)
    y='Y',          # Dependent variable
    ceilings=['ce_fdh', 'cr_fdh'],  # Ceiling techniques
    test_rep=1000   # Number of permutation test repetitions
)
```

### `nca_output()`

Displays analysis results:

```python
nca_output(
    model,
    plots=True,        # Show scatter plots
    summaries=True,    # Show summary statistics
    bottlenecks=True   # Show bottleneck tables
)
```

## Ceiling Techniques

| Technique | Description |
|-----------|-------------|
| `ce_fdh` | Ceiling Envelopment - Free Disposal Hull |
| `cr_fdh` | Ceiling Regression - Free Disposal Hull |
| `ce_vrs` | Ceiling Envelopment - Variable Returns to Scale |
| `cr_vrs` | Ceiling Regression - Variable Returns to Scale |
| `ols` | Ordinary Least Squares (for comparison) |

## Output Metrics

- **Effect size**: The proportion of the scope above the ceiling line (0-1)
- **Ceiling zone**: The area above the ceiling line
- **c-accuracy**: Percentage of observations on or below the ceiling
- **Fit**: How well the ceiling fits the data
- **p-value**: Statistical significance from permutation tests
- **Inefficiency**: Various inefficiency measures

## Documentation

For detailed documentation, see:
- [Quick Start Guide](https://repub.eur.nl/pub/78323/)
- [NCA Website](https://www.erim.nl/nca)

## Citation

If you use this package, please cite:

```bibtex
@software{matraku2025nca,
  title={NCA: Necessary Condition Analysis for Python},
  author={Matraku, Gerandi},
  year={2025},
  url={https://github.com/Gerandi/nca},
  note={Python implementation of NCA}
}
```

And the original methodology:

```bibtex
@article{dul2016necessary,
  title={Necessary Condition Analysis (NCA): Logic and methodology of "Necessary but Not Sufficient" causality},
  author={Dul, Jan},
  journal={Organizational Research Methods},
  volume={19},
  number={1},
  pages={10--52},
  year={2016},
  publisher={SAGE Publications}
}
```

## References

- Dul, J. (2016). "Necessary Condition Analysis (NCA): Logic and Methodology of 'Necessary but Not Sufficient' Causality." Organizational Research Methods 19(1), 10-52.
- Dul, J. (2020). "Conducting Necessary Condition Analysis." SAGE Publications.
- Dul, J., van der Laan, E., & Kuik, R. (2020). "A statistical significance test for Necessary Condition Analysis." Organizational Research Methods, 23(2), 385-395.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
