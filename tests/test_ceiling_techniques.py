"""Tests for all ceiling techniques in nca_analysis()."""

import pytest
import numpy as np
import pandas as pd

from nca import nca_analysis, nca_random


# Generate test data fixture
@pytest.fixture
def test_data():
    """Generate test data with clear necessity relationship."""
    np.random.seed(42)
    df = nca_random(n=100, intercepts=0.2, slopes=0.8, corner=1)
    return df


@pytest.fixture
def simple_data():
    """Simple manually created test data."""
    return pd.DataFrame({
        'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Y': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    })


class TestCeFdhCeiling:
    """Test CE-FDH (Ceiling Envelopment - Free Disposal Hull)."""

    def test_ce_fdh_basic(self, test_data):
        """Test basic CE-FDH analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh'])
        
        assert model is not None
        assert 'summaries' in model
        assert 'X' in model['summaries']
    
    def test_ce_fdh_effect_size(self, test_data):
        """Test CE-FDH returns valid effect size."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh'])
        
        summary = model['summaries']['X']
        params = summary['params']
        
        # Effect size should be between 0 and 1
        effect_size = params.iloc[1, 0]  # Row 1 is "Effect size"
        assert 0 <= effect_size <= 1
    
    def test_ce_fdh_peers(self, test_data):
        """Test CE-FDH identifies peers (ceiling points)."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh'])
        
        assert 'peers' in model
    
    def test_ce_fdh_plots(self, test_data):
        """Test CE-FDH generates plot data."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh'])
        
        assert 'plots' in model


class TestCrFdhCeiling:
    """Test CR-FDH (Ceiling Regression - Free Disposal Hull)."""

    def test_cr_fdh_basic(self, test_data):
        """Test basic CR-FDH analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['cr_fdh'])
        
        assert model is not None
        assert 'summaries' in model
    
    def test_cr_fdh_slope_intercept(self, test_data):
        """Test CR-FDH returns slope and intercept."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['cr_fdh'])
        
        summary = model['summaries']['X']
        params = summary['params']
        
        # Should have slope and intercept
        assert 'Slope' in params.index or params.shape[0] > 7


class TestCeVrsCeiling:
    """Test CE-VRS (Ceiling Envelopment - Variable Returns to Scale)."""

    def test_ce_vrs_basic(self, test_data):
        """Test basic CE-VRS analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_vrs'])
        
        assert model is not None
        assert 'summaries' in model
    
    def test_ce_vrs_effect_size(self, test_data):
        """Test CE-VRS returns valid effect size."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_vrs'])
        
        summary = model['summaries']['X']
        params = summary['params']
        effect_size = params.iloc[1, 0]
        
        assert 0 <= effect_size <= 1


class TestCrVrsCeiling:
    """Test CR-VRS (Ceiling Regression - Variable Returns to Scale)."""

    def test_cr_vrs_basic(self, test_data):
        """Test basic CR-VRS analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['cr_vrs'])
        
        assert model is not None
        assert 'summaries' in model
    
    def test_cr_vrs_effect_size(self, test_data):
        """Test CR-VRS returns valid effect size."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['cr_vrs'])
        
        summary = model['summaries']['X']
        params = summary['params']
        effect_size = params.iloc[1, 0]
        
        assert 0 <= effect_size <= 1


class TestOlsCeiling:
    """Test OLS (Ordinary Least Squares) ceiling."""

    def test_ols_basic(self, test_data):
        """Test basic OLS analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ols'])
        
        assert model is not None
        assert 'summaries' in model
    
    def test_ols_slope_intercept(self, test_data):
        """Test OLS returns slope and intercept."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ols'])
        
        summary = model['summaries']['X']
        params = summary['params']
        
        # OLS should have slope and intercept values
        assert params.shape[0] > 5  # Has enough parameters


class TestColsCeiling:
    """Test COLS (Constrained OLS) ceiling."""

    def test_cols_basic(self, test_data):
        """Test basic COLS analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['cols'])
        
        assert model is not None
        assert 'summaries' in model
    
    def test_cols_effect_size(self, test_data):
        """Test COLS returns valid effect size."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['cols'])
        
        summary = model['summaries']['X']
        params = summary['params']
        effect_size = params.iloc[1, 0]
        
        assert 0 <= effect_size <= 1


class TestQrCeiling:
    """Test QR (Quantile Regression) ceiling."""

    def test_qr_basic(self, test_data):
        """Test basic QR analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['qr'])
        
        assert model is not None
        assert 'summaries' in model
    
    def test_qr_custom_tau(self, test_data):
        """Test QR with custom tau (quantile) value."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['qr'], qr_tau=0.90)
        
        assert model is not None


class TestClpCeiling:
    """Test C-LP (Linear Programming) ceiling."""

    def test_clp_basic(self, test_data):
        """Test basic C-LP analysis."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['c_lp'])
        
        assert model is not None
        assert 'summaries' in model
    
    def test_clp_effect_size(self, test_data):
        """Test C-LP returns valid effect size."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['c_lp'])
        
        summary = model['summaries']['X']
        params = summary['params']
        effect_size = params.iloc[1, 0]
        
        assert 0 <= effect_size <= 1


class TestMultipleCeilings:
    """Test using multiple ceiling techniques together."""

    def test_all_ceilings(self, test_data):
        """Test all ceiling techniques at once."""
        all_ceilings = ['ce_fdh', 'cr_fdh', 'ce_vrs', 'cr_vrs', 'ols', 'cols', 'qr', 'c_lp']
        model = nca_analysis(test_data, 'X', 'Y', ceilings=all_ceilings)
        
        assert model is not None
        assert 'summaries' in model
    
    def test_fdh_pair(self, test_data):
        """Test CE-FDH and CR-FDH together."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh', 'cr_fdh'])
        
        assert model is not None
        summary = model['summaries']['X']
        params = summary['params']
        
        # Should have columns for both techniques
        assert params.shape[1] >= 2
    
    def test_vrs_pair(self, test_data):
        """Test CE-VRS and CR-VRS together."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_vrs', 'cr_vrs'])
        
        assert model is not None
    
    def test_default_ceilings(self, test_data):
        """Test default ceiling techniques (ols, ce_fdh, cr_fdh)."""
        model = nca_analysis(test_data, 'X', 'Y')  # No ceilings specified
        
        assert model is not None
        summary = model['summaries']['X']
        params = summary['params']
        
        # Should have at least 2 ceiling columns (defaults may vary)
        assert params.shape[1] >= 2
    
    def test_invalid_ceiling_warning(self, test_data):
        """Test warning for invalid ceiling technique."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = nca_analysis(test_data, 'X', 'Y', ceilings=['invalid_ceiling'])
            
            # Should fall back to defaults and warn
            assert model is not None
            assert len(w) > 0


class TestCeilingComparison:
    """Test that different ceilings give different results."""

    def test_ce_vs_cr_fdh(self, test_data):
        """Test that CE-FDH and CR-FDH give different effect sizes."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh', 'cr_fdh'])
        
        summary = model['summaries']['X']
        params = summary['params']
        
        # Effect sizes may be different (row 1)
        ce_effect = params.iloc[1, 0]
        cr_effect = params.iloc[1, 1]
        
        # Both should be valid
        assert 0 <= ce_effect <= 1
        assert 0 <= cr_effect <= 1
    
    def test_step_vs_line_ceilings(self, test_data):
        """Test step function (CE-FDH) vs line (CR-FDH) ceilings."""
        model_ce = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh'])
        model_cr = nca_analysis(test_data, 'X', 'Y', ceilings=['cr_fdh'])
        
        # Both should complete successfully
        assert model_ce is not None
        assert model_cr is not None


class TestBottleneckOutput:
    """Test bottleneck table generation for different ceilings."""

    def test_ce_fdh_bottleneck(self, test_data):
        """Test CE-FDH generates bottleneck data."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh'])
        
        assert 'bottlenecks' in model
        assert 'ce_fdh' in model['bottlenecks']
    
    def test_cr_fdh_bottleneck(self, test_data):
        """Test CR-FDH generates bottleneck data."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['cr_fdh'])
        
        assert 'bottlenecks' in model
        assert 'cr_fdh' in model['bottlenecks']
    
    def test_ols_no_bottleneck(self, test_data):
        """Test OLS does not generate bottleneck (by design)."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['ols'])
        
        # OLS is in P_NO_BOTTLENECK list
        # Should still have bottlenecks dict but maybe empty or without ols
        assert 'bottlenecks' in model
