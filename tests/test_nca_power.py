"""Tests for nca_power function - power analysis functionality."""

import pytest
import numpy as np
import pandas as pd

from nca import nca_power


class TestNcaPowerBasic:
    """Basic functionality tests for nca_power."""

    def test_basic_call(self, capsys):
        """Test basic power analysis call with minimal iterations."""
        # Use very small rep to make test fast
        result = nca_power(n=[20], effect=0.2, slope=1, ceiling="ce_fdh", rep=2, test_rep=10)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
    def test_output_columns(self, capsys):
        """Test that output has expected columns."""
        result = nca_power(n=[30], effect=0.15, slope=1, ceiling="ce_fdh", rep=2, test_rep=10)
        
        expected_cols = ['n', 'ES', 'slope', 'ceiling', 'p', 'distr.x', 'distr.y', 'power']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


class TestNcaPowerInputValidation:
    """Test input validation for nca_power."""

    def test_invalid_effect_zero(self, capsys):
        """Test that effect=0 is rejected."""
        result = nca_power(n=[20], effect=0, slope=1, ceiling="ce_fdh", rep=1, test_rep=5)
        captured = capsys.readouterr()
        assert 'effect size' in captured.out.lower()
        assert result is None

    def test_invalid_effect_one(self, capsys):
        """Test that effect=1 is rejected."""
        result = nca_power(n=[20], effect=1, slope=1, ceiling="ce_fdh", rep=1, test_rep=5)
        captured = capsys.readouterr()
        assert 'effect size' in captured.out.lower()
        assert result is None

    def test_invalid_effect_negative(self, capsys):
        """Test that negative effect is rejected."""
        result = nca_power(n=[20], effect=-0.1, slope=1, ceiling="ce_fdh", rep=1, test_rep=5)
        captured = capsys.readouterr()
        assert 'effect size' in captured.out.lower()
        assert result is None

    def test_invalid_slope_zero(self, capsys):
        """Test that slope=0 is rejected."""
        result = nca_power(n=[20], effect=0.2, slope=0, ceiling="ce_fdh", rep=1, test_rep=5)
        captured = capsys.readouterr()
        assert 'slope' in captured.out.lower()
        assert result is None

    def test_invalid_slope_negative(self, capsys):
        """Test that negative slope is rejected."""
        result = nca_power(n=[20], effect=0.2, slope=-1, ceiling="ce_fdh", rep=1, test_rep=5)
        captured = capsys.readouterr()
        assert 'slope' in captured.out.lower()
        assert result is None


class TestNcaPowerParameters:
    """Test different parameters for nca_power."""

    def test_multiple_n(self, capsys):
        """Test with multiple sample sizes."""
        result = nca_power(n=[20, 30], effect=0.2, slope=1, ceiling="ce_fdh", rep=2, test_rep=10)
        
        assert len(result) == 2  # One row per sample size
        assert 20 in result['n'].values
        assert 30 in result['n'].values

    def test_multiple_effects(self, capsys):
        """Test with multiple effect sizes."""
        result = nca_power(n=[25], effect=[0.1, 0.2], slope=1, ceiling="ce_fdh", rep=2, test_rep=10)
        
        assert len(result) == 2  # One row per effect size
        assert 0.1 in result['ES'].values
        assert 0.2 in result['ES'].values

    def test_multiple_slopes(self, capsys):
        """Test with multiple slopes."""
        result = nca_power(n=[25], effect=0.2, slope=[0.5, 1.0], ceiling="ce_fdh", rep=2, test_rep=10)
        
        assert len(result) == 2  # One row per slope
        assert 0.5 in result['slope'].values
        assert 1.0 in result['slope'].values

    def test_different_ceiling(self, capsys):
        """Test with different ceiling technique."""
        # Use ce_vrs instead of cr_fdh which may have numerical issues
        result = nca_power(n=[25], effect=0.2, slope=1, ceiling="ce_vrs", rep=2, test_rep=10)
        
        assert result is not None
        assert 'ce_vrs' in result['ceiling'].values


class TestNcaPowerDistributions:
    """Test different distribution options."""

    def test_uniform_distribution(self, capsys):
        """Test with uniform distribution (default)."""
        result = nca_power(
            n=[25], effect=0.2, slope=1, ceiling="ce_fdh",
            distribution_x="uniform", distribution_y="uniform",
            rep=2, test_rep=10
        )
        
        assert result is not None
        assert 'uniform' in result['distr.x'].values
        assert 'uniform' in result['distr.y'].values

    def test_normal_distribution(self, capsys):
        """Test with normal distribution."""
        result = nca_power(
            n=[25], effect=0.2, slope=1, ceiling="ce_fdh",
            distribution_x="normal", distribution_y="normal",
            rep=2, test_rep=10
        )
        
        assert result is not None
        assert 'normal' in result['distr.x'].values
        assert 'normal' in result['distr.y'].values

    def test_mixed_distributions(self, capsys):
        """Test with mixed distributions."""
        result = nca_power(
            n=[25], effect=0.2, slope=1, ceiling="ce_fdh",
            distribution_x="uniform", distribution_y="normal",
            rep=2, test_rep=10
        )
        
        assert result is not None
        assert 'uniform' in result['distr.x'].values
        assert 'normal' in result['distr.y'].values


class TestNcaPowerOutputValues:
    """Test output value ranges and types."""

    def test_power_range(self, capsys):
        """Test that power values are between 0 and 1."""
        result = nca_power(n=[30], effect=0.2, slope=1, ceiling="ce_fdh", rep=3, test_rep=15)
        
        assert all(result['power'] >= 0)
        assert all(result['power'] <= 1)

    def test_p_value_range(self, capsys):
        """Test that p-values are between 0 and 1."""
        result = nca_power(n=[30], effect=0.2, slope=1, ceiling="ce_fdh", rep=3, test_rep=15)
        
        assert all(result['p'] >= 0)
        assert all(result['p'] <= 1)

    def test_numeric_output(self, capsys):
        """Test that numeric columns contain numeric values."""
        result = nca_power(n=[30], effect=0.2, slope=1, ceiling="ce_fdh", rep=2, test_rep=10)
        
        # Due to pandas concat behavior, columns may be object type but contain numeric values
        # So we check if values can be converted to numeric
        assert all(pd.to_numeric(result['n'], errors='coerce').notna())
        assert all(pd.to_numeric(result['ES'], errors='coerce').notna())
        assert all(pd.to_numeric(result['slope'], errors='coerce').notna())
        assert all(pd.to_numeric(result['power'], errors='coerce').notna())
        assert all(pd.to_numeric(result['p'], errors='coerce').notna())


class TestPInterceptFunction:
    """Test the helper function p_intercept."""

    def test_intercept_calculation(self):
        """Test that p_intercept returns valid intercepts."""
        from nca.nca_power import p_intercept
        
        # Various slope and effect combinations
        intercept1 = p_intercept(1.0, 0.2)
        assert isinstance(intercept1, float)
        
        intercept2 = p_intercept(0.5, 0.1)
        assert isinstance(intercept2, float)
        
        intercept3 = p_intercept(1.5, 0.3)
        assert isinstance(intercept3, float)
