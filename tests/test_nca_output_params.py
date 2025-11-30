"""Tests for nca_output() and analysis parameters."""

import pytest
import numpy as np
import pandas as pd
import io
import sys
import warnings

from nca import nca_analysis, nca_output, nca_random


@pytest.fixture
def test_data():
    """Generate test data with clear necessity relationship."""
    np.random.seed(42)
    df = nca_random(n=100, intercepts=0.2, slopes=0.8, corner=1)
    return df


@pytest.fixture
def model_ce_fdh(test_data):
    """Pre-computed CE-FDH model."""
    return nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh'])


@pytest.fixture
def model_multi(test_data):
    """Pre-computed model with multiple ceilings."""
    return nca_analysis(test_data, 'X', 'Y', ceilings=['ce_fdh', 'cr_fdh', 'ols'])


class TestNcaOutputSummaries:
    """Test nca_output() with summaries option."""

    def test_summaries_true(self, model_ce_fdh, capsys):
        """Test output with summaries=True."""
        nca_output(model_ce_fdh, summaries=True, plots=False, bottlenecks=False)
        
        captured = capsys.readouterr()
        # Should print something
        assert len(captured.out) > 0
    
    def test_summaries_false(self, model_ce_fdh, capsys):
        """Test output with summaries=False."""
        nca_output(model_ce_fdh, summaries=False, plots=False, bottlenecks=False)
        
        captured = capsys.readouterr()
        # Minimal output (just newline)
        assert len(captured.out.strip()) <= 10


class TestNcaOutputBottlenecks:
    """Test nca_output() with bottlenecks option."""

    def test_bottlenecks_true(self, model_ce_fdh, capsys):
        """Test output with bottlenecks=True."""
        nca_output(model_ce_fdh, summaries=False, plots=False, bottlenecks=True)
        
        captured = capsys.readouterr()
        # Should print bottleneck table
        assert len(captured.out) > 0
    
    def test_bottleneck_table_format(self, model_ce_fdh):
        """Test bottleneck table is properly formatted."""
        # Check the model has bottleneck data
        assert 'bottlenecks' in model_ce_fdh
        assert 'ce_fdh' in model_ce_fdh['bottlenecks']
        
        bn_table = model_ce_fdh['bottlenecks']['ce_fdh']
        assert isinstance(bn_table, pd.DataFrame)
        assert len(bn_table) > 0


class TestNcaOutputPlots:
    """Test nca_output() with plots option."""

    def test_plots_true(self, model_ce_fdh):
        """Test output with plots=True generates plot data."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Should not raise
        nca_output(model_ce_fdh, summaries=False, plots=True, bottlenecks=False)
    
    def test_plots_data_exists(self, model_ce_fdh):
        """Test that plot data exists in model."""
        assert 'plots' in model_ce_fdh


class TestCornerParameter:
    """Test corner parameter for different data orientations."""

    def test_corner_1(self, test_data):
        """Test corner=1 (upper left empty)."""
        model = nca_analysis(test_data, 'X', 'Y', corner=1)
        assert model is not None
    
    def test_corner_2(self):
        """Test corner=2 (upper right empty)."""
        # Need negative slope data
        df = nca_random(n=50, intercepts=0.8, slopes=-0.6, corner=2)
        model = nca_analysis(df, 'X', 'Y', corner=2)
        assert model is not None
    
    def test_corner_3(self):
        """Test corner=3 (lower left empty)."""
        # Need negative slope data  
        df = nca_random(n=50, intercepts=0.8, slopes=-0.6, corner=3)
        model = nca_analysis(df, 'X', 'Y', corner=3)
        assert model is not None
    
    def test_corner_4(self, test_data):
        """Test corner=4 (lower right empty)."""
        model = nca_analysis(test_data, 'X', 'Y', corner=4)
        assert model is not None


class TestFlipParameters:
    """Test flip_x and flip_y parameters."""

    def test_flip_x_false(self, test_data):
        """Test with flip_x=False (default)."""
        model = nca_analysis(test_data, 'X', 'Y', flip_x=False)
        assert model is not None
    
    def test_flip_x_true(self, test_data):
        """Test with flip_x=True."""
        model = nca_analysis(test_data, 'X', 'Y', flip_x=True)
        assert model is not None
    
    def test_flip_y_false(self, test_data):
        """Test with flip_y=False (default)."""
        model = nca_analysis(test_data, 'X', 'Y', flip_y=False)
        assert model is not None
    
    def test_flip_y_true(self, test_data):
        """Test with flip_y=True."""
        model = nca_analysis(test_data, 'X', 'Y', flip_y=True)
        assert model is not None
    
    def test_both_flips(self, test_data):
        """Test with both flip_x=True and flip_y=True."""
        model = nca_analysis(test_data, 'X', 'Y', flip_x=True, flip_y=True)
        assert model is not None


class TestScopeParameter:
    """Test custom scope parameter."""

    def test_scope_none(self, test_data):
        """Test with default scope (None)."""
        model = nca_analysis(test_data, 'X', 'Y', scope=None)
        assert model is not None
    
    def test_scope_custom(self, test_data):
        """Test with custom scope [Xmin, Xmax, Ymin, Ymax]."""
        model = nca_analysis(test_data, 'X', 'Y', scope=[0, 1, 0, 1])
        assert model is not None
    
    def test_scope_partial(self, test_data):
        """Test with partial custom scope."""
        model = nca_analysis(test_data, 'X', 'Y', scope=[0.1, 0.9, 0.1, 0.9])
        assert model is not None


class TestBottleneckXY:
    """Test bottleneck_x and bottleneck_y parameters."""

    def test_bottleneck_x_percentage_range(self, test_data):
        """Test bottleneck_x='percentage.range'."""
        model = nca_analysis(test_data, 'X', 'Y', bottleneck_x='percentage.range')
        assert 'bottlenecks' in model
    
    def test_bottleneck_x_percentage_max(self, test_data):
        """Test bottleneck_x='percentage.max'."""
        model = nca_analysis(test_data, 'X', 'Y', bottleneck_x='percentage.max')
        assert 'bottlenecks' in model
    
    def test_bottleneck_x_actual(self, test_data):
        """Test bottleneck_x='actual'."""
        model = nca_analysis(test_data, 'X', 'Y', bottleneck_x='actual')
        assert 'bottlenecks' in model
    
    def test_bottleneck_y_percentage_range(self, test_data):
        """Test bottleneck_y='percentage.range'."""
        model = nca_analysis(test_data, 'X', 'Y', bottleneck_y='percentage.range')
        assert 'bottlenecks' in model
    
    def test_bottleneck_y_percentage_max(self, test_data):
        """Test bottleneck_y='percentage.max'."""
        model = nca_analysis(test_data, 'X', 'Y', bottleneck_y='percentage.max')
        assert 'bottlenecks' in model
    
    def test_bottleneck_y_actual(self, test_data):
        """Test bottleneck_y='actual'."""
        model = nca_analysis(test_data, 'X', 'Y', bottleneck_y='actual')
        assert 'bottlenecks' in model
    
    def test_bottleneck_combinations(self, test_data):
        """Test different combinations of bottleneck_x and bottleneck_y."""
        model = nca_analysis(
            test_data, 'X', 'Y',
            bottleneck_x='actual',
            bottleneck_y='percentage.range'
        )
        assert 'bottlenecks' in model


class TestStepsParameter:
    """Test steps and step_size parameters."""

    def test_steps_10(self, test_data):
        """Test default steps=10."""
        model = nca_analysis(test_data, 'X', 'Y', steps=10)
        
        bn = model['bottlenecks']['ce_fdh']
        assert len(bn) == 11  # 0%, 10%, 20%, ..., 100%
    
    def test_steps_5(self, test_data):
        """Test steps=5."""
        model = nca_analysis(test_data, 'X', 'Y', steps=5)
        
        bn = model['bottlenecks']['ce_fdh']
        assert len(bn) == 6  # 0%, 20%, 40%, 60%, 80%, 100%
    
    def test_steps_20(self, test_data):
        """Test steps=20."""
        model = nca_analysis(test_data, 'X', 'Y', steps=20)
        
        bn = model['bottlenecks']['ce_fdh']
        assert len(bn) == 21  # 0% to 100% in 5% increments
    
    def test_step_size(self, test_data):
        """Test custom step_size - creates many steps based on percentage."""
        model = nca_analysis(test_data, 'X', 'Y', step_size=0.25)
        
        # step_size is interpreted as percentage increment (0.25% steps from 0 to 100)
        bn = model['bottlenecks']['ce_fdh']
        assert len(bn) > 0  # Just verify it works


class TestCutoffParameter:
    """Test cutoff parameter."""

    def test_cutoff_zero(self, test_data):
        """Test default cutoff=0."""
        model = nca_analysis(test_data, 'X', 'Y', cutoff=0)
        assert model is not None
    
    def test_cutoff_positive(self, test_data):
        """Test positive cutoff."""
        model = nca_analysis(test_data, 'X', 'Y', cutoff=0.1)
        assert model is not None


class TestMultipleXVariables:
    """Test analysis with multiple X variables."""

    def test_two_x_variables(self):
        """Test with two X variables."""
        df = nca_random(n=100, intercepts=[0.2, 0.3], slopes=[0.8, 0.7])
        
        model = nca_analysis(df, ['X1', 'X2'], 'Y', ceilings=['ce_fdh'])
        
        assert 'summaries' in model
        assert 'X1' in model['summaries']
        assert 'X2' in model['summaries']
    
    def test_three_x_variables(self):
        """Test with three X variables."""
        df = nca_random(
            n=100,
            intercepts=[0.2, 0.3, 0.25],
            slopes=[0.8, 0.7, 0.75]
        )
        
        model = nca_analysis(df, ['X1', 'X2', 'X3'], 'Y', ceilings=['ce_fdh'])
        
        assert 'summaries' in model
        assert len(model['summaries']) == 3


class TestQrTauParameter:
    """Test qr_tau parameter for quantile regression."""

    def test_qr_tau_default(self, test_data):
        """Test default qr_tau=0.95."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['qr'], qr_tau=0.95)
        assert model is not None
    
    def test_qr_tau_090(self, test_data):
        """Test qr_tau=0.90."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['qr'], qr_tau=0.90)
        assert model is not None
    
    def test_qr_tau_099(self, test_data):
        """Test qr_tau=0.99."""
        model = nca_analysis(test_data, 'X', 'Y', ceilings=['qr'], qr_tau=0.99)
        assert model is not None


class TestSelectionParameter:
    """Test selection parameter in nca_output."""

    def test_selection_none(self, model_multi):
        """Test selection=None (all summaries)."""
        nca_output(model_multi, summaries=True, plots=False, bottlenecks=False, selection=None)
    
    def test_selection_index(self, model_multi):
        """Test selection by index."""
        nca_output(model_multi, summaries=True, plots=False, bottlenecks=False, selection=[0])
    
    def test_selection_name(self, model_multi):
        """Test selection by name."""
        nca_output(model_multi, summaries=True, plots=False, bottlenecks=False, selection=['X'])
