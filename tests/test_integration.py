"""Integration tests - Complete end-to-end tests of all NCA functionality."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from nca import nca_analysis, nca_output, nca_random, nca_outliers, nca_power


class TestCompleteWorkflow:
    """Test complete NCA workflow from data generation to output."""

    def test_single_x_complete_workflow(self):
        """Test complete workflow with single X variable."""
        # Step 1: Generate data
        np.random.seed(42)
        df = nca_random(n=100, intercepts=[0.15], slopes=[0.75])
        
        assert len(df) == 100
        assert 'X' in df.columns
        assert 'Y' in df.columns
        
        # Step 2: Run analysis with multiple ceilings
        model = nca_analysis(
            df, 'X', 'Y',
            ceilings=['ce_fdh', 'cr_fdh', 'ce_vrs', 'ols'],
            steps=10
        )
        
        # Verify model structure
        assert 'summaries' in model
        assert 'plots' in model
        assert 'bottlenecks' in model
        assert 'X' in model['summaries']
        
        # Verify summary content
        summary = model['summaries']['X']
        assert 'params' in summary
        assert 'global' in summary
        
        # Verify bottleneck tables
        for ceiling in ['ce_fdh', 'cr_fdh', 'ce_vrs']:
            assert ceiling in model['bottlenecks']
            bn = model['bottlenecks'][ceiling]
            assert len(bn) > 0  # Should have bottleneck data
        
        # Step 3: Generate output
        with tempfile.TemporaryDirectory() as tmpdir:
            import matplotlib
            matplotlib.use('Agg')
            
            nca_output(
                model,
                plots=True,
                summaries=True,
                bottlenecks=True,
                pdf=True,
                path=tmpdir
            )
            
            # Verify PDFs were created
            pdf_files = [f for f in os.listdir(tmpdir) if f.endswith('.pdf')]
            assert len(pdf_files) >= 2

    def test_multiple_x_complete_workflow(self):
        """Test complete workflow with multiple X variables."""
        # Step 1: Generate data with 3 X variables
        np.random.seed(42)
        df = nca_random(
            n=100,
            intercepts=[0.15, 0.2, 0.1],
            slopes=[0.75, 0.65, 0.8]
        )
        
        assert len(df) == 100
        assert 'X1' in df.columns
        assert 'X2' in df.columns
        assert 'X3' in df.columns
        assert 'Y' in df.columns
        
        # Step 2: Run analysis
        model = nca_analysis(
            df, ['X1', 'X2', 'X3'], 'Y',
            ceilings=['ce_fdh', 'ols']
        )
        
        # Verify summaries for all X variables
        assert 'X1' in model['summaries']
        assert 'X2' in model['summaries']
        assert 'X3' in model['summaries']
        
        # Step 3: Test selection in output
        nca_output(model, plots=False, summaries=True, selection=[0, 2])
        # Should work without error

    def test_with_permutation_tests(self):
        """Test workflow with permutation tests."""
        np.random.seed(42)
        df = nca_random(n=50, intercepts=[0.2], slopes=[0.7])
        
        # Run with permutation tests
        model = nca_analysis(
            df, 'X', 'Y',
            ceilings=['ce_fdh'],
            test_rep=20  # Small number for speed
        )
        
        # Verify tests were performed
        assert 'tests' in model
        if 'X' in model['tests']:
            assert 'ce_fdh' in model['tests']['X']
            test_result = model['tests']['X']['ce_fdh']
            assert 'p_value' in test_result
            assert 0 <= test_result['p_value'] <= 1

    def test_outlier_detection_workflow(self):
        """Test outlier detection in complete workflow."""
        np.random.seed(42)
        df = nca_random(n=80, intercepts=[0.15], slopes=[0.8])
        
        # Add a potential outlier
        outlier = pd.DataFrame({'X': [0.1], 'Y': [0.95]}, index=[100])
        df = pd.concat([df, outlier])
        
        # Run outlier detection
        outliers = nca_outliers(df, 'X', 'Y', ceiling='ce_fdh')
        
        # Should either find outliers or return None (no outliers)
        assert outliers is None or isinstance(outliers, pd.DataFrame)

    def test_corner_variations(self):
        """Test all corner variations work correctly."""
        np.random.seed(42)
        df = nca_random(n=50, intercepts=[0.2], slopes=[0.7])
        
        for corner in [1, 2, 3, 4]:
            model = nca_analysis(
                df, 'X', 'Y',
                ceilings=['ce_fdh'],
                corner=corner
            )
            assert 'summaries' in model
            assert 'X' in model['summaries']

    def test_bottleneck_options(self):
        """Test different bottleneck table options."""
        np.random.seed(42)
        df = nca_random(n=50, intercepts=[0.2], slopes=[0.7])
        
        # Test different bottleneck_x and bottleneck_y options
        for bn_x in ['percentage.range', 'percentage.max', 'actual']:
            for bn_y in ['percentage.range', 'percentage.max', 'actual']:
                model = nca_analysis(
                    df, 'X', 'Y',
                    ceilings=['ce_fdh'],
                    bottleneck_x=bn_x,
                    bottleneck_y=bn_y
                )
                assert 'bottlenecks' in model
                assert 'ce_fdh' in model['bottlenecks']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_sample_size(self):
        """Test with minimum sample size."""
        np.random.seed(42)
        df = nca_random(n=10, intercepts=[0.2], slopes=[0.7])
        
        model = nca_analysis(df, 'X', 'Y', ceilings=['ce_fdh'])
        assert 'summaries' in model

    def test_large_sample_size(self):
        """Test with larger sample size."""
        np.random.seed(42)
        df = nca_random(n=500, intercepts=[0.2], slopes=[0.7])
        
        model = nca_analysis(df, 'X', 'Y', ceilings=['ce_fdh'])
        assert 'summaries' in model

    def test_different_distributions(self):
        """Test with different distributions."""
        np.random.seed(42)
        
        # Uniform X, Normal Y
        df1 = nca_random(
            n=50,
            intercepts=[0.2],
            slopes=[0.7],
            distribution_x='uniform',
            distribution_y='normal'
        )
        model1 = nca_analysis(df1, 'X', 'Y', ceilings=['ce_fdh'])
        assert 'summaries' in model1
        
        # Normal X, Uniform Y
        df2 = nca_random(
            n=50,
            intercepts=[0.2],
            slopes=[0.7],
            distribution_x='normal',
            distribution_y='uniform'
        )
        model2 = nca_analysis(df2, 'X', 'Y', ceilings=['ce_fdh'])
        assert 'summaries' in model2

    def test_steep_and_gentle_slopes(self):
        """Test with various slope values."""
        np.random.seed(42)
        
        # Steep slope
        df1 = nca_random(n=50, intercepts=[0.05], slopes=[0.95])
        model1 = nca_analysis(df1, 'X', 'Y', ceilings=['ce_fdh'])
        assert 'summaries' in model1
        
        # Gentle slope
        df2 = nca_random(n=50, intercepts=[0.4], slopes=[0.3])
        model2 = nca_analysis(df2, 'X', 'Y', ceilings=['ce_fdh'])
        assert 'summaries' in model2


class TestRealWorldScenarios:
    """Test scenarios that mimic real-world usage."""

    def test_management_research_scenario(self):
        """Simulate a typical management research analysis."""
        np.random.seed(42)
        
        # Generate data simulating leadership -> performance relationship
        df = nca_random(
            n=150,
            intercepts=[0.2],
            slopes=[0.7],
            distribution_x='normal',
            distribution_y='normal'
        )
        df.columns = ['Leadership', 'Performance']
        
        # Run NCA
        model = nca_analysis(
            df, 'Leadership', 'Performance',
            ceilings=['ce_fdh', 'cr_fdh'],
            test_rep=10
        )
        
        # Get effect size
        params = model['summaries']['Leadership']['params']
        assert len(params) > 0
        
        # Check bottleneck table
        assert 'ce_fdh' in model['bottlenecks']

    def test_multiple_predictors_scenario(self):
        """Simulate analysis with multiple predictors."""
        np.random.seed(42)
        
        # Generate data with 4 predictors
        df = nca_random(
            n=200,
            intercepts=[0.15, 0.2, 0.1, 0.25],
            slopes=[0.75, 0.65, 0.8, 0.55]
        )
        df.columns = ['Training', 'Experience', 'Education', 'Motivation', 'Performance']
        
        # Run NCA
        model = nca_analysis(
            df, 
            ['Training', 'Experience', 'Education', 'Motivation'], 
            'Performance',
            ceilings=['ce_fdh']
        )
        
        # Verify all predictors analyzed
        for predictor in ['Training', 'Experience', 'Education', 'Motivation']:
            assert predictor in model['summaries']

    def test_publication_ready_output(self):
        """Test generating publication-ready output."""
        np.random.seed(42)
        df = nca_random(n=100, intercepts=[0.15], slopes=[0.75])
        
        model = nca_analysis(
            df, 'X', 'Y',
            ceilings=['ce_fdh', 'ols'],  # Use stable ceilings
            steps=10,
            test_rep=50
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import matplotlib
            matplotlib.use('Agg')
            
            # Generate all outputs
            nca_output(
                model,
                plots=True,
                summaries=True,
                bottlenecks=True,
                test=True,
                pdf=True,
                path=tmpdir
            )
            
            # Verify comprehensive output
            pdf_files = [f for f in os.listdir(tmpdir) if f.endswith('.pdf')]
            assert len(pdf_files) >= 3  # Summary, plot, bottleneck at minimum


class TestVersionAndImports:
    """Test package metadata and imports."""

    def test_version_string(self):
        """Test version is properly set."""
        import nca
        assert hasattr(nca, '__version__')
        assert nca.__version__ == '1.0.0'

    def test_author_info(self):
        """Test author info is set."""
        import nca
        assert hasattr(nca, '__author__')
        assert 'Gerandi Matraku' in nca.__author__

    def test_public_api(self):
        """Test all public functions are accessible."""
        import nca
        
        # Check all public functions are available
        assert hasattr(nca, 'nca_analysis')
        assert hasattr(nca, 'nca_output')
        assert hasattr(nca, 'nca_random')
        assert hasattr(nca, 'nca_outliers')
        assert hasattr(nca, 'nca_power')
        
        # Check they are callable
        assert callable(nca.nca_analysis)
        assert callable(nca.nca_output)
        assert callable(nca.nca_random)
        assert callable(nca.nca_outliers)
        assert callable(nca.nca_power)
