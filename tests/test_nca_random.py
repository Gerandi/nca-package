"""Tests for nca_random() function - data generation."""

import pytest
import numpy as np
import pandas as pd

from nca import nca_random


class TestNcaRandomBasic:
    """Basic functionality tests for nca_random."""

    def test_basic_generation(self):
        """Test basic data generation with default parameters."""
        df = nca_random(n=50, intercepts=0.2, slopes=0.8)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert 'X' in df.columns
        assert 'Y' in df.columns
    
    def test_sample_sizes(self):
        """Test different sample sizes."""
        for n in [10, 50, 100]:
            df = nca_random(n=n, intercepts=0.3, slopes=0.7)
            assert len(df) == n
    
    def test_data_range(self):
        """Test that generated data is within [0, 1] bounds."""
        df = nca_random(n=100, intercepts=0.2, slopes=0.8)
        
        assert df['X'].min() >= 0
        assert df['X'].max() <= 1
        assert df['Y'].min() >= 0
        assert df['Y'].max() <= 1


class TestNcaRandomDistributions:
    """Test different distribution options."""

    def test_uniform_distribution(self):
        """Test uniform distribution for X and Y."""
        df = nca_random(
            n=100, 
            intercepts=0.2, 
            slopes=0.8,
            distribution_x='uniform',
            distribution_y='uniform'
        )
        
        assert len(df) == 100
        # Uniform should have reasonable spread
        assert df['X'].std() > 0.1
        assert df['Y'].std() > 0.1
    
    def test_normal_distribution(self):
        """Test normal distribution for X and Y."""
        df = nca_random(
            n=100, 
            intercepts=0.2, 
            slopes=0.8,
            distribution_x='normal',
            distribution_y='normal',
            mean_x=0.5,
            mean_y=0.5,
            sd_x=0.2,
            sd_y=0.2
        )
        
        assert len(df) == 100
        # Data should still be within bounds due to truncation
        assert df['X'].min() >= 0
        assert df['X'].max() <= 1
    
    def test_mixed_distributions(self):
        """Test mixed distributions (uniform X, normal Y)."""
        df = nca_random(
            n=100, 
            intercepts=0.2, 
            slopes=0.8,
            distribution_x='uniform',
            distribution_y='normal'
        )
        
        assert len(df) == 100


class TestNcaRandomCorners:
    """Test corner parameter options."""

    def test_corner_1(self):
        """Test corner 1 (upper left empty)."""
        df = nca_random(n=50, intercepts=0.2, slopes=0.8, corner=1)
        assert len(df) == 50
    
    def test_corner_2(self):
        """Test corner 2 (upper right empty) - requires negative slope."""
        df = nca_random(n=50, intercepts=0.8, slopes=-0.6, corner=2)
        assert len(df) == 50
    
    def test_corner_3(self):
        """Test corner 3 (lower left empty) - requires negative slope."""
        df = nca_random(n=50, intercepts=0.8, slopes=-0.6, corner=3)
        assert len(df) == 50
    
    def test_corner_4(self):
        """Test corner 4 (lower right empty)."""
        df = nca_random(n=50, intercepts=0.2, slopes=0.8, corner=4)
        assert len(df) == 50


class TestNcaRandomSlopesIntercepts:
    """Test various slope and intercept combinations."""

    def test_steep_slope(self):
        """Test steep slope."""
        df = nca_random(n=50, intercepts=0.1, slopes=0.9)
        assert len(df) == 50
    
    def test_gentle_slope(self):
        """Test gentle slope."""
        df = nca_random(n=50, intercepts=0.3, slopes=0.5)
        assert len(df) == 50
    
    def test_negative_slope(self):
        """Test negative slope."""
        df = nca_random(n=50, intercepts=0.8, slopes=-0.6, corner=2)
        assert len(df) == 50
    
    def test_multiple_slopes(self):
        """Test multiple X variables with different slopes."""
        df = nca_random(
            n=50, 
            intercepts=[0.2, 0.3], 
            slopes=[0.8, 0.7]
        )
        
        assert len(df) == 50
        assert 'X1' in df.columns
        assert 'X2' in df.columns
        assert 'Y' in df.columns


class TestNcaRandomErrors:
    """Test error handling."""

    def test_invalid_n(self):
        """Test error on invalid sample size."""
        with pytest.raises(ValueError):
            nca_random(n=0, intercepts=0.2, slopes=0.8)
    
    def test_invalid_distribution(self):
        """Test error on invalid distribution type."""
        with pytest.raises(ValueError):
            nca_random(n=50, intercepts=0.2, slopes=0.8, distribution_x='invalid')
    
    def test_invalid_combination(self):
        """Test error on invalid slope/intercept combination."""
        with pytest.raises(ValueError):
            nca_random(n=50, intercepts=1.5, slopes=0.8)
    
    def test_mismatched_lengths(self):
        """Test error on truly mismatched intercepts/slopes lengths (neither broadcasts)."""
        with pytest.raises(ValueError):
            # 2 intercepts, 3 slopes - cannot broadcast
            nca_random(n=50, intercepts=[0.2, 0.3], slopes=[0.8, 0.7, 0.6])


class TestNcaRandomAttributes:
    """Test data attributes are correctly set."""

    def test_uniform_attributes(self):
        """Test attributes for uniform distribution."""
        df = nca_random(
            n=50, 
            intercepts=0.2, 
            slopes=0.8,
            distribution_x='uniform',
            distribution_y='uniform'
        )
        
        assert df.attrs.get('distribution.x') == 'uniform'
        assert df.attrs.get('distribution.y') == 'uniform'
    
    def test_normal_attributes(self):
        """Test attributes for normal distribution."""
        df = nca_random(
            n=50, 
            intercepts=0.2, 
            slopes=0.8,
            distribution_x='normal',
            distribution_y='normal',
            mean_x=0.5,
            mean_y=0.5,
            sd_x=0.2,
            sd_y=0.2
        )
        
        assert df.attrs.get('distribution.x') == 'normal'
        assert df.attrs.get('distribution.y') == 'normal'
        assert df.attrs.get('mean.x') == 0.5
        assert df.attrs.get('sd.x') == 0.2
