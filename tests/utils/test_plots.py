import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.utils.plots import *


@pytest.fixture
def sample_data():
    """Fixture to create sample DataFrame for testing."""
    return pd.DataFrame({
        'Signal1': np.random.rand(100),
        'Signal2': np.random.rand(100),
        'Signal3': np.random.rand(100)
    })

@pytest.fixture
def sample_signals():
    """Fixture to create sample signals data."""
    return {
        'Signal1': {'unit': 'V'},
        'Signal2': {'unit': 'A'},
        'Signal3': {'unit': 'm'},
    }

def test_plot_signals_default_shape(sample_data, sample_signals):
    """Test plotting with default shape."""
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        plot_signals(sample_data, sample_signals, 'test_plot')

        # Check if savefig was called
        mock_savefig.assert_called_once_with('..\\..\\results\\test_plot.pdf')

def test_plot_signals_custom_shape(sample_data, sample_signals):
    """Test plotting with custom figure shape."""
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        plot_signals(sample_data, sample_signals, 'test_plot_custom_shape', fig_shape=(2, 2))

        # Check if savefig was called with the expected filename
        mock_savefig.assert_called_once_with('..\\..\\results\\test_plot_custom_shape.pdf')

def test_find_signal_info_called(sample_data, sample_signals):
    """Test if find_signal_info is called correctly."""
    with patch('src.utils.helpers.find_signal_info') as mock_find_signal_info:
        mock_find_signal_info.return_value = 'unit'

        plot_signals(sample_data, sample_signals, 'test_plot_find_signal_info')

        # Check if find_signal_info was called for each signal
        assert mock_find_signal_info.call_count == len(sample_data.columns)

def test_invalid_data_type():
    """Test if function raises error on invalid data type."""
    with pytest.raises(AttributeError):
        plot_signals("invalid_data", {}, 'test_invalid_plot')

def test_invalid_signals_type(sample_data):
    """Test if function raises error on invalid signals type."""
    with pytest.raises(TypeError):
        plot_signals(sample_data, "invalid_signals", 'test_invalid_signals')
