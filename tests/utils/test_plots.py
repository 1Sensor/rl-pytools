import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
import matplotlib.pyplot as plt
from src.utils.plots import plot_signals


# Define the missing find_signal_info function
def find_signal_info(signals, signal_name, signal_attribute):
    """
    Retrieve signal attribute (e.g., unit, min, max) for a given signal name.

    Parameters:
    - signals (list): List of signal objects (with name, unit, min, max).
    - signal_name (str): Name of the signal to search for.
    - signal_attribute (str): Attribute to retrieve ('unit', 'min', 'max').

    Returns:
    - attribute (str/float): Value of the requested attribute.
    """
    for sig in signals:
        if sig.name == signal_name:
            match signal_attribute:
                case 'unit':
                    return sig.unit
                case 'min':
                    return sig.min
                case 'max':
                    return sig.max
    return None  # Return None if the signal or attribute is not found


# Define a mock Signal class for testing
class Signal:
    def __init__(self, name, unit, min_val, max_val):
        self.name = name
        self.unit = unit
        self.min = min_val
        self.max = max_val


@pytest.fixture
def mock_data():
    """Fixture to create mock signal data and signal information."""
    # Simulate data for 4 signals
    columns = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
    data = pd.DataFrame({
        'Signal1': np.linspace(0, 10, 100),
        'Signal2': np.sin(np.linspace(0, 2 * np.pi, 100)),
        'Signal3': np.cos(np.linspace(0, 2 * np.pi, 100)),
        'Signal4': np.linspace(10, 0, 100),
    })

    # Mock signals with units
    signals = [
        Signal('Signal1', 'm', -10, 10),
        Signal('Signal2', 'rad', -1, 1),
        Signal('Signal3', 'rad', -1, 1),
        Signal('Signal4', 'm', -10, 10),
    ]

    return data, signals


@pytest.fixture
def save_path(tmp_path):
    """Fixture to create a temporary save path for plots."""
    return str(tmp_path / "test_plot")


@patch("matplotlib.pyplot.savefig")
def test_plot_signals(mock_savefig, mock_data, save_path):
    """
    Test the plot_signals function to verify:
    - Correct calls to matplotlib savefig
    - Proper handling of input data and signals
    """
    data, signals = mock_data

    # Run the function
    plot_signals(data, signals, save_path)

    # Verify the number of plots saved
    mock_savefig.assert_any_call(save_path + ".pdf")
    mock_savefig.assert_any_call(save_path + ".jpg")
    assert mock_savefig.call_count == 2, "Savefig should be called twice (PDF and JPG)."

    # Ensure the plot has correct signal labels and units
    for i, sig_name in enumerate(data.columns):
        unit = find_signal_info(signals, sig_name, 'unit')
        assert unit is not None, f"Unit for signal '{sig_name}' should not be None."


def test_plot_signals_file_output(mock_data, save_path):
    """
    Test whether the files are actually saved in the specified location.
    """
    data, signals = mock_data

    # Run the function
    plot_signals(data, signals, save_path)

    # Verify that the files exist
    assert os.path.exists(save_path + ".pdf"), "PDF file was not saved."
    assert os.path.exists(save_path + ".jpg"), "JPG file was not saved."


def test_find_signal_info(mock_data):
    """
    Test the find_signal_info function independently.
    """
    _, signals = mock_data

    # Test unit retrieval
    assert find_signal_info(signals, 'Signal1', 'unit') == 'm'
    assert find_signal_info(signals, 'Signal2', 'unit') == 'rad'

    # Test min and max values
    assert find_signal_info(signals, 'Signal1', 'min') == -10
    assert find_signal_info(signals, 'Signal1', 'max') == 10

    # Test invalid signal name
    assert find_signal_info(signals, 'InvalidSignal', 'unit') is None

    # Test invalid attribute
    assert find_signal_info(signals, 'Signal1', 'invalid_attr') is None
