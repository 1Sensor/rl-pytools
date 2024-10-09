from signal import signal

import pandas as pd


def get_signal_info(signals):
    """Creates a dataframe with signal information"""
    names = []
    units = []
    min_vals = []
    max_vals = []

    for sig in signals:
        names.append(sig.name)
        units.append(sig.unit)
        min_vals.append(sig.min)
        max_vals.append(sig.max)

    return pd.DataFrame({'name':names,
                         'unit':units,
                         'min': min_vals,
                         'max': max_vals})

def find_signal_info(signals, signal_name, signal_attribute):
    for sig in signals:
        if sig.name == signal_name:
            match signal_attribute:
                case 'unit':
                    attribute = sig.unit
                case 'min':
                    attribute = sig.min
                case 'max':
                    attribute = sig.max
    return attribute


