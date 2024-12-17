import subprocess
import pandas as pd


def get_signal_info(signals):
    """Creates a dataframe with signal information"""
    names = []
    units = []
    min_vals = []
    max_vals = []
    symbols = []

    for sig in signals:
        names.append(sig.name)
        units.append(sig.unit)
        min_vals.append(sig.min)
        max_vals.append(sig.max)
        symbols.append(sig.symbol)

    return pd.DataFrame({'name':names,
                         'unit':units,
                         'min': min_vals,
                         'max': max_vals,
                         'symbol': symbols})

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

def get_git_repo_path():
    try:
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True)
        result = result.stdout.strip()
        result = result.replace("/", "\\")
        return result
    except subprocess.CalledProcessError:
        return "You are not in Git repository."

def append_new_values(init_dict, new_dict):
    for key in init_dict:
        init_dict[key].append(new_dict[key])




