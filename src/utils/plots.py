from src.utils.helpers import find_signal_info
import matplotlib.pyplot as plt


# Set constants for plots
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16

# Set global font properties
plt.rcParams['font.family'] = 'serif'  # or 'sans-serif', 'monospace', etc.
plt.rcParams['font.serif'] = ['Times New Roman']  # Specify the font name if using serif

def plot_signals(data, signals, plot_name, fig_shape=None):
    signal_no = len(signals)
    if fig_shape is None:
        ncols = 2
        if signal_no % ncols:
            nrows = int(signal_no // ncols + 1)
        else:
            nrows = int(signal_no / ncols)

    else:
        nrows = fig_shape[0]
        ncols = fig_shape[1]

    fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 5 * nrows))
    # Flatten the 2D array of subplots to simplify indexing
    axs = axs.flatten()

    for i, sig_name in enumerate(data.columns):
        axs[i].plot(data[sig_name])
        axs[i].set_xlabel('Time [s]', fontsize=LABEL_FONTSIZE)
        unit = find_signal_info(signals, sig_name, 'unit')
        axs[i].set_ylabel(sig_name + ' [' + unit + ']', fontsize=LABEL_FONTSIZE)
        axs[i].grid(color='gray', linestyle='--', linewidth=0.5)
        axs[i].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

    plt.savefig('..\\..\\results\\' + plot_name + ".pdf")