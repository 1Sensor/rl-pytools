from src.utils.plots import plot_signals


def aggregate_simulation_data(control, results, obj):
    plot_signals(data=control, signals=obj.input, plot_name="input")
    plot_signals(data=results, signals=obj.output, plot_name="output")