from src.utils.plots import plot_signals
from src.utils.helpers import get_git_repo_path
import datetime
import os


def create_new_directory_and_get_its_path():
    """
    Check repository path and create new directory in results for simulation data
    :return: path to the newly created directory for saving files from simulation
    """
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    path = get_git_repo_path()
    save_path = path + '\\results\\' + formatted_datetime + '\\'
    os.mkdir(save_path)
    return save_path

def aggregate_simulation_data(simulation_data, signals, data_names):
    """
    Aggregate all simulation data, create plots and signals trace
    :param simulation_data: array of df with signal trace and its names
    :param signals: array of signals descriptions
    :param plot_names: array of name of the plot to be saved
    """
    save_path = create_new_directory_and_get_its_path()

    for data, signal, name in zip(simulation_data, signals, data_names):
        file_save_path = save_path + name
        plot_signals(data=data, signals=signal, save_path=file_save_path)
        data.to_csv(file_save_path + '.csv')
        data.to_excel(file_save_path + '.xlsx')