import numpy as np
from abc import ABC, abstractmethod
from src.utils.helpers import get_signal_info, append_new_values
from src.model.Model import Signal


class Algorithm(ABC):
    def __init__(self, model_signals):
        self.gain = None
        self.K = None

        if model_signals is not None:
            self.create_gain_description(model_signals)

    @abstractmethod
    def compute_gains(self):
        """
        Compute the optimal gain matrix K and solution to the Riccati equation P.
        """
        pass

    @abstractmethod
    def control_input(self):
        """
        Compute the control input.
        """
        pass

    def create_gain_description(self, model_signals):
        """
        Create gain variable with description of signals in the
        form of Signal dataclass.

        Parameters:
        model_signals (dataclass Signal): Model signals information.
        """
        signal_info = get_signal_info(model_signals)
        self.gain = [Signal('$K_{'+ symbol + '}$', '-', -np.inf, np.inf, '-') for symbol in signal_info['symbol']]

    def create_init_dict(self):
        """
        Create empty dataframe for collection of gain values.

        Parameters:
        signals (dataclass Signal): Information about the signals of the model.

        Returns:
        dict: Empty dict of gain values.
        """
        signal_info = get_signal_info(self.gain)
        return {name: [] for name in signal_info['name']}

    def filter_gains(self, tol=1e-10):
        """
        Flattens an ndarray of gains and removes values close to zero.

        Parameters:
        - tol: float - tolerance for values close to zero (default is 1e-10)

        Returns:
        - numpy.ndarray - a flattened array without values close to zero
        """
        flattened = self.K.flatten()
        filtered = np.round(flattened[np.abs(flattened) > tol], 4)
        gain_info = get_signal_info(self.gain)
        return dict(zip(gain_info['name'].to_list(), filtered))

# Example usage
if __name__ == "__main__":
    from src.model.Crane1D import Crane1D
    from src.utils.aggregation import aggregate_simulation_data
    import pandas as pd

    # Example system parameters
    obj = Crane1D()
    Q = np.eye(6)
    R = np.eye(2)
    dt = 0.1

    # Create LQR controller
    lqr = LQR(obj.sys.A, obj.sys.B, Q, R, obj.output)
    # Set desired parameters
    desired_x = 2   # [m]
    desired_l = 0.5 # [m]
    desired = np.array([desired_x, desired_l])

    u, y = obj.create_init_dict()
    k = lqr.create_init_dict()
    # Control loop
    for i in range(100):

        # Compute control input
        feedback = lqr.control_input(obj.state)
        control = desired - feedback

        # Simulate object
        u_new, y_new = obj.simulate(control, dt=dt)

        # Update state-space matrices
        obj.update_matrices(sling_length=obj.state[4])
        A_new = obj.sys.A
        B_new = obj.sys.B
        lqr.update_state_matrices(A_new, B_new)

        append_new_values(u, u_new)
        append_new_values(y, y_new)
        append_new_values(k, lqr.filter_gains())

    u = pd.DataFrame(u)
    y = pd.DataFrame(y)
    k = pd.DataFrame(k)
    signals = [obj.input, obj.output, lqr.gain]
    data_names = ['input', 'output', 'gain']
    aggregate_simulation_data([u, y, k], signals, data_names)

