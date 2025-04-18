import numpy as np
from scipy.linalg import solve_continuous_are
from src.utils.helpers import get_signal_info, append_new_values
from src.model.Model import Signal
from src.algorithm.Algorithm import Algorithm

class LQR(Algorithm):
    def __init__(self, A, B, Q, R, model_signals=None):
        super().__init__(model_signals)
        """
        Initialize the LQR controller.

        Parameters:
        A (numpy.ndarray): System dynamics matrix.
        B (numpy.ndarray): Input matrix.
        Q (numpy.ndarray): State cost matrix.
        R (numpy.ndarray): Input cost matrix.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = None
        self.P = None
        self.gain = None

        self.compute_gains()
        if model_signals is not None:
            self.create_gain_description(model_signals)

    def compute_gains(self):
        """
        Compute the optimal gain matrix K and solution to the Riccati equation P.
        """
        # Solve the continuous-time algebraic Riccati equation (ARE)
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Compute the LQR gain K
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P

    def control_input(self, x):
        """
        Compute the control input based on the current state x.

        Parameters:
        x (numpy.ndarray): Current state vector.

        Returns:
        numpy.ndarray: Control input vector.
        """
        if self.K is None:
            raise ValueError("The gain matrix K has not been computed.")

        return self.K @ x

    def update_state_matrices(self, A, B):
        """
        Update the state-space matrices and recompute the LQR gains.

        Parameters:
        A (numpy.ndarray): New system dynamics matrix.
        B (numpy.ndarray): New input matrix.
        """
        self.A = A
        self.B = B
        self.compute_gains()

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

