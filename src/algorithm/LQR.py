import numpy as np
from scipy.linalg import solve_continuous_are

class LQR:
    def __init__(self, A, B, Q, R):
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

        self.compute_gains()

    def compute_gains(self):
        """
        Compute the optimal gain matrix K and solution to the Riccati equation P.
        """
        # Solve the continuous-time algebraic Riccati equation (ARE)
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Compute the LQR gain K
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P

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
    lqr = LQR(obj.sys.A, obj.sys.B, Q, R)
    # Set desired parameters
    desired_x = 2   # [m]
    desired_l = 0.5 # [m]
    desired = np.array([desired_x, desired_l])

    u, y = obj.create_init_df()
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

        u = pd.concat([u, u_new], ignore_index=True)
        y = pd.concat([y, y_new], ignore_index=True)



    signals = [obj.input, obj.output]
    data_names = ['input', 'output']
    aggregate_simulation_data([u, y], signals, data_names)

