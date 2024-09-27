from scipy.integrate import odeint


def derivative(self, u, dt):
    u = np.array(u)

    _, columns_B = np.array(self.ss.B).shape
    if columns_B != u.shape[0]:
        raise ValueError("Inputs shape not equal!")

    def state_derivative(x, t, A, B, u):
        # Compute the state derivatives
        dxdt = A @ x + B @ u  # Use matrix multiplication
        return dxdt

    # Initial conditions for the state variables
    x0 = self.state  # Initial state vector
    # Define the time span for the simulation
    t = np.array([0, dt])  # From 0 to 10 seconds, 1000 points
    # Solve the differential equations using odeint
    x = odeint(state_derivative, x0, t, args=(self.ss.A, self.ss.B, u))
    # Calculate the output
    y = np.array([self.ss.C @ xi + self.ss.D @ u for xi, ti in zip(x, t)])  # Output computation