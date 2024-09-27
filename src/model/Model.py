import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
from scipy.signal import lsim


@dataclass
class Signal:
    name: str
    unit: str
    min:  float
    max:  float

@dataclass
class Parameter:
    name: str
    unit: str
    value:  float


class Model:
    def __init__(self, init_state=None):
        self.state = None
        if init_state is None:
            raise ValueError("No initial state provided!")
        self.init_state(init_state)
        self.update_matrices()


    @abstractmethod
    def update_matrices(self):
        """Abstract method to calculate state-space matrixes"""
        pass

    def init_state(self, init_state):
        """Function run at the init of the object, set initial state"""
        self.state = np.array(init_state)

    def check_state(self, new_state=None):
        if new_state is None:
            new_state = self.state
        min_state = np.array([param.min for param in self.signals])
        max_state = np.array([param.max for param in self.signals])
        if any(new_state < min_state) or any(new_state > max_state):
            raise ValueError("State out of bounds!")

    def set_state(self, new_state):
        self.check_state(new_state)
        self.state = new_state

    def get_state(self):
        return self.state

    def simulate(self, u, dt):
        # Ensure u is a 2D array and repeat the input for each time step
        u = np.array([u, u])  # Repeat the input u for two time steps (t[0] and t[1])

        # Check the dimensions of the input matrix u
        _, columns_b = np.array(self.sys.B).shape
        if columns_b != u.shape[1]:
            raise ValueError("Inputs shape not equal!")

        # Define the time span for the simulation (2 time steps)
        t = np.array([0, dt])

        # Solve the differential equations and calculate output
        t, y, state = lsim(self.sys, u, t, X0=self.state)

        # Update the internal state to the new state after the simulation
        self.state = state[-1]  # Take the last state (after dt)
        return y[-1]  # Return the output at the last time step

    def get_param(self, parameter_name):
        for p in self.parameters:
            if parameter_name == p.name:
                return p.value

    def set_param(self, parameter_name, value):
        for p in self.parameters:
            if parameter_name == p.name:
                p.value = value
                #self.update_matrices()