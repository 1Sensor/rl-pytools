import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
import pandas as pd
from scipy.signal import lsim
from src.utils.helpers import get_signal_info

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
        self._init_state = init_state
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
        min_state = np.array([param.min for param in self.output])
        max_state = np.array([param.max for param in self.output])
        if any(new_state < min_state) or any(new_state > max_state):
            raise ValueError("State out of bounds!")

    def set_state(self, new_state):
        self.check_state(new_state)
        self.state = new_state

    def get_state(self):
        return self.state

    def simulate(self, u, dt=None, t=None):
        if t is None:
            # Ensure u is a 2D array and repeat the input for each time step
            u = np.array([u, u])  # Repeat the input u for two time steps (t[0] and t[1])
            # Define the time span for the simulation (2 time steps)
            t = np.array([0, dt])
        else:
            if len(u) == len(self.input):
                u = np.array([u for ti in t])
            else:
                u = np.array(u)
        # Check the dimensions of the input matrix u
        _, columns_b = np.array(self.sys.B).shape
        if columns_b != u.shape[1]:
            raise ValueError("Inputs shape not equal!")

        # Solve the differential equations and calculate output
        t, y, states = lsim(self.sys, u, t, X0=self.state)

        # Update the internal state to the new state after the simulation
        self.state = states[-1]  # Take the last state (after dt)

        # Create input dataframe
        signal_info = get_signal_info(self.input)
        input_data = pd.DataFrame(columns=signal_info['name'], data=u)
        # Create output dataframe
        signal_info = get_signal_info(self.output)
        if dt is not None:
            y = y[-1]
            y = y.reshape([1,6])
            y = y.flatten()
        output_data = pd.DataFrame(columns=signal_info['name'], data=y)

        return input_data, output_data  # Return the output at the last time step

    def get_param(self, parameter_name):
        for p in self.parameters:
            if parameter_name == p.name:
                return p.value

    def set_param(self, parameter_name, value):
        for p in self.parameters:
            if parameter_name == p.name:
                p.value = value
                #self.update_matrices()