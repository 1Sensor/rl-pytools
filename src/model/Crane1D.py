import numpy as np
from scipy.constants import g, pi
from scipy.signal import StateSpace
from src.model.Model import Model, Signal, Parameter


class Crane1D(Model):
    def __init__(self):
        self.sys = None
        self.signals = [Signal('Cart position', 'm', 0, 2),
                        Signal('Cart velocity', 'm/s', -1, 1),
                        Signal('Sway angle', 'rad', -pi, pi),
                        Signal('Angular velocity', 'rad/s', -np.inf, np.inf),
                        Signal('Sling length', 'm', 0.1, 1),
                        Signal('Sling length changing speed', 'm/s', -1, 1)]
        self.parameters = [Parameter('Cart mass', 'kg', 1),
                           Parameter('Payload mass', 'kg', 2)]
        init_state = [0, 0, 0, 0, 0.1, 0]
        super().__init__(init_state)

    def update_matrices(self, payload_mass=None, sling_length=None):
        if payload_mass is None:
            payload_mass = self.get_param('Payload mass')
        if sling_length is None:
            sling_length = self.state[4]
        cart_mass = self.get_param('Cart mass')
        self.set_param('Payload mass', payload_mass)
        self.state[4] = sling_length


        matrix_a = [[0, 1, 0, 0, 0, 0],
             [0, 0, payload_mass * g / cart_mass, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, -g*(cart_mass + payload_mass) / cart_mass / sling_length, 0, 0, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0]]

        matrix_b = [[0, 0],
             [1 / cart_mass, 0],
             [0, 0],
             [-1 / cart_mass / sling_length, 0],
             [0, 0],
             [0, 1 / 2 / payload_mass]]

        matrix_c = [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]]

        matrix_d = [[0, 0],
             [0, 0],
             [0, 0],
             [0, 0],
             [0, 0],
             [0, 0]]

        self.sys = StateSpace(matrix_a, matrix_b, matrix_c, matrix_d)
        