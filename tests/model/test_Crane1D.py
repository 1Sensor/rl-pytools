import pytest
import numpy as np
from scipy.constants import g
from src.model.Crane1D import Crane1D


@pytest.fixture
def crane():
    """Fixture to create a Crane1D instance for testing."""
    return Crane1D()


def test_initial_state(crane):
    """Test if the initial state is set correctly."""
    expected_state = np.array([0, 0, 0, 0, 0.1, 0])  # Initial state based on the minimum values
    np.testing.assert_array_equal(crane.get_state(), expected_state)


def test_set_and_get_state(crane):
    """Test setting and getting the state."""
    new_state = np.array([1, 0.5, 0, 0, 0.5, 0])
    crane.set_state(new_state)
    np.testing.assert_array_equal(crane.get_state(), new_state)


def test_set_state_out_of_bounds(crane):
    """Test if ValueError is raised when setting an out-of-bounds state."""
    with pytest.raises(ValueError):
        crane.set_state(np.array([3, 0, 0, 0, 0.5, 0]))  # Exceeds max for 'Cart position'


def test_simulate(crane):
    """Test the simulate method."""
    u = [1, 1]
    dt = 0.1
    input_data, output_data = crane.simulate(u=u, dt=dt)
    #[0.00425885, 0.07179626, -0.03888268, -0.57694394, 0.10125, 0.025]
    expected_output = [0.0043, 0.0718, -0.0389, -0.5769, 0.1013, 0.025]
    assert output_data is not None, "Simulation output is None!"
    np.testing.assert_array_equal(np.round(output_data, 4), expected_output)
    assert crane.get_state() is not None, "State after simulation is None!"

def test_update_matrices(crane):
    """Test if the matrices are updated correctly."""
    crane.update_matrices(payload_mass=3, sling_length=0.8)
    cart_mass = crane.get_param('Cart mass')
    # Check if the matrices are updated correctly
    expected_a = [[0, 1, 0, 0, 0, 0],
                  [0, 0, 3 * g / cart_mass, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, -g*(cart_mass + 3) / cart_mass / 0.8, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0]]
    expected_b = [[0, 0],
                  [1 / cart_mass, 0],
                  [0, 0],
                  [-1 / cart_mass / 0.8, 0],
                  [0, 0],
                  [0, 1 / 2 / 3]]

    np.testing.assert_array_equal(crane.sys.A, expected_a)
    np.testing.assert_array_equal(crane.sys.B, expected_b)


def test_set_param(crane):
    """Test setting parameter values."""
    crane.set_param('Cart mass', 2)
    assert crane.get_param('Cart mass') == 2


if __name__ == "__main__":
    pytest.main()
