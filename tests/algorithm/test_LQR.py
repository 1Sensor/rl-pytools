import pytest
import numpy as np
from src.algorithm.LQR import LQR  # Replace with the actual module name

def test_lqr_initialization():
    """
    Test initialization of the LQR controller and computation of gains.
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1]])

    lqr = LQR(A, B, Q, R)

    assert lqr.A.shape == (2, 2)
    assert lqr.B.shape == (2, 1)
    assert lqr.Q.shape == (2, 2)
    assert lqr.R.shape == (1, 1)
    assert lqr.K is not None
    assert lqr.P is not None

def test_control_input():
    """
    Test computation of the control input.
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1]])

    lqr = LQR(A, B, Q, R)

    x = np.array([1, 0])
    u = lqr.control_input(x)

    assert u.shape == (1,)
    assert np.allclose(u, -lqr.K @ x)

def test_update_state_matrices():
    """
    Test updating state-space matrices and recomputing gains.
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1]])

    lqr = LQR(A, B, Q, R)

    A_new = np.array([[0, 1], [-1, -1]])
    B_new = np.array([[0], [1]])

    lqr.update_state_matrices(A_new, B_new)

    assert np.allclose(lqr.A, A_new)
    assert np.allclose(lqr.B, B_new)
    assert lqr.K is not None
    assert lqr.P is not None

def test_invalid_control_input():
    """
    Test error handling when control input is called without computed gains.
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.array([[1]])

    lqr = LQR(A, B, Q, R)
    lqr.K = None  # Manually unset K to simulate error condition

    x = np.array([1, 0])

    with pytest.raises(ValueError, match="The gain matrix K has not been computed."):
        lqr.control_input(x)

def test_invalid_dimensions():
    """
    Test error handling for invalid matrix dimensions.
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array
