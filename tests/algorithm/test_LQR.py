import pytest
import numpy as np
from src.algorithm.LQR import LQR
from src.model.Model import Signal
from unittest.mock import MagicMock


# Testy dla klasy LQR

@pytest.fixture
def lqr():
    # Tworzenie przykładowego obiektu LQR przed każdym testem
    A = np.array([[0, 0], [0, 1]])
    B = np.array([[1], [1]])
    Q = np.eye(2)
    R = np.eye(1)
    sig = [Signal('Sig1', 'N', 0, 1, 's_1'),
           Signal('Sig2', 'N', 0, 1, 's_2')]
    return LQR(A, B, Q, R, sig)


def test_initialization(lqr):
    """Test for initialization of LQR controller."""
    # Testujemy, czy obiekt LQR jest prawidłowo zainicjowany
    assert lqr.K is not None, "Gain matrix K was not computed."
    assert lqr.P is not None, "Riccati equation solution P was not computed."
    assert isinstance(lqr.K, np.ndarray), "K should be a numpy array."
    assert lqr.K.shape == (1, 2), "K shape is incorrect."


def test_compute_gains(lqr):
    """Test for computing the LQR gains."""
    # Sprawdzamy, czy obliczenia macierzy K i P są prawidłowe
    K_expected = np.array([[-1.0, 4.2360]])  # Przykładowa oczekiwana wartość
    np.testing.assert_array_almost_equal(lqr.K, K_expected, decimal=4, err_msg="Gain matrix K is incorrect.")

    # Testujemy rozwiązanie Riccati
    assert lqr.P is not None, "P should not be None."
    assert lqr.P.shape == (2, 2), "P matrix shape is incorrect."


def test_update_state_matrices(lqr):
    """Test for updating state-space matrices."""
    # Nowe macierze A i B
    A_new = np.array([[1, 1], [0, 1]])
    B_new = np.array([[0], [1]])

    lqr.update_state_matrices(A_new, B_new)

    # Sprawdzamy, czy macierze zostały zaktualizowane
    np.testing.assert_array_equal(lqr.A, A_new, "Matrix A was not updated.")
    np.testing.assert_array_equal(lqr.B, B_new, "Matrix B was not updated.")


def test_control_input(lqr):
    """Test for control input calculation."""
    x = np.array([1, 2])  # Przykładowy wektor stanu
    expected_control_input = lqr.K @ x  # Spodziewana wartość sterowania

    control_input = lqr.control_input(x)

    np.testing.assert_array_equal(control_input, expected_control_input, "Control input calculation is incorrect.")


def test_filter_gains(lqr):
    """Test for filtering gains close to zero."""
    lqr.K = np.array([[0.0001, 0.5], [0.3, -0.00005]])  # Wartości zbliżone do zera

    filtered_gains = lqr.filter_gains(tol=1e-4)

    # Sprawdzamy, czy wartości bliskie zeru zostały usunięte
    assert '$K_{s_1}$' in filtered_gains, "Filtered gains should contain '$K_{s_1}$'."
    assert '$K_{s_2}$' in filtered_gains, "Filtered gains should contain '$K_{s_2}$'."
    assert np.abs(filtered_gains['$K_{s_1}$']) > 0, "Gain $K_{s_1}$ should be non-zero after filtering."
    assert np.abs(filtered_gains['$K_{s_2}$']) > 0, "Gain $K_{s_2}$ should be non-zero after filtering."


def test_create_init_dict(lqr):
    """Test for creating init dictionary."""
    init_dict = lqr.create_init_dict()

    # Sprawdzamy, czy słownik został poprawnie utworzony
    assert isinstance(init_dict, dict), "Output should be a dictionary."
    assert '$K_{s_1}$' in init_dict, "Init dict should contain '$K_{s_1}$'."
    assert len(init_dict) > 0, "Init dict should not be empty."
