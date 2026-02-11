
import numpy as np
from src.mo_utils import pareto_filter, hypervolume_2d, igd

def test_pareto_filter_basic():
    pts = np.array([[1,1],[2,2],[1,2]])
    nd = pareto_filter(pts)
    assert nd.shape[0] == 1

def test_hv_2d_positive():
    pts = np.array([[0.2, 0.3], [0.4, 0.2]])
    hv = hypervolume_2d(pts, ref=[1.0,1.0])
    assert hv > 0

def test_igd_zero_when_same():
    A = np.array([[0.1,0.2],[0.2,0.1]])
    assert igd(A, A) == 0.0
