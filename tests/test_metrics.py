import numpy as np

from adl_repro.metrics import HistogramAUC, exact_auc, relative_improvement


def test_exact_auc_and_ties():
    assert exact_auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
    assert exact_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_histogram_auc():
    metric = HistogramAUC(bins=1000)
    metric.update(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metric.compute() == 1.0


def test_paper_relative_improvement():
    assert abs(relative_improvement(0.6179, 0.5948) - 0.2437) < 0.001
