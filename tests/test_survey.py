"""Survey validation and analytic geometric factors, including remote electrodes."""

import math

import numpy as np
import pytest
import torch

from adtlert.survey import REMOTE_ELECTRODE, Survey

R = REMOTE_ELECTRODE
LINE = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])


def test_geometric_factors_drop_the_terms_of_a_remote_electrode():
    survey = Survey.from_arrays(
        LINE, [[0, 1, 2, 3], [0, R, 1, 2], [0, 1, 2, R], [0, R, 1, R]]
    )
    expected = [
        2 * math.pi / (1 / 2 - 1 / 3 - 1 + 1 / 2),
        2 * math.pi / (1 - 1 / 2),
        2 * math.pi / (1 / 2 - 1),
        2 * math.pi,
    ]
    np.testing.assert_allclose(survey.geometric_factors(), expected, rtol=1e-6)
    assert survey.measurement_slots().tolist()[3] == [0, 4, 1, 4]


@pytest.mark.parametrize(
    "quad, message",
    [
        ([R, R, 1, 2], "needs an electrode"),
        ([0, 1, R, R], "needs an electrode"),
        ([0, -2, 1, 2], "other than -1"),
        ([0, 0, 1, 2], "distinct"),
        ([0, 1, 2, 4], "outside the survey"),
    ],
)
def test_invalid_measurements_are_rejected(quad, message):
    with pytest.raises(ValueError, match=message):
        Survey.from_arrays(LINE, [quad])
