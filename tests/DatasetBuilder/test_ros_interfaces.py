import pytest
from ambf6dpose.DataCollection.RosClients import RawSimulationData
import numpy as np


def test_RawSimulationDataConstructor():
    try:
        raw_data = RawSimulationData(
            np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
        )
    except ValueError:
        pytest.fail("RawSimulationData constructor failed")


def test_RawSimulationDataConstructorWithNone():
    try:
        raw_data = RawSimulationData(None, None, None, np.zeros((3, 3)))
    except ValueError:
        return True

    pytest.fail("RawSimulationData constructor failed")
