#!/usr/bin/env python3
############################################
## \* PSM Surgical Instrument FK Utilities *\##########
## \author: Haoying Zhou
############################################

import os
import sys
import numpy as np
from typing import Union


def skew(vector: Union[np.array, list]) -> np.ndarray:
    """
    Calculates the skew bucket matrix from the given vector
    :param vector: any 3x1 vector
    :return: 3x3 skew-symmetric matrix
    """
    assert len(vector) == 3, "The vector should be a 3 by 1 vector"
    skew_mtx = np.zeros((3, 3))
    skew_mtx[0, 1] = -vector[2]
    skew_mtx[0, 2] = vector[1]
    skew_mtx[1, 0] = vector[2]
    skew_mtx[1, 2] = -vector[0]
    skew_mtx[2, 0] = -vector[1]
    skew_mtx[2, 1] = vector[0]
    return skew_mtx


def axisangle2rot(omega: np.ndarray, theta: float) -> np.ndarray:
    """
    Calculate the rotation matrix given the axis-angle representation, implementation based on Rodrigues' formula
    :param omega: axis of rotation, 3x1
    :param theta: rotation angle
    :return: 3x3 rotation matrix
    """
    omega_skew = skew(omega)
    R = (
        np.eye(3)
        + np.sin(theta) * omega_skew
        + (1 - np.cos(theta)) * np.dot(omega_skew, omega_skew)
    )
    return R


def twist2ht(S: list, q: float) -> np.ndarray:
    """
    Calculate the homogenous transformation matrix given the twist representation
    :param S: screw axis representation
    :param q: joint value (angle in rad, distance in meter)
    :return: 4x4 homogenous transformation matrix
    """
    omega = S[0:3]
    v = S[3:6]
    T = np.zeros((4, 4))
    T[3, 3] = 1
    if np.linalg.norm(omega) < 1e-8:
        R = np.eye(3)
        p = np.transpose(v) * q
    else:
        omega_skew = skew(omega)
        R = axisangle2rot(omega, q)
        p = np.dot(
            (
                np.eye(3) * q
                + (1 - np.cos(q)) * omega_skew
                + (q - np.sin(q)) * np.dot(omega_skew, omega_skew)
            ),
            v,
        )
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


if __name__ == "__main__":
    test = 0
