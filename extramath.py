"""Extra math functions"""
# Author: Ben Gravell

import numpy as np


def symlog(X,scale=1):
    """Symmetric log transform"""
    return np.multiply(np.sign(X),np.log(1+np.abs(X)/(10**scale)))