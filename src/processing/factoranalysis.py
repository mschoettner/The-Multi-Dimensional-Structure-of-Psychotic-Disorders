import numpy as np
import pandas as pd

def calculate_scores(data, loadings, uniquenesses, method = "bartlett"):
    """
    Calculate factor scores.
    """
    if method == "anderson":
        raise NotImplementedError
        #Z = data.to_numpy()
        U = np.diag(uniquenesses)
        U_inv = np.linalg.inv(U)
        #A = loadings
        #
        #f = Z@U_inv@A@G_inv_sq
        B_t = (P.T@U_inv@R@U_inv@P)
    if method == "regression":
        X = data.to_numpy()
        R = data.corr()
        R_inv = np.linalg.inv(R)
        P = loadings
        B = R_inv@P
        F = X@B
    if method == "bartlett":
        X = data.to_numpy()
        P = loadings
        U = np.diag(uniquenesses)
        U_inv = np.linalg.inv(U)
        one=P.T@U_inv
        two=P.T@U_inv@P
        B_t = np.linalg.inv(two)@one
        B = B_t.T
        F = X@B
    return F