"""General matrix math functions."""
# Author: Ben Gravell

import numpy as np
from numpy import linalg as la
from scipy.linalg import solve_discrete_lyapunov,solve_discrete_are
from functools import reduce


def vec(A):
    """Return the vectorized matrix A by stacking its columns."""
    return A.reshape(-1, order="F")


def sympart(A):
    """Return the symmetric part of matrix A."""
    return 0.5*(A+A.T)


def is_pos_def(A):
    """Check if matrix A is positive definite."""
    try:
        la.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def succ(A,B):
    """Check the positive definite partial ordering of A > B."""
    return is_pos_def(A-B)


def psdpart(X):
    """Return the positive semidefinite part of a symmetric matrix."""
    X = sympart(X)
    Y = np.zeros_like(X)
    eigvals, eigvecs = la.eig(X)
    for i in range(X.shape[0]):
        if eigvals[i] > 0:
            Y += eigvals[i]*np.outer(eigvecs[:,i],eigvecs[:,i])
    Y = sympart(Y)
    return Y


def kron(*args):
    """Overload and extend the numpy kron function to take a single argument."""
    if len(args)==1:
        return np.kron(args[0],args[0])
    else:
        return np.kron(*args)


def mdot(*args):
    """Multiple dot product."""
    return reduce(np.dot, args)


def mip(A,B):
    """Matrix inner product of A and B."""
    return np.trace(mdot(A.T,B))


def specrad(A):
    """Spectral radius of matrix A."""
    try:
        return np.max(np.abs(la.eig(A)[0]))
    except np.linalg.LinAlgError:
        return np.nan


def printeigs(A):
    """Print all eigenvalues of matrix A."""
    print(la.eig(A)[0])
    return


def minsv(A):
    """Minimum singular value."""
    return la.svd(A)[1].min()


def solveb(a,b):
    """
    Solve a = bx.
    Similar to MATLAB / operator for square invertible matrices.
    """
    return la.solve(b.T,a.T).T


def lstsqb(a,b):
    """
    Return least-squares solution to a = bx.
    Similar to MATLAB / operator for rectangular matrices.
    """
    return la.lstsq(b.T,a.T,rcond=None)[0].T


def dlyap(A,Q):
    """
    Solve the discrete-time Lyapunov equation.
    Wrapper around scipy.linalg.solve_discrete_lyapunov.
    Pass a copy of input matrices to protect them from modification.
    """
    try:
        return solve_discrete_lyapunov(np.copy(A),np.copy(Q))
    except ValueError:
        return np.full_like(Q,np.inf)


def dare(A,B,Q,R):
    """
    Solve the discrete-time algebraic Riccati equation.
    Wrapper around scipy.linalg.solve_discrete_are.
    Pass a copy of input matrices to protect them from modification.
    """
    return solve_discrete_are(np.copy(A),np.copy(B),np.copy(Q),np.copy(R))


def dare_gain(A,B,Q,R):
    """
    Solve the discrete-time algebraic Riccati equation.
    Return the optimal cost-to-go matrix P and associated gains K
    such that u = Kx is the optimal control.
    """
    P = dare(A,B,Q,R)
    K = -la.solve((R + mdot(B.T, P, B)),  mdot(B.T, P, A))
    return P, K