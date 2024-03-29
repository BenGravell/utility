"""General matrix math functions."""
# Author: Ben Gravell

import numpy as np
from numpy import linalg as la
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are
from functools import reduce
from copy import copy

from .extramath import quadratic_formula


def vec(A):
    """Return the vectorized matrix A by stacking its columns."""
    return A.reshape(-1, order="F")


def mat(v, shape=None):
    """Return matricization i.e. the inverse operation of vec of vector v."""
    if shape is None:
        dim = int(np.sqrt(v.size))
        shape = dim, dim
    matrix = v.reshape(shape[1], shape[0]).T
    return matrix


def svec(A):
    """Return the symmetric vectorization i.e. the vectorization of the upper triangular part of matrix A."""
    return A[np.triu_indices(A.shape[0])]


def svec2(A):
    """
    Return the symmetric vectorization i.e. the vectorization of the upper triangular part of matrix A
    with off-diagonal entries multiplied by sqrt(2) so that la.norm(A, ord='fro')**2 == np.dot(svec2(A), svec2(A))
    """
    B = A + np.triu(A, 1)*(2**0.5 - 1)
    return B[np.triu_indices(A.shape[0])]


def smat(v):
    """Return the symmetric matricization i.e. the inverse operation of svec of vector v."""
    m = v.size
    n = int(((1 + m*8)**0.5 - 1)/2)
    idx_upper = np.triu_indices(n)
    idx_lower = np.tril_indices(n, -1)
    A = np.zeros([n, n])
    A[idx_upper] = v
    A[idx_lower] = A.T[idx_lower]
    return A


def smat2(v):
    """Return the symmetric matricization i.e. the inverse operation of svec2 of vector v."""
    m = v.size
    n = int(((1 + m*8)**0.5 - 1)/2)
    idx_upper = np.triu_indices(n)
    idx_lower = np.tril_indices(n, -1)
    A = np.zeros([n, n])
    A[idx_upper] = v
    A[np.triu_indices(n, 1)] /= 2**0.5
    A[idx_lower] = A.T[idx_lower]
    return A


def sympart(A):
    """Return the symmetric part of matrix A."""
    return 0.5*(A + A.T)


def is_pos_def(A):
    """Check if matrix A is positive definite."""
    try:
        la.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def succ(A, B):
    """Check the positive definite partial ordering of A > B."""
    return is_pos_def(A - B)


def psdpart(X):
    """Return the positive semidefinite part of a symmetric matrix."""
    X = sympart(X)
    Y = np.zeros_like(X)
    eigvals, eigvecs = la.eig(X)
    for i in range(X.shape[0]):
        if eigvals[i] > 0:
            Y += eigvals[i]*np.outer(eigvecs[:, i], eigvecs[:, i])
    Y = sympart(Y)
    return Y


def kron(*args):
    """Overload and extend the numpy kron function to take a single argument."""
    if len(args) == 1:
        return np.kron(args[0], args[0])
    else:
        return np.kron(*args)


def mdot(*args):
    """Multiple dot product, thin convenience wrapper around numpy.linalg.multi_dot()."""
    return np.linalg.multi_dot(args)


def mip(A, B):
    """Matrix inner product of A and B."""
    return np.tensordot(A, B).item()


def matmul_lr(A, P):
    """Matrix multiplication of P on the left by A and on the right by A.T."""
    return mdot(A, P, A.T)


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


def solveb(a, b):
    """
    Solve a = bx.
    Similar to MATLAB / operator for square invertible matrices.
    """
    return la.solve(b.T, a.T).T


def lstsqb(a, b):
    """
    Return least-squares solution to a = bx.
    Similar to MATLAB / operator for rectangular matrices.
    If b is invertible then the solution is la.solve(a, b).T
    """
    return la.lstsq(b.T, a.T, rcond=None)[0].T


def dlyap(A, Q):
    """
    Solve the discrete-time Lyapunov equation.
    Wrapper around scipy.linalg.solve_discrete_lyapunov.
    Pass a copy of input matrices to protect them from modification.
    """
    try:
        return solve_discrete_lyapunov(np.copy(A), np.copy(Q))
    except ValueError:
        return np.full_like(Q, np.inf)


def dare_scalar(A, B, Q, R, E=None, S=None):
    """
    Solve the discrete-time algebraic Riccati equation for the scalar case of
    a single state and a single input.
    In this case the equation is a scalar quadratic equation.
    """
    A, B, Q, R = [float(var) for var in [A, B, Q, R]]
    if E is None:
        E = 1
    if S is None:
        S = 0

    A2 = A**2
    B2 = B**2
    E2 = E**2
    S2 = S**2

    aa = -B2*E2
    bb = R*(A2 - E2) + Q*B2 + 2*A*B*S
    cc = Q*R - S2

    roots = np.array(quadratic_formula(aa, bb, cc))

    if not (roots[0] > 0 or roots[1] > 0):
        P = None
    else:
        P = roots[roots > 0][0]*np.eye(1)
    return P


def dare_gain_scalar(A, B, Q, R, E=None, S=None):
    P = dare_scalar(A, B, Q, R, E, S)
    Huu = R + B*B*P
    Hux = B*P*A
    if S is not None:
        Hux += S
    K = - Huu/Hux
    return P, K


def dare(A, B, Q, R, E=None, S=None):
    """
    Solve the discrete-time algebraic Riccati equation.
    Wrapper around scipy.linalg.solve_discrete_are.
    Pass a copy of input matrices to protect them from modification.
    """
    # Handle the scalar case more efficiently by solving the ARE exactly
    if A.shape[1] == 1 and B.shape[1] == 1:
        return dare_scalar(A, B, Q, R, E, S)*np.eye(1)
    else:
        return solve_discrete_are(copy(A), copy(B), copy(Q), copy(R), copy(E), copy(S))


def dare_gain(A, B, Q, R, E=None, S=None):
    """
    Solve the discrete-time algebraic Riccati equation.
    Return the optimal cost-to-go matrix P and associated gains K
    such that u = Kx is the optimal control.
    """
    # Handle the scalar case more efficiently by solving the ARE exactly
    if A.shape[1] == 1 and B.shape[1] == 1:
        P, K = dare_gain_scalar(A, B, Q, R, E, S)
        return P*np.eye(1), K*np.eye(1)
    else:
        P = dare(A, B, Q, R, E, S)
        Hux = mdot(B.T, P, A)
        Huu = R + mdot(B.T, P, B)
        if S is not None:
            Hux += S.T
        K = -la.solve(Huu, Hux)
        return P, K
