#cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.math cimport exp 
import numpy as np

cdef extern from "digamma_c.c":
    double digamma(double x)

def digamma_cython_np(double[:] X):

    cdef int N = X.shape[0]
    cdef int i
    cdef double[:] Y = np.zeros(N)

    for i in range(N):
        Y[i] = digamma(X[i])

    return np.asarray(Y)

def digamma_cython_d(double x):
    return digamma(x)