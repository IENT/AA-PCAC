import numpy as np


def fprintf(s,k):
    print(s.format(k))

def floor(s):
    return np.floor(s)

def bitshift(A, k):
    #If k is positive, MATLAB® shifts the bits to the left and inserts k 0-bits on the right.
    #If k is negative and A is nonnegative, then MATLAB shifts the bits to the right and inserts |k| 0-bits on the left.
    k = int(k)
    A = np.int64(A)
    if k >= 0:
        return A << k
    else:
        return A >> -k
    
def bitor(A,k):
    A = np.uint64(A)
    k = np.uint64(k)
    return np.bitwise_or(A, k)

def bitand(A,k):
    A = np.uint64(A)
    k = np.uint64(k)
    return np.bitwise_and(A,k)

def bitget(number, pos):
    return (number >> int(pos-1)) & 1 
    
def double(A):
    return np.double(A)

def uint64(A):
    return np.uint64(A)