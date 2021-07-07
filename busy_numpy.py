#!/usr/bin/env python3
# Run
# ./busy_numpy.py 8000 3

import sys
from time import time, localtime, strftime
from numpy import eye
from numpy.random import rand
from numpy.linalg import norm

# Keep CPU-RAM busy, that's it.
def busy_numpy(n = 4000, k_max = 2**31-2):
    gflo = n ** 3 * 2 / 1e9
    v = rand(n,1);  v = v/norm(v)
    u = rand(n,1);  u = u/norm(u)
    # a (not very) random (but fast generated) orthogonal matrix
    a = eye(n) - 2 * u @ u.T - 2 * v @ v.T + 4 * u * (u.T @ v) @ v.T
    c = a;
    for k in range(1, k_max+1):
        t0 = time()
        c = c @ a
        t = time() - t0
        s = strftime("%Y-%m-%d %H:%M:%S %Z", localtime())
        print('%s, t=%.3f, #%d, GFLOPS=%5.1f.' % (s, t, k, gflo/t))

if __name__ == '__main__':
    param = [int(i) for i in sys.argv[1:]]
    busy_numpy(*param)
