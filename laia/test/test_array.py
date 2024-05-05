import numpy as np
import sys
sys.path.append("../build/test")
import laia_test

input = np.random.rand(1000, 10)

laia_test.recv_py_array(input)

results = laia_test.get_dist()
print(results.shape)
print(results)

# this should be error: queue is empty
results = laia_test.get_dist()
print(type(results))
print(results)