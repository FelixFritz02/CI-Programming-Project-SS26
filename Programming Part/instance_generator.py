# Generates an instance of DRAUSP using T_d time steps, K resources, and a list of capacities C_k for each resource.
# The parameter lam controls the average number of requests in the Poisson distribution for generating the request vector q. 

import random
import numpy as np

def instance_generator(T_d=10, K=5, lam=1):

    total_requests = []
    for _ in range(T_d):
        r = random.gauss(10, 3)  # normal distribution
        len_q = random.randint(1, K)  # uniform integer
        q = [np.random.poisson(lam) + 1 for _ in range(len_q)]
        q += [0] * (K - len_q)
        total_requests.append([r] + q)
        
    return total_requests