import numpy as np
import cvxpy as cp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Define the objective function
def objective(p):
    return cp.sum([n * X[n-1] * p[n-1] for n in range(1, N+1)])

def check_constraints(X, p, R, N):
    # Check the first set of constraints
    # Check the second set of constraints
    for y in range(2, N+1):
        first_sum = sum([(X[i] - y * X[i-1+y]) * p[i] for i in range(0,N-y+1)])
        second_sum = sum([X[i] *p[i] for i in range(N-y+1,N)])
        check = first_sum + second_sum
        if check < tol :
            print(y,check,[(X[i-1] - y * X[i-1+y])*p[i] for i in range(0,N-y+1)])
            return False
    return True

# Constants
N = 5
R = 1
tol = -1e-10
# Variables
X = cp.Variable(N)



p_values = [[0,1,0,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,1/2,1/2],[0,1/2,0,0,1/2],[0,1/2,0,1/2,0],[1/5]*5]
# p_values = [[0,0,0,1/100,1-1/100]]

for p in p_values:
    # Constraints
    constraints = [0 <= X, X <= R/(1+np.arange(0,N))]
    for y in range(2, N+1):
        first_sum = sum([(X[i] - y * X[i-1+y]) * p[i] for i in range(0,N-y+1)])
        second_sum = sum([X[i] *p[i] for i in range(N-y+1,N)])
        constraints.append( first_sum + second_sum>= 0)

    prob = cp.Problem(cp.Maximize(objective(p)), constraints)
    prob.solve(solver=cp.GLPK)
    print(prob.value,X.value)
    # x = [0, 0, 0, 0, 1/5]
    if check_constraints(X.value, p, R, N):
        print("Solution satisfies all constraints.")
    else:
        print("Solution does not satisfy all constraints.")

