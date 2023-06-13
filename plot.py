import numpy as np
import cvxpy as cp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from scipy.interpolate import griddata


def min_distance(point,data_points):
    distances = [sum([(x[i]-point[i])**2 for i in range(0,len(point))]) for x in data_points]
    return distances.index(min(distances))
def interpolate(data_points,results):
    n = 10**2
    set_points = [[0,0,1-i/n-j/n,i/n,j/n] for i in range(0,n) for j in range(0,n+1-i)]
    new_results = [results[min_distance(element,data_points)] for element in set_points]
    return set_points, new_results
# Constants
N = 5
R = 1
num_samples = 10**3 # Increase this number to get more points

# Variables
X = cp.Variable(N)

# Define the objective function
def objective(p):
    return cp.sum([n*X[n-1]*p[n-1] for n in range(1, N+1)])

# Colors for the plot
cmap = plt.get_cmap("jet")  # Change colormap to 'jet'

# Generate points (p vectors) in the simplex
def generate_points(size):
    points = [[0,0,random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for i in range(size)]
    points = [np.array(element)/sum(element) for element in points]

    return points

p_vectors = generate_points(num_samples)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

p_vectors = p_vectors + [[0,0,0,1,0],[0,0,0,0,1],[0,0,1,0,0]]
# p_vectors = [[0,0,0,1-i/10000,i/10000] for i in range(9000,10000+1)]
min_value =1
results = []
cont = 0

for p in p_vectors:
       # Constraints
    constraints = [0 <= X, X <= R/(1+np.arange(0,N))]
    for y in range(2, N+1):
        first_sum = sum([(X[i] - y * X[i-1+y]) * p[i] for i in range(0,N-y+1)])
        second_sum = sum([X[i] *p[i] for i in range(N-y+1,N)])
        constraints.append( first_sum + second_sum>= 0)

    prob = cp.Problem(cp.Maximize(objective(p)), constraints)
    prob.solve(solver=cp.GLPK,max_iters=1000)
    results.append(prob.value)
    if prob.value < min_value:
        min_value = prob.value
        prob_min = p
    if cont %100==0:
        print(f"lef {len(p_vectors)-cont}")
    cont +=1 


p_vectors, results = interpolate(p_vectors,results)
_min = min(results)
_max = max(results)
results = [(element - _min)/(_max-_min) for element in results]
# normalize results for color mapping
results = np.array(results)
# Scatter plot object

p_1 = [element[2] for element in p_vectors]
p_3 = [element[3] for element in p_vectors]
p_4 = [element[4] for element in p_vectors]


scat = ax.scatter(p_1, p_3, p_4, c=results, cmap=cmap, vmin=_min, vmax=1)


# Add colorbar
fig.colorbar(scat, shrink=0.5, aspect=5)

#plt.savefig('ke_plot.png')
plt.show()
