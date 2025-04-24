import numpy as np
from scipy.optimize import minimize

# Define the objective function (screw)
def screw(x):
    return (np.pi / 4) * (0.405 * x[0]**2 + x[1]**2 * x[3] + x[2]**2 * x[4])

# Define the inequality constraints
def gi(x):
    return np.array([
        -(38.88 + 96 * x[3] + 96 * x[4] - np.pi * 20000 * x[1]**3),
        -(6 * (x[6] / x[5]) - np.pi * 22000 * x[0]**3),
        -(8.345 - x[3] - x[4]),
        -( -9.595 + x[3] + x[4]),
        -(x[1] - 0.625),
        -(5 * (x[5] / x[6]) - 0.0583 * x[7]),
        -((1.5 * x[1] * ((np.pi * 0.35 * x[1] + 0.5 / x[7]) / (0.5 * np.pi * x[1] - 0.35 / x[7]))) - 0.125 * (x[6] / x[5])),
        -(x[2] - x[1]),
        -(x[0] - x[1])
    ])

# Define equality constraint
def h(x):
    return -x[2] + x[1] - 1.299 / x[7]

# Bounds for the variables
LB = [0, 0, 0.1875, 7.023, 1.1525, 8, 0, 0]
UB = [np.inf, 0.625, np.inf, 7.523, 1.6525, np.inf, 52, 24]

# Initial guess
x0 = [0.2, 0.2, 0.2, 7.3, 1.4, 10, 30, 22]

# Define bounds as tuples (lower, upper)
bounds = [(LB[i], UB[i]) for i in range(len(LB))]

# Define constraints (both inequality and equality)
constraints = [{'type': 'ineq', 'fun': gi},
               {'type': 'eq', 'fun': h}]

# Optimization
# result = minimize(screw, x0, bounds=bounds, constraints=constraints, options={'maxiter': 200, 'disp': True})
result = minimize(screw, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'disp': True})


# Results
print(f"Optimal solution: {result.x}")
print(f"Objective function value at optimal solution: {result.fun}")
# print(f"Exit status: {result.message}")
# print(f"Number of iterations: {result.nit}")
