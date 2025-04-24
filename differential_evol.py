import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint

# Objective function
def fun(x):
    return (np.pi / 4) * (0.405 * x[0]**2 + x[1]**2 * x[3] + x[2]**2 * x[4])

# Nonlinear constraints
def cons(x):
    c = [
        38.88 + 96 * x[3] + 96 * x[4] - np.pi * 20000 * x[1]**3,
        6 * (x[6] / x[5]) - np.pi * 22000 * x[0]**3,
        8.345 - x[3] - x[4],
        -9.595 + x[3] + x[4],
        x[1] - 0.625,
        5 * (x[5] / x[6]) - 0.0583 * x[7],
        (1.5 * x[1] * ((np.pi * 0.35 * x[1] + 0.5 / x[7]) / (0.5 * np.pi * x[1] - 0.35 / x[7]))) - 0.125 * (x[6] / x[5]),
        x[2] - x[1],
        x[0] - x[1],
        x[2] - x[1] + 1.299 / x[7]
    ]
    return np.array(c)

bounds = [
    (0, 1e6),        # x1
    (0, 0.625),      # x2
    (0.1875, 1e6),   # x3
    (7.023, 7.523),  # x4
    (1.1525, 1.6525),# x5
    (8, 1e6),        # x6
    (0, 52),         # x7
    (0, 24)          # x8
]

#all constraints ≤ 0
nonlinear_constraint = NonlinearConstraint(cons, -np.inf, 0)

result = differential_evolution(
    fun,
    bounds,
    constraints=(nonlinear_constraint,),
    strategy='best1bin',
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42
)

print("Optimal x:", result.x)
print("Objective function value:", result.fun)
