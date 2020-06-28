# -*- coding: utf-8 -*-
"""
linear-programming-tutorial.py

Real Python tutorial: Hands-On Linear Programming: Optimization With Python
https://realpython.com/linear-programming-python/

Author: Kelly Gilbert
Created: 2020-06-27
Requirements: 
    - scipy 1.4
    - pulp 2.1
"""


#-----------------------------------------------------------------------------
# scipy.optimize
# 
# does not allow >= inequality constraints
# solves only minimization problems
# 
# other constraints of scipy.optimize:
# - SciPy can’t run various external solvers.
# - SciPy can’t work with integer decision variables.
# - SciPy doesn’t provide classes or functions that facilitate model building. 
#   You have to define arrays and matrices, which might be a tedious and 
#   error-prone task for large problems.
# - SciPy doesn’t allow you to define maximization problems directly. 
#   You must convert them to minimization problems.
# - SciPy doesn’t allow you to define constraints using the
#   greater-than-or-equal-to sign directly. You must use the 
#   less-than-or-equal-to instead.


#-----------------------------------------------------------------------------

from scipy.optimize import linprog

# Problem #1

# minimize -z = -x - 2y
# st        2x + y <= 20 (red)
#          -4x + 5y <= 10 (blue)
#           x - 2y <= 2 (yellow)
#          -x + 5y = 15 (green)
#           x >= 0
#           y >= 0

obj = [-1, -2]
#      ─┬  ─┬
#       │   └┤ Coefficient for y
#       └────┤ Coefficient for x

lhs_ineq = [[ 2,  1],  # Red constraint left side
            [-4,  5],  # Blue constraint left side
            [ 1, -2]]  # Yellow constraint left side

rhs_ineq = [20,  # Red constraint right side
            10,  # Blue constraint right side
             2]  # Yellow constraint right side

lhs_eq = [[-1, 5]]  # Green constraint left side
rhs_eq = [15]       # Green constraint right side

# note, linprog takes the bounds zero to positive infinity by default
# can also use math.inf, numpy.inf, or scipy.inf
bnd = [(0, float("inf")),  # Bounds of x
       (0, float("inf"))]  # Bounds of y


# solve the problem of interest
# c = coefficients of the objective function
# a_ub and b_ub are the left and right sides of the inequality constraints
# a_eq and b_eq are the left and right sides of the equality constraints
# bounds = upper and lower bounds for the decision variables
# method: 
#   'interior-point' (default)
#   'revised simplex' (two-phase simplex method)
#   'simpex' (legacy two-phase simplex method)

opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
              A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
              method="revised simplex")
opt


# results:
#      con: array([0.])
#      fun: -16.818181818181817
#  message: 'Optimization terminated successfully.'
#      nit: 3
#    slack: array([ 0.        , 18.18181818,  3.36363636])
#   status: 0
#  success: True
#        x: array([7.72727273, 4.54545455])

# .con = equality constraints residuals
# .fun = objective function value at optimum
# .message = status of the solution
# .nit = number of iterations
# .slack = slack variables (diff between values of lh and rh of the constraints)
# .status = status of the solution (0 = optimal solution was found)
# .success = boolean that shows whether the optimal solution has been found
# .x = a numpy array holding the optimal values of the decision variables 
# access the values separately with opt.x (e.g. opt.con)
# 


# Problem #2: resource allocation

# max 20x1 + 12x2 + 40x3 + 25x4
# st  x1 + x2 + x3 + x4 <= 50
#     3x1 + 2x2 + x3 <= 100 
#     x2 + 2x3 + 3x4 <= 90 
#     x1, x2, x3, x4 >= 0

obj2 = [-20, -12, -40, -25]

lhs_ineq2 = [[1, 1, 1, 1],  # manpower
            [3, 2, 1, 0],  # material a
            [0, 1, 2, 3]]  # material b

rhs_ineq2 = [50,   # manpower
            100,  # material a
             90]   # material b

# no equality constraints

# note, linprog takes the bounds zero to positive infinity by default
# can also use math.inf, numpy.inf, or scipy.inf
bnd2 = [(0, float("inf")),  # Bounds of x1
       (0, float("inf")),
       (0, float("inf")),
       (0, float("inf"))]  # Bounds of x4

opt2 = linprog(c=obj2, A_ub=lhs_ineq2, b_ub=rhs_ineq2,
              bounds=bnd2,
              method="revised simplex")
opt2



#-----------------------------------------------------------------------------
# PuLP
# 
# 
#-----------------------------------------------------------------------------

from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable


# Problem #1

# minimize z = x + 2y
# st        2x + y <= 20 (red)
#          -4x + 5y <= 10 (blue)
#           -x + 2y >= -2 (yellow)
#          -x + 5y = 15 (green)
#           x >= 0
#           y >= 0

# create the model
model = LpProblem(name='small-problem', sense=LpMaximize)

# initialize decision variables
# default is negative infinity to positive infinity
# can set upper bound with upBound
# can specify category with cat='Continuous', 'Integer', 'Binary'
x = LpVariable(name='x', lowBound=0)
y = LpVariable(name='y', lowBound=0)

# now that we have create the decision variables x and y, we can use them 
# to create other PuLP objects
expression = 2*x + 4*y
type(expression)

constraint = 2*x + 4*y >= 8
type(constraint)

# you can add or subtract variables or expressions and you can multiply them
# with constants. you can also combine linear expressions, variables, 
# and scalars with operators ==, <=, or >= to get instances of pulp.Lp.Constraint

# add the constraints to the model
model += (2*x + y <= 20, 'red_constraint')
model += (4*x - 5*y >= -10, 'blue_constraint')
model += (-x + 2*y >= -2, 'yellow_constraint')
model += (-x + 5*y == 15, 'green_constraint')

# set the objective function
obj_func = x + 2*y
model += obj_func
# equivalent syntax: model += x + 2*y
# equivalent: model += lpSum([x, 2*y]) -- useful for larger problems

# see the model definition
model

# solve the problem
# status will be 1 if the optimum is found
# other status codes: https://www.coin-or.org/PuLP/constants.html#pulp.constants.LpStatus
# note: calling .solve() changes the state of the decision variable objects (x and y)
status = model.solve()
status

# get the optimization results and attributes of the model
print(f"status: {model.status}, {LpStatus[model.status]}")

print(f"objective: {model.objective.value()}")

for var in model.variables():
    print(f"{var.name} : {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name} : {constraint.value()}")

# see which solver was used
model.solver

# CBC is the default
# can set a different one in .solve(solver=xxxx) for example solver=GLPK(msg=False)
#   keep in mind that you will need to install the related solver, 
#   e.g. from pulp import GLPK


# solving the same problem (#1) with GLPK...

from pulp import GLPK

# Create the model - this is all the same as above
model = LpProblem(name="small-problem", sense=LpMaximize)

x = LpVariable(name="x", lowBound=0)
y = LpVariable(name="y", lowBound=0)

model += (2 * x + y <= 20, "red_constraint")
model += (4 * x - 5 * y >= -10, "blue_constraint")
model += (-x + 2 * y >= -2, "yellow_constraint")
model += (-x + 5 * y == 15, "green_constraint")

model += lpSum([x, 2 * y])

# Solve the problem
status = model.solve(solver=GLPK(msg=False))
status

# output the results (this is all the same as above)
print(f"status: {model.status}, {LpStatus[model.status]}")

print(f"objective: {model.objective.value()}")

for var in model.variables():
    print(f"{var.name} : {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name} : {constraint.value()}")

model.solver


# solving problem #1 again as a mixed-integer problem

# Create the model
model = LpProblem(name="small-problem", sense=LpMaximize)

# Initialize the decision variables: x is integer, y is continuous
x = LpVariable(name="x", lowBound=0, cat="Integer")
y = LpVariable(name="y", lowBound=0)

# Add the constraints to the model
model += (2 * x + y <= 20, "red_constraint")
model += (4 * x - 5 * y >= -10, "blue_constraint")
model += (-x + 2 * y >= -2, "yellow_constraint")
model += (-x + 5 * y == 15, "green_constraint")

# Add the objective function to the model
model += lpSum([x, 2 * y])

# Solve the problem
status = model.solve()

# display the results
print(f"status: {model.status}, {LpStatus[model.status]}")

print(f"objective: {model.objective.value()}")

for var in model.variables():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")

model.solver



# Problem #2: resource allocation (this time with PuLP)

# max 20x1 + 12x2 + 40x3 + 25x4
# st  x1 + x2 + x3 + x4 <= 50
#     3x1 + 2x2 + x3 <= 100 
#     x2 + 2x3 + 3x4 <= 90 
#     x1, x2, x3, x4 >= 0

# Define the model
model = LpProblem(name="resource-allocation", sense=LpMaximize)

# Define the decision variables
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 5)}

# Add constraints
model += (lpSum(x.values()) <= 50, "manpower")
model += (3 * x[1] + 2 * x[2] + x[3] <= 100, "material_a")
model += (x[2] + 2 * x[3] + 3 * x[4] <= 90, "material_b")

# Set the objective
model += 20 * x[1] + 12 * x[2] + 40 * x[3] + 25 * x[4]

# Solve the optimization problem
status = model.solve()

# Get the results
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

for var in x.values():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")

model.solver


# making this example more complicated:
# say the factory can't produce the first and third products in parallel 
# due to a machinery issue 

# Define the model
model = LpProblem(name="resource-allocation", sense=LpMaximize)

# Define the decision variables
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 5)}
y = {i: LpVariable(name=f"y{i}", cat="Binary") for i in (1, 3)}
# y contains binary decision variables (x1 yes/no and x3 yes/no)


# Add constraints
model += (lpSum(x.values()) <= 50, "manpower")
model += (3 * x[1] + 2 * x[2] + x[3] <= 100, "material_a")
model += (x[2] + 2 * x[3] + 3 * x[4] <= 90, "material_b")

M = 100    # arbitrarily large number
model += (x[1] <= y[1]*M, 'x1_constraint')    # if y[1] is zero, then x[1] must
                                              # be zero, else it can be any + num
model += (x[3] <= y[3]*M, 'x3_constraint')    # if y[3] is zero, then x[3] must
                                              # be zero, else it can be any + num
model += (y[1] + y[3] <= 1, 'y_constraint')   # either y[1] or y[3] or both is zero

# Set the objective
model += 20 * x[1] + 12 * x[2] + 40 * x[3] + 25 * x[4]

# Solve the optimization problem
status = model.solve()

# Get the results
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

for var in x.values():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")

model.solver
