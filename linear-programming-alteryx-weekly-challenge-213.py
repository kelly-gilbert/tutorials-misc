# -*- coding: utf-8 -*-
"""
linear-programming-alteryx-weekly-challenge-213.py

Use PuLP to solve Alteryx Weekly Challenge #213
https://community.alteryx.com/t5/Weekly-Challenge/Challenge-213-Optimized-Flower-Arrangements/td-p/561687

Author: Kelly Gilbert
Created: 2020-06-27
Requirements: 
    - pulp 2.1+
"""

flower_costs = { 'red_rose' : 2.30,
                 'white_daisy' : 2.25,
                 'white_calla_lily' : 2.53,
                 'red_gerbera_daisy' : 2.45,
                 'red_carnation' : 2.17,
                 'white_carnation' : 2.15, 
                 'spider_mum' : 2.50,
                 'filler_greens' : 2.10 
               }

# Define the model
model = LpProblem(name="flower-arrangements", sense=LpMaximize)

# Define the decision variables
# this creates a dictionary of LpVariables with the same keys as flower_costs
flower_vars = {}
for k in flowers.keys():
    if k == 'filler_greens': 
        lb = 30
        ub = 40
    else:
        lb = 40
        ub = 80

    flower_vars[k] = LpVariable(name=k, lowBound=lb, upBound=ub, cat='Integer')

# Add constraints
model += lpSum(flower_costs[i] * flower_vars[i] for i in flower_vars.keys()) <= 950, \
               'Total cost constraint'

# Set the objective
model += lpSum(flower_vars[i] for i in flower_vars.keys()), \
               'Total number of flowers'
    
# Solve the optimization problem
status = model.solve()

# Get the results
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

for var in flower_vars.values():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")

model.solver
