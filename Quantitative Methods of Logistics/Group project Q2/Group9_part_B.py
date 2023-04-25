# QML Q2 Assignment v1.2: Subtour and time window Constraint combined 

from gurobipy import *
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt

model = Model('TSPproblem')

# import data
file_path = "data_small.txt"
colnames = ['LOC_ID', 'XCOORD', 'YCOORD', 'DEMAND', 'READYTIME', 'DUETIME', 'SERVICETIME']
data = pd.read_csv(file_path, names=colnames, delimiter='\t')
M = 5000

# ---------------------- parameters --------------------- 
# given information
loc_id = list(data['LOC_ID'])
xcoord = list(data['XCOORD'])
ycoord = list(data['YCOORD'])
demand = list(data['DEMAND'])
readytime = list(data['READYTIME'])
duetime = list(data['DUETIME'])
servetime = list(data['SERVICETIME'])

# sets of nodes
N = len(loc_id)
print(N)

# matrix of distance from i to j
distance = []
for i in range(N):
    d_i = []  
    d = 0 
    for j in range(N):
        d = math.sqrt((xcoord[i]-xcoord[j])**2 + (ycoord[i]-ycoord[j])**2)
        d_i.append(d)
    distance.append(d_i)

# ----------------------- Variables ----------------------
# whether path ij in the solution
x = {} 
for i in range(N):
    for j in range(N):
        x[i,j] = model.addVar(vtype=GRB.BINARY, name = 'X[' + str(i) + ',' + str(j) + ']')

# time at which the vehicle arrives at node i
t = {}
for i in range(N):
    t[i] = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'T['+ str(i) + ']')

# load of the vehicle at node i
l = {}
for i in range(N):
    l[i] = model.addVar(lb = 0, vtype = GRB.INTEGER, name = 'L[' + str(i) + ']')


# ----------------------- Objective function ----------------------
model.setObjective(quicksum(x[i,j] * distance[i][j] for i in range(N) for j in range(N)))

model.update()
model.modelSense = GRB.MINIMIZE
model.update ()

# ----------------------- Constraints ----------------------
# each node is only entered once
con1 = {}
for j in range(N):
    list_1=[]
    for i in range(N):
        if i==j:
            con1[j] = model.addConstr(x[i,j] == 0)
            continue
        else:
            list_1.append(i)    
    con1[j] = model.addConstr(quicksum(x[i,j] for i in list_1) == 1, 'con1[' + str(j) + ']')

# each node is only left once 
con2 = {}
for i in range(N):
    list_2=[]
    for j in range(N):
        if i==j:
            con2[i] = model.addConstr(x[i,j]  == 0)
            continue
        else:
            list_2.append(j)
    con2[i] = model.addConstr(quicksum(x[i,j] for j in list_2) == 1, 'con2[' + str(i) + ']')

# subtour constrain, j is immediately visited after i
con3 = {}
for i in range(N):
    for j in range (N):
        if i == j:
            continue
        elif i == 0:
            con3[i,j] = model.addConstr((servetime[i] + distance[i][j]) <= t[j], 'con3[' + str(i) + str(j) + ']')
        else:
            con3[i,j] = model.addConstr(t[i]+ servetime[i] + distance[i][j] <= t[j]+ M*(1-x[i,j]), 'con3[' + str(i) + str(j) + ']')

# time window constrain
con4 = {}
for i in range(N):  
    con4[i] = model.addConstr(t[i] >= readytime[i], 'con4[' + str(i) + ']')

# time window constrain 
con5 = {}
for i in range(N):
    con5[i] = model.addConstr(t[i] <= duetime[i], 'con5[' + str(i) + ']')

# load continuaty constrain 1
con6 = {}
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        elif i == 0:
            con6[i,j] = model.addConstr(l[i] == 0, 'con6[' + str(i) + str(j) + ']') 
        else: 
            con6[i,j] = model.addConstr(l[i] - demand[i] - M*(1-x[i,j]) <= l[j], 'con6[' + str(i) + str(j) + ']')

# load continuaty constrain 
con7 = {} 
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        elif i == 0:
            con7[i,j] = model.addConstr(l[i] == 0, 'con7[' + str(i) + str(j) + ']') 
        else: 
            con7[i,j] = model.addConstr(l[j] <= l[i] - demand[i] + M*(1-x[i,j]), 'con7[' + str(i) + str(j) + ']')

# ensures that the vehicle has enough capacity to satisfy demand
con8 = {}
for i in range(N):
    con8[i] = model.addConstr(demand[i] <= l[i], 'con8[' + str(i) + ']')


# ----------------------- Set Model Parameters ----------------------
# Set time constraint for optimization (5minutes)
model.setParam('TimeLimit', 30 * 60)

model.setParam('OutputFlag', True)  # silencing gurobi output or not
model.setParam ('MIPGap', 0);       # find the optimal solution
model.write('output.lp')            # print the model in .lp format file

model.optimize ()

# ----------------------- Print Results ----------------------
print ('\n--------------------------------------------------------------------\n')
status = model.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    # Print optimal solution
    print ('Minimum distance : %10.2f' % model.objVal)
    print ('')

    print ("time to node:")
    d={}
    for i in range(N):
        print ("node ",i,":  ",t[i].x)
        d[i]=t[i].x
    print(d)
    
    sequence = sorted(d.items(),  key = lambda kv: kv[1])
    print("Sequence       Time of Visit")
    for i in range(N):
        s = '%6s   ' %sequence[i][0]+'%12d   ' %sequence[i][1]
        print (s)
    
    print("Sequence       Load of vehicle when arriving at location")
    for i in sequence:
        s = '%6s   ' %i[0]+'%12d   ' %l[i[0]].x
        print(s)

    # Plotting
    plt.title("Route")
    plt.xlabel("x")
    plt.ylabel("y")
    
    x = [data.loc[i[0],'XCOORD'] for i in sequence]
    x.insert(0,40)
    y = [data.loc[i[0],'YCOORD'] for i in sequence]
    y.insert(0,50)
    seq = []
    for i in sequence:
        seq.append(i[0])
    seq.insert(0,0)

    plt.plot(x, y,color='r')
    for i in range(len(x)):
        plt.text(x[i], y[i], seq[i]) 
    plt.show()


elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)