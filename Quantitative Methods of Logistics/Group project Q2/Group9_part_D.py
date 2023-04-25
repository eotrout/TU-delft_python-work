# QML Q2 Assignment v1.2: Subtour and time window Constraint combined 


from gurobipy import *
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt

model = Model('CVRP')

# import data
file_path = "data_small.txt"
colnames = ['LOC_ID', 'XCOORD', 'YCOORD', 'DEMAND', 'READYTIME', 'DUETIME', 'SERVICETIME']
data = pd.read_csv(file_path, names=colnames, delimiter='\t')
M = 5000
b=[120] #capacities of vehicles used - CASES: [60,60], [20,20,20,20,20,20]
K=len(b)

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
        for v in range(K):
            x[i,j,v] = model.addVar(vtype=GRB.BINARY, name = 'X[' + str(i) + ',' + str(j) + ',' + str(v) + ']')

# time at which the vehicle arrives at node i
t = {}
for i in range(N):
    for v in range(K):
        t[i,v] = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'T['+ str(i) + ','+ str(v) + ']')

# load of the vehicle at node i
l = {}
for i in range(N):
    for v in range(K):
        l[i,v] = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'L[' + str(i) + ',' + str(v) + ']')

z={}
for i in range(N):
    for v in range(K):
        z[i,v] = model.addVar(vtype=GRB.BINARY, name = 'Z[' + str(i) + ',' + str(j) + ']')


# ----------------------- Objective function ----------------------
model.setObjective(quicksum(x[i,j,v] * distance[i][j] for i in range(N) for j in range(N) for v in range(K)))

model.update()
model.modelSense = GRB.MINIMIZE
model.update ()


# ----------------------- Constraints ----------------------

con1 = {}
for i in range(N):
    if i==0:
        con1[i] = model.addConstr(quicksum(z[i,v] for v in range(K)) <= K, 'con1[' + str(i) + ']')
    else:
        con1[i] = model.addConstr(quicksum(z[i,v] for v in range(K)) == 1, 'con1[' + str(i) + ']')

con2 = {}
for v in range(K):
    con2[v] = model.addConstr(quicksum(demand[i]*z[i,v] for i in range(N)) <= b[v], 'con2[' + str(v) + ']')

# each node is only entered once
con3 = {}
for j in range(N):
    for v in range(K):
        con3[j,v] = model.addConstr(quicksum(x[i,j,v] for i in range(N)) == z[j,v], 'con3[' + str(j) + ']')

# each node is only left once 
con4 = {}
for i in range(N):
    for v in range(K):
        con4[i,v] = model.addConstr(quicksum(x[i,j,v] for j in range(N)) == z[i,v], 'con4[' + str(i) + ']')

con5 = {}
for i in range(N):
    for j in range(N):
        for v in range(K):
            if i==j:
                con5[i,j,v] = model.addConstr(x[i,j,v] == 0, 'con5[' + str(i) + ',' + str(j) + ',' + str(v) + ']')

# subtour constrain, j is immediately visited after i
con6 = {}
for i in range(N):
    for j in range (N):
        for v in range(K):
            if i == j:
                continue
            elif i == 0:
                con6[i,j,v] = model.addConstr((servetime[i] + distance[i][j]) <= t[j,v] + M*(1-x[i,j,v]), 'con6[' + str(i) + ',' + str(j) + ',' + str(v) + ']')
            else:
                con6[i,j,v] = model.addConstr(t[i,v]+ servetime[i] + distance[i][j] <= t[j,v] + M*(1-x[i,j,v]), 'con6[' + str(i) + ',' + str(j) + ',' + str(v) + ']')

# time window constrain
con7 = {}
for i in range(N):  
    for v in range(K):
        con7[i,v] = model.addConstr(t[i,v] >= readytime[i], 'con7[' + str(i) + ',' + str(v) + ']')

# time window constrain 
con8 = {}
for i in range(N):
    for v in range(K):
        con8[i,v] = model.addConstr(t[i,v] <= duetime[i], 'con8[' + str(i) + ',' + str(v) + ']')

# load continuaty constrain 1
con9 = {}
for i in range(N):
    for j in range(N):
        for v in range(K):
            if i == j:
                continue
            elif i == 0:
                con9[i,j,v] = model.addConstr(l[i,v] == 0, 'con9[' + str(i) + ',' + str(j) + ',' + str(v) +']') 
            else: 
                con9[i,j,v] = model.addConstr(l[i,v] - demand[i] - M*(1-x[i,j,v]) <= l[j,v], 'con9[' + str(i) + str(j) +  str(v) +']')

# load continuaty constrain 
con10 = {} 
for i in range(N):
    for j in range(N):
        for v in range(K):
            if i == j:
                continue
            elif i == 0:
                con10[i,j,v] = model.addConstr(l[i,v] == 0, 'con10[' + str(i) + str(j) + str(v) +']') 
            else: 
                con10[i,j,v] = model.addConstr(l[j,v] <= l[i,v] - demand[i] + M*(1-x[i,j,v]), 'con10[' + str(i) + str(j) +  str(v) +']')

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
    
    s="      "
    for v in range(K):
        s = s + " vehicle %d  " %v
    print (s)

    s=""
    for i in range(N):
        s="node "+str(i)
        for v in range(K):
            lol=t[i,v].x*z[i,v].x
            s = s + '%9.0f' %lol         
        print (s)
    
    complete_data={}
    for v in range(K):
        vehicle_data={}
        for i in range(N):
            if z[i,v].x==0:
                continue
            else:
                vehicle_data[i] = t[i,v].x * z[i,v].x
        complete_data[v]=vehicle_data

    complete_data_ordered={}
    for v in range(K):
        sequence = sorted(complete_data[v].items(),  key = lambda kv: kv[1])
        complete_data_ordered[v]=sequence

    # sequence of travel
    print("Sequence of travel")
    for v in complete_data_ordered:
        s="vehicle "+str(v+1)+":"
        for i in complete_data_ordered[v]:
            s=s+" "+str(i[0])
        print(s)
    
    print("Node       Load of vehicle when arriving at location")
    for v in complete_data_ordered:
        print(v)
        for i in complete_data_ordered[v]:
            s = '%6d   ' %i[0]+'%12d   ' %l[i[0],v].x
            print(s)
    
    plt.title("Route")
    plt.xlabel("x")
    plt.ylabel("y")

    for v in complete_data_ordered:
        x = [xcoord[j[0]] for j in complete_data_ordered[v]]
        x.insert(0,40)
        y = [ycoord[j[0]] for j in complete_data_ordered[v]]
        y.insert(0,50)
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(x, y,color=col)
        for i in complete_data_ordered[v]:
            plt.text(xcoord[i[0]], ycoord[i[0]], i[0]) 
    plt.show()
elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)