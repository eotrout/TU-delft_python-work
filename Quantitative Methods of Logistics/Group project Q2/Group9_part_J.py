from gurobipy import *
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt


model = Model('HSVRP')

# import data
file_path = "data_large.txt"
colnames = ['LOC_ID', 'XCOORD', 'YCOORD', 'DEMAND', 'READYTIME', 'DUETIME', 'SERVICETIME']
data = pd.read_csv(file_path, names=colnames, delimiter='\t')

# ---------------------- parameters --------------------- 
# given information
loc_id = list(data['LOC_ID'])
xcoord = list(data['XCOORD'])
ycoord = list(data['YCOORD'])
demand = list(data['DEMAND'])
readytime = list(data['READYTIME'])
duetime = list(data['DUETIME'])
servetime = list(data['SERVICETIME'])

M = 5000
N = len(loc_id)      # set of nodes
K = 25               # set of vehicles

capacity = [20]*10 + [100]*15       # list capacity of each vehicle
cost = [0 ]*10 + [4000]*15         # list of one-payment cost of each vehicle

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
for v in range(K):
    for i in range(N):
        for j in range(N):
            x[i,j,v] = model.addVar(vtype = GRB.BINARY, name = 'X[' + str(i) + ',' + str(j) + ',' + str(v) + ']')

# time at which the vehicle arrives at node i
t = {}
for v in range(K):
    for i in range(N):
        t[i,v] = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'T['+ str(i) + ',' + str(v) + ']')

# load of the vehicle at node i
l = {}
for v in range(K):
    for i in range(N):
        l[i,v] = model.addVar(lb = 0, vtype = GRB.INTEGER, name = 'L[' + str(i) + ',' + str(v) + ']')

# proportion of demand at i delievered by v
p = {}
for v in range(K):
    for i in range(N):
        p[i,v] = model.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name = 'P[' + str(i) + ',' + str(v) + ']')

# whether costumer i is visited by vehicle v
z = {}
for v in range(K):
    for i in range(N):
        z[i,v] = model.addVar(vtype = GRB.BINARY, name = 'Z[' + str(i) + ',' + str(v) + ']')

# Whether vehicle v is used in the solution
y = {}
for v in range(K):
    y[v] = model.addVar(vtype = GRB.BINARY, name = 'Y[' + str(v) + ']')


# ----------------------- Objective function ----------------------
model.setObjective(
    quicksum(x[i,j,v] * distance[i][j] * 0.2 for i in range(N) for j in range(N) for v in range(K)) + 
    quicksum(y[v] * cost[v] for v in range(K))
)

model.update()
model.modelSense = GRB.MINIMIZE
model.update ()


# ----------------------- Constraints ----------------------
# demand at each customer is satisfied
con1 = {}
for i in range(N):
    if i == 0:
        continue
    else:
        con1[i] = model.addConstr(quicksum(p[i,v] for v in range(K)) >= 1, 'con1[' + str(i) + ']')

# vehicle capacity constraints
con2 = {}
for v in range(K):
    con2[v] = model.addConstr(quicksum(demand[i]*p[i,v] for i in range(1,N)) <= capacity[v], 'con2[' + str(v) + ']')

# load continuaty constrain - left part
con3 = {}
for v in range(K):
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            elif i == 0:
                con3[i,j,v] = model.addConstr(
                    quicksum(l[i,v] for v in range(K)) == quicksum(demand[j] for j in range(N)), 
                    'con3[' + str(i) + ',' + str(j) + ',' + str(v) + ']'
                ) 
            else: 
                con3[i,j,v] = model.addConstr(
                    l[i,v] - demand[i]*p[i,v] - M*(1-x[i,j,v]) <= l[j,v], 
                    'con3[' + str(i) + ',' + str(j) + ',' + str(v)+ ']'
                )

# load continuaty constrain - right part
con4 = {} 
for v in range(K):
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            elif i == 0:
                con4[i,j,v] = model.addConstr(
                    quicksum(l[i,v] for v in range(K)) == quicksum(demand[j] for j in range(N)), 
                    'con4[' + str(i) + ',' + str(j) + ',' + str(v) + ']'
                ) 
            else: 
                con4[i,j,v] = model.addConstr(
                    l[j,v] <= l[i,v] - demand[i]*p[i,v] + M*(1-x[i,j,v]), 
                    'con4[' + str(i) + ',' + str(j) + ',' + str(v)+ ']'
                )

# flow conservation 
con5 = {}
for q in range(N):
    for v in range(K):
        con5[q,v] = model.addConstr(
            quicksum(x[i,q,v] for i in range(N)) - x[q,q,v] == quicksum(x[q,j,v] for j in range(N)) - x[q,q,v],
            'con5[' + str(i) + ',' + str(v) + ']'
        )


# subtour elimination
con6 = {}
for v in range(K):
    for i in range(N):
        for j in range (N):
            if i == j:
                continue
            elif i == 0:
                con6[i,j,v] = model.addConstr(
                    (servetime[i] + distance[i][j]) <= t[j,v] + M*(1-x[i,j,v]), 
                    'con6[' + str(i) + ',' + str(j) + ',' + str(v) + ']')
            else:
                con6[i,j,v] = model.addConstr(
                    t[i,v]+ servetime[i] + distance[i][j] <= t[j,v]+ M*(1-x[i,j,v]), 
                    'con6[' + str(i) + ',' + str(j) + ',' + str(v) + ']'
                )

# time window constrain - left part
con7 = {}
for v in range(K):
    for i in range(N):  
        con7[i,v] = model.addConstr(t[i,v] >= readytime[i], 'con7[' + str(i) + ',' + str(v) + ']')

# time window constrain - right part
con8 = {}
for v in range(K):
    for i in range(N):  
        con8[i,v] = model.addConstr(t[i,v] <= duetime[i], 'con8[' + str(i) + ',' + str(v) + ']')

# i is visited by v if part of the demand of i is satisfied by v
con9 = {}
for v in range(K):
    for i in range(N):
        if i == 0:
            continue
        else:
            con9[i,v] = model.addConstr(z[i,v] >= p[i,v], 'con9[' + str(i) + ',' + str(v) + ']')

# for each node, each vehicle only only has one entry
con10 = {}
for v in range(K):
    for i in range(N):
        con10[i,v] = model.addConstr(
            quicksum(x[i,j,v] for j in range(N)) - x[i,i,v] == z[i,v],
            'con10[' + str(i) + ',' + str(v) + ']'
        )

# for each node, each vehicle only only has one exit
con11 = {}
for v in range(K):
    for j in range(N):
        con11[j,v] = model.addConstr(
            quicksum(x[i,j,v] for i in range(N)) - x[j,j,v] == z[j,v],
            'con11[' + str(j) + ',' + str(v) + ']'
        )

# vehicle occupation
con12 = {}
for v in range(K):
    con12[v] = model.addConstr(quicksum(z[i,v] for i in range(1,N))/(N-1) <= y[v], 'con12[' + str(v) + ']')


# ----------------------- Set Model Parameters ----------------------
# Set time constraint for optimization (30minutes)
model.setParam('TimeLimit', 30 * 60)
model.setParam('OutputFlag', True)  # silencing gurobi output or not
model.setParam ('MIPGap', 0)        # find the optimal solution
model.write('output2.lp')           # print the model in .lp format file

model.optimize()

# ----------------------- Print Results ----------------------
print ('\n--------------------------------------------------------------------\n')
status = model.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    # Print optimal solution
    print ('Objective function : %10.2f' % model.objVal)
    print("")

    print("Total distance travelled")
    print(sum(x[i,j,v].x * distance[i][j] for i in range(N) for j in range(N) for v in range(K)))
    print("")

    print("Total fixed cost")
    print(sum(y[v].x * cost[v] for v in range(K)))    
    print("")

    print("Total number of nodes: %d"%len(loc_id))
    print("Nodes with split delivery:")
    for i in range(1,N):
        total=0
        for j in range(N):
            for v in range(K):
                if x[i,j,v].x==1:
                    total=total+1
                else:
                    continue
        if total>1:
            print('Node %d:'%i+' %d'%total)
        else:
            continue
    print("")      

    # calculate the number of vehicles used 
    print("Number of vehicles used")
    k=sum(x[0,j,v].x for j in range(N) for v in range(K))
    print(k)
    print("")
    
    complete_data={}
    for v in range(K):
        vehicle_data={}
        for i in range(N):
            if z[i,v].x <= 0.01:
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
        s="vehicle "+str(v+1)+": 0"
        for i in complete_data_ordered[v]:
            s=s+"-"+str(i[0])
        print(s)
    print("")

    # sequence of travel time
    print("Sequence of time")
    for v in complete_data_ordered:
        s="vehicle "+str(v+1)+": 0"
        for i in complete_data_ordered[v]:
            s=s+"-%.0f"%t[i[0],v].x
        print(s)
    print("")

    # Printing load of vehicle arriving at each node
    print("load of vehicle when arriving at node i")
    for v in complete_data_ordered:
        s="vehicle "+str(v+1)+":"
        for i in complete_data_ordered[v]:
            s=s+" %.0f"%l[i[0],v].x
        print(s)
    print("")
    
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