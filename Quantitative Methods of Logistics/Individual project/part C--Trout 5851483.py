#!/usr/bin/env python
# coding: utf-8

# ## Assignment #1

# Part C: addition of personnel constraints

# In[1]:


from gurobipy import *

model = Model ('Production Factory')


# ---- Parameters ----


# Product characteristics
productname  = ('one', 'two', 'three')
holdcost   = (6, 8, 10)        # euros / (product-month)


NrOfMonths = 12                                # planning horizon 
                                            
proddemand =  [[750, 650, 600, 500, 130.3, 650, 600, 750, 650, 600, 500, 550],   # product demand for each month 
              [550, 500, 450, 275, 350, 300, 500, 600, 500, 400.6, 300, 250],
              [550, 500, 500, 320.5, 300, 150.2, 225, 500, 450, 350, 300, 350]]
       
    
 # Personnel characteristics
    
perscost = (2000, 2000, 2500, 2500, 2500, 3000, 
            3000, 3000, 2500, 2500, 2000, 2000 )  # euro/ (employee-month)
prodamount = (15, 20, 10)              # products/ (month-employee)


# ---- Sets ----

P = range (len (productname) )                # set of products                
M = range (NrOfMonths)                      #set of months in year

# ---- Variables ----

# Decision Variables: 
#NEW: this time the employees can't change between products within the same month
z = {}                 #z(p,m) (employees working full time, producing product p in month m)
for p in P:
    for m in M:
            z[m,p] = model.addVar (lb = 0, vtype = GRB.INTEGER, name = 'Z[' + str(m) + ',' + str(p) + ']')
# Integrate new variables
model.update ()

h = {}           #h(p,m) (number of products stored per month)
for p in P:
    for m in M:
        h[m,p] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS, name = 'H[' + str(m) + ',' + str(p) + ']')
# Integrate new variables
model.update ()

# ---- Objective Function ----

model.setObjective ( quicksum (holdcost[p] * h[m,p] for p in P for m in M) + quicksum (perscost[m] * z[m,p] for p in P for m in M)) 
model.modelSense = GRB.MINIMIZE
model.update ()


# ---- Constraints ----

# Constraint 1: if production beats demand in 1st month, then the product gets stored
con1 = {}
for p in P:
        con1[p] = model.addConstr(  h[0,p] == ((z[0,p] * prodamount[p]) - proddemand[p][0]), 'con1[' + str(p) + ']-')
            
# Constraint 2: if production beats demand in month, then the product gets stored
con2 = {}
for p in P:
    for m in range (1, len(M)):
        con2[m,p] = model.addConstr( h[m,p] == ((z[m,p] * prodamount[p] + h[M[m-1],p]) - proddemand[p][m]), 'con2[' + str(m) + ',' + str(p) + ']-')
                               
# NEW Constraint 3: the total of the personnel must be the same for the 1st quarter
con3 = {}
for m in range (0, 2):
    con3[m,p] = model.addConstr( quicksum (z[m,p] for p in P) - quicksum (z[M[m+1],p] for p in P) == 0, 'con3[' + str(p) + ']-')

# NEW Constraint 4: the total of the personnel must be the same for the 2nd quarter
con4 = {}
for m in range (3, 5):
    con4[m,p] = model.addConstr( quicksum (z[m,p] for p in P) - quicksum (z[M[m+1],p] for p in P) == 0, 'con4[' + str(p) + ']-')

# NEW Constraint 5: the total of the personnel must be the same for the 3rd quarter
con5 = {}
for m in range (6, 8):
    con5[m,p] = model.addConstr( quicksum (z[m,p] for p in P) - quicksum (z[M[m+1],p] for p in P) == 0, 'con5[' + str(p) + ']-')

# NEW Constraint 6: the total of the personnel must be the same for the 4th quarter
con6 = {}
for m in range (9, 11):
    con6[m,p] = model.addConstr( quicksum (z[m,p] for p in P) - quicksum (z[M[m+1],p] for p in P) == 0, 'con6[' + str(p) + ']-')
           

# ---- Solve ----

model.setParam('OutputFlag', True) # silencing gurobi output or not
model.setParam ('MIPGap', 0);       # find the optimal solution
model.write("output.lp")            # print the model in .lp format file

model.optimize ()

# --- Print results ---
print ('\n--------------------------------------------------------------------\n')
    
if model.status == GRB.Status.OPTIMAL: # If optimal solution is found
    print ('Total cost : %10.2f euro' % model.objVal)
    print ('Total personnel cost : %10.2f euro' % sum (perscost[m] * sum (z[m,p].x for p in P) for m in M))
    print ('Total holding cost: %10.2f euro' % sum (holdcost[p] * h[m,p].x for p in P for m in M))
    print ('')
    print ('All decision variables:\n')
    
    month  = ('one', 'two', 'three', 'sum', '    holding cost or personnel cost per month')
    D = range(len(month))
    variables = ('Z', 'H')
    for m in M:
        s = 'Month %d' % (m+1)
        for d in D:
            s = s + '%8s' % month[d]
        print (s)
        
        s =  '%8s' % variables[0] 
        print (s)
        for p in P:
            s = s + '%8.1f' % z[m,p].x
        s = s + '%8.1f' % sum (z[m,p].x for p in P)
        s = s + '%12.1f' % (perscost[m] * (sum (z[m,p].x for p in P)))
        print (s)    

        s =  '%8s' % variables[1]
        for p in P:
            s = s + '%8.1f' % h[m,p].x
        s = s + '%8.1f' % sum (h[m,p].x for p in P)
        s = s + '%12.1f' % sum (holdcost[p] * h[m,p].x for p in P)
        print (s)    

        print ('\n')       

else:
    print ('\nNo feasible solution found')

print ('\nREADY\n')

