from utils import workWithStateData, computeDistanceMatrix, plotRegionalizationHessModelMulticolors
import gurobipy as gp
from gurobipy import GRB
import random

file_state_data = "../data/State_SaoPaulo/Municipios_SP.xlsx"
nrows_file_state_data = 646
file_state_map_json = "../data/State_SaoPaulo/SP_Municipios_2022/SP_Municipios_2022.json"
object_name = "SP_Municipios_2022"
file_state_map_shapefile = "../data/State_SaoPaulo/SP_Municipios_2022/SP_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

### Compute matrix of distance between units
distance = computeDistanceMatrix(municipalities)

### Optimization Model

# Number of regions
NUM_OF_REGIONS = 62
# number of units
NUM_UNITS = len(municipalities)

m = gp.Model("map")

#Decision variables
x = m.addVars([(i, j) for i in municipalities for j in municipalities], lb=0, ub=1
              , vtype=GRB.INTEGER
              )
y = m.addVars([(a, i) for a in arcs for i in municipalities], lb=0)

#Constraints
m.addConstrs((gp.quicksum(x[i,j] for j in municipalities) == 1 for i in municipalities),name="atribuicao unica")
m.addConstr((gp.quicksum(x[j,j] for j in municipalities) == NUM_OF_REGIONS), name="numero_de_distritos")
m.addConstrs(x[i,j] <= x[j,j] for i in municipalities for j in municipalities)

# Contiguity Constraints
m.addConstrs((gp.quicksum(y[(j,i),a] for j in municipalities[i]["vizinhos"]) - gp.quicksum(y[(i,j),a] for j in municipalities[i]["vizinhos"]) == x[i,a] for i in municipalities for a in municipalities if i != a), name="flow1")
m.addConstrs((gp.quicksum(y[(j,i),a] for j in municipalities[i]["vizinhos"]) <= (NUM_UNITS - NUM_OF_REGIONS) * x[i,a] for i in municipalities for a in municipalities if a != i), name="flow2")
m.addConstrs((gp.quicksum(y[(j,a),a] for j in municipalities[a]["vizinhos"]) == 0 for a in municipalities), name="flow3")

#Objective function
m.setObjective(gp.quicksum(x[i,j] * distance[(i,j)] for i in municipalities for j in municipalities), GRB.MINIMIZE)

m.optimize()

### Generate figure
plotRegionalizationHessModelMulticolors(x, stateMap, municipalities, showCentroids=False, showRegionCentroids=True, showBoundary=False, numColors=110)


### New Optimization Model - Location-Allocation

m2 = gp.Model("map")

#Decision Variables
x2 = m2.addVars([(i, j) for i in municipalities for j in municipalities], lb=0, ub=1
              , vtype=GRB.INTEGER
              )
y2 = m2.addVars([(a, i) for a in arcs for i in municipalities], lb=0)

#Constraints
m2.addConstrs((gp.quicksum(x2[i,j] for j in municipalities) == 1 for i in municipalities),name="atribuicao unica")
m2.addConstr((gp.quicksum(x2[j,j] for j in municipalities) == NUM_OF_REGIONS), name="numero_de_distritos")
m2.addConstrs(x2[i,j] <= x2[j,j] for i in municipalities for j in municipalities)

#New constraints for contiguity guarantee
m2.addConstrs((gp.quicksum(y2[(j,i),a] for j in municipalities[i]["vizinhos"]) - gp.quicksum(y2[(i,j),a] for j in municipalities[i]["vizinhos"]) == x2[i,a] for i in municipalities for a in municipalities if i != a), name="flow1")
m2.addConstrs((gp.quicksum(y2[(j,i),a] for j in municipalities[i]["vizinhos"]) <= (NUM_UNITS - NUM_OF_REGIONS) * x2[i,a] for i in municipalities for a in municipalities if a != i), name="flow2")
m2.addConstrs((gp.quicksum(y2[(j,a),a] for j in municipalities[a]["vizinhos"]) == 0 for a in municipalities), name="flow3")

#Fixing roots of regions
disp = mun_list
#SP - Fixing some cities as a root
m2.addConstr(x2["3520400","3520400"] == 1)
m2.addConstr(x2["3500600","3500600"] == 1)
for ind in range(NUM_OF_REGIONS-2):
    center = random.choice(disp)
    m2.addConstr(x2[center,center] == 1)
    disp.remove(center)

#Objective function
m2.setObjective(gp.quicksum(x2[i,j] * distance[(i,j)] for i in municipalities for j in municipalities), GRB.MINIMIZE)

m2.optimize()

### Generate figure
plotRegionalizationHessModelMulticolors(x2, stateMap, municipalities, showCentroids=False, showRegionCentroids=True, showBoundary=False, numColors=110)