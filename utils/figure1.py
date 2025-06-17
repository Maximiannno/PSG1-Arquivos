from utils import workWithStateData, computeDistanceMatrix, plotRegionalizationLabelingModel, djikstraDistance
import gurobipy as gp
from gurobipy import GRB

file_state_data = "../data/State_Acre/Municipios_AC.xlsx"
nrows_file_state_data = 23
file_state_map_json = "../data/State_Acre/AC_Municipios_2022/AC_Municipios_2022.json"
object_name = "AC_Municipios_2022"
file_state_map_shapefile = "../data/State_Acre/AC_Municipios_2022/AC_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

### Compute matrix of distance between units
distance = computeDistanceMatrix(municipalities)

### Optimization Model - Optimizes territorial area equity

# Number of regions
NUM_OF_REGIONS = 2
# number of units
NUM_UNITS = len(municipalities)

m = gp.Model("map")

#Decision variables - Labeling
x = m.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub=1
              , vtype=GRB.INTEGER
              )

w = m.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub = 1,
              vtype=GRB.INTEGER)

y = m.addVars([(i, j, c) for i in municipalities for j in municipalities for c in range(NUM_OF_REGIONS)], lb=0)

#Control variables for area equity objective
tam = m.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)
dis = m.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)

#Constraints
m.addConstrs((gp.quicksum(x[i,c] for c in range(NUM_OF_REGIONS)) == 1 for i in municipalities),name="atribuicao unica")
m.addConstrs((gp.quicksum(w[i,c] for i in municipalities) == 1 for c in range(NUM_OF_REGIONS)), name="raiz unica")
m.addConstrs((w[i,c] <= x[i,c] for i in municipalities for c in range(NUM_OF_REGIONS)),name="atribuicao raiz")

#M Constant
M = NUM_UNITS - NUM_OF_REGIONS + 1

# Contiguity constraints
m.addConstrs((y[i,j,c] <= x[i,c] * M for i in municipalities for j in municipalities[i]["vizinhos"] for c in range(NUM_OF_REGIONS)), name="flow1")
m.addConstrs((y[i,j,c] <= x[j,c] * M for i in municipalities for j in municipalities[i]["vizinhos"] for c in range(NUM_OF_REGIONS)), name="flow2")
m.addConstrs((gp.quicksum(y[i,j,c] for j in municipalities[i]["vizinhos"]) - gp.quicksum(y[j,i,c] for j in municipalities[i]["vizinhos"]) >= x[i,c] - M * w[i,c] for i in municipalities for c in range(NUM_OF_REGIONS)), name="flow3")

# Constraints to calculate area of regions
MEAN_AREA = stateArea/NUM_OF_REGIONS
m.addConstrs(tam[c] == gp.quicksum(x[i,c] * municipalities[i]["AREA_KM2"] for i in municipalities) for c in range(NUM_OF_REGIONS))
m.addConstrs(dis[c] >= tam[c]-MEAN_AREA for c in range(NUM_OF_REGIONS))
m.addConstrs(dis[c] >= MEAN_AREA-tam[c] for c in range(NUM_OF_REGIONS))

#Objective function
m.setObjective(dis.sum(), GRB.MINIMIZE)

m.optimize()

### Generate figure
base = plotRegionalizationLabelingModel(x, w, stateMap, municipalities, showCentroids=False, showRegionCentroids=False, showBoundary=True)


### New Optimization Model - Optimizes population equity

m = gp.Model("map")

#Decision variables - Labeling
x = m.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub=1
              , vtype=GRB.INTEGER
              )

w = m.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub = 1,
              vtype=GRB.INTEGER)

y = m.addVars([(i, j, c) for i in municipalities for j in municipalities for c in range(NUM_OF_REGIONS)], lb=0)

#Control variables for population equity objective
pop = m.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)
dis = m.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)

#Constraints
m.addConstrs((gp.quicksum(x[i,c] for c in range(NUM_OF_REGIONS)) == 1 for i in municipalities),name="atribuicao unica")
m.addConstrs((gp.quicksum(w[i,c] for i in municipalities) == 1 for c in range(NUM_OF_REGIONS)), name="raiz unica")
m.addConstrs((w[i,c] <= x[i,c] for i in municipalities for c in range(NUM_OF_REGIONS)),name="atribuicao raiz")

#M Constant
M = NUM_UNITS - NUM_OF_REGIONS + 1

# Contiguity constraints
m.addConstrs((y[i,j,c] <= x[i,c] * M for i in municipalities for j in municipalities[i]["vizinhos"] for c in range(NUM_OF_REGIONS)), name="flow1")
m.addConstrs((y[i,j,c] <= x[j,c] * M for i in municipalities for j in municipalities[i]["vizinhos"] for c in range(NUM_OF_REGIONS)), name="flow2")
m.addConstrs((gp.quicksum(y[i,j,c] for j in municipalities[i]["vizinhos"]) - gp.quicksum(y[j,i,c] for j in municipalities[i]["vizinhos"]) >= x[i,c] - M * w[i,c] for i in municipalities for c in range(NUM_OF_REGIONS)), name="flow3")

# Constraints to calculate population of regions
MEAN_POP = statePop/NUM_OF_REGIONS
m.addConstrs(pop[c] == gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) for c in range(NUM_OF_REGIONS))
m.addConstrs(dis[c] >= pop[c]-MEAN_POP for c in range(NUM_OF_REGIONS))
m.addConstrs(dis[c] >= MEAN_POP-pop[c] for c in range(NUM_OF_REGIONS))

#Objective function
m.setObjective(dis.sum(), GRB.MINIMIZE)

m.optimize()

### Generate figure
base = plotRegionalizationLabelingModel(x, w, stateMap, municipalities, showCentroids=False, showRegionCentroids=False, showBoundary=True)