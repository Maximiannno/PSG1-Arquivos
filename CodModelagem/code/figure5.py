from utils import workWithStateData, computeDistanceMatrix, plotRegionalizationLabelingModel, djikstraDistance
import gurobipy as gp
from gurobipy import GRB

file_state_data = "../data/State_Roraima/Municipios_RR.xlsx"
nrows_file_state_data = 168
file_state_map_json = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.json"
object_name = "RR_Municipios_2022"
file_state_map_shapefile = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

### Compute matrix of distance between units
distance = computeDistanceMatrix(municipalities)

### Optimization Model

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
base = plotRegionalizationLabelingModel(x, w, stateMap, municipalities, showCentroids=True, showRegionCentroids=False, showBoundary=True)

#Now, it's necessary highligh the units that have a minimum path
points = ["1400159","1400308"]
base_a = stateMap.loc[stateMap["CD_MUN"].isin(points)].centroid.plot(ax=base, color="orange", markersize=50)

points2 = ["1400100"]
stateMap.loc[stateMap["CD_MUN"].isin(points2)].centroid.plot(ax=base_a, color="purple", markersize=50)


### New Optimization Model - Contiguity in shortest paths

# Compute matrix D of shortest paths with Djikstra's Algorithm
distance_D = djikstraDistance(municipalities, distance)

m2 = gp.Model("map")

#Decision Variable
x2 = m2.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub=1
              , vtype=GRB.INTEGER
              )
w2 = m2.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub = 1,
              vtype=GRB.INTEGER)

#Control variables for area equity objective
tam = m2.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)
dis = m2.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)

#Constraints
m2.addConstrs((gp.quicksum(x2[i,c] for c in range(NUM_OF_REGIONS)) == 1 for i in municipalities),name="atribuicao unica")
m2.addConstrs((gp.quicksum(w2[i,c] for i in municipalities) == 1 for c in range(NUM_OF_REGIONS)), name="raiz unica")
m2.addConstrs((w2[i,c] <= x2[i,c] for i in municipalities for c in range(NUM_OF_REGIONS)),name="atribuicao raiz")

#Contiguity in shortest paths constraint
m2.addConstrs(x2[i,c] + x2[j,c] - 1 <= x2[r,c] for i in municipalities for j in municipalities if i != j for r in distance_D[i][j]["vertices"] for c in range(NUM_OF_REGIONS))

# Constraints to calculate area of regions
MEAN_AREA = stateArea/NUM_OF_REGIONS
m2.addConstrs(tam[c] == gp.quicksum(x2[i,c] * municipalities[i]["AREA_KM2"] for i in municipalities) for c in range(NUM_OF_REGIONS))
m2.addConstrs(dis[c] >= tam[c]-MEAN_AREA for c in range(NUM_OF_REGIONS))
m2.addConstrs(dis[c] >= MEAN_AREA-tam[c] for c in range(NUM_OF_REGIONS))

#Objective function
m2.setObjective(dis.sum(), GRB.MINIMIZE)

m2.optimize()

### Generate figure
base = plotRegionalizationLabelingModel(x2, w2, stateMap, municipalities, showCentroids=True, showRegionCentroids=False, showBoundary=True)

#Now, it's necessary highligh the units that have a minimum path
points = ["1400159","1400308"]
base_a = stateMap.loc[stateMap["CD_MUN"].isin(points)].centroid.plot(ax=base, color="orange", markersize=50)

points2 = ["1400100"]
stateMap.loc[stateMap["CD_MUN"].isin(points2)].centroid.plot(ax=base_a, color="purple", markersize=50)