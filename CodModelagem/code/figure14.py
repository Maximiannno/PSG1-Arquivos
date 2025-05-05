from utils import workWithStateData, computeDistanceMatrix, plotRegionalizationHessModel, findRegionsHessModel
import gurobipy as gp
from gurobipy import GRB

file_state_data = "../data/State_Amazonas/Municipios_AM.xlsx"
nrows_file_state_data = 63
file_state_map_json = "../data/State_Amazonas/AM_Municipios_2022/AM_Municipios_2022.json"
object_name = "AM_Municipios_2022"
file_state_map_shapefile = "../data/State_Amazonas/AM_Municipios_2022/AM_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

### Compute matrix of distance between units
distance = computeDistanceMatrix(municipalities)

### Optimization Model

# Number of regions
NUM_OF_REGIONS = 3
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
base = plotRegionalizationHessModel(x, stateMap, municipalities, showCentroids=False, showRegionCentroids=False, showBoundary=False)

#Higlight bordering units
regions_assign, centers = findRegionsHessModel(x, municipalities)

for i in range(len(mun_list)):
    municipalities[mun_list[i]]["REGIAO"] = regions_assign[i]
    
#Iterate in neighborhood finding border units
boundaryUnits = []
for i in municipalities:
    boundaryUnit = False
    for j in municipalities[i]["vizinhos"]:
        if municipalities[i]["REGIAO"] != municipalities[j]["REGIAO"]:
            boundaryUnit = True
    
    if boundaryUnit:
        boundaryUnits.append(municipalities[i]["CD_MUN"])
        
base = stateMap.loc[stateMap["CD_MUN"].isin(boundaryUnits)].boundary.plot(ax=base, color="black")

#Articulation units was identified by simple observation
points = ["1300607","1301407","1304104","1302702"]
stateMap.loc[stateMap["CD_MUN"].isin(points)].centroid.plot(ax=base, color="red", markersize=40)