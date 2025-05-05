from utils import workWithStateData, computeDistanceMatrix, plotRegionalizationHessModel
import gurobipy as gp
from gurobipy import GRB

file_state_data = "../data/State_RioGrandeDoNorte/Municipios_RN.xlsx"
nrows_file_state_data = 168
file_state_map_json = "../data/State_RioGrandeDoNorte/RN_Municipios_2022/RN_Municipios_2022.json"
object_name = "RN_Municipios_2022"
file_state_map_shapefile = "../data/State_RioGrandeDoNorte/RN_Municipios_2022/RN_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

### Compute matrix of distance between units
distance = computeDistanceMatrix(municipalities)

### Optimization Model

# Number of regions
NUM_OF_REGIONS = 8
# number of units
NUM_UNITS = len(municipalities)

m = gp.Model("map")

x = m.addVars([(i, j) for i in municipalities for j in municipalities], lb=0, ub=1
              , vtype=GRB.INTEGER
              )

#Constraints
m.addConstrs((gp.quicksum(x[i,j] for j in municipalities) == 1 for i in municipalities),name="atribuicao unica")
m.addConstr((gp.quicksum(x[j,j] for j in municipalities) == NUM_OF_REGIONS), name="numero_de_distritos")
m.addConstrs(x[i,j] <= x[j,j] for i in municipalities for j in municipalities)

#Objective function
m.setObjective(gp.quicksum(x[i,j] * distance[(i,j)] for i in municipalities for j in municipalities), GRB.MINIMIZE)

m.optimize()

### Generate figure
plotRegionalizationHessModel(x, stateMap, municipalities, showCentroids=True, showRegionCentroids=True, showBoundary=True)