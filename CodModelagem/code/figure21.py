from utils import workWithStateData, plotRegionalizationLabelingModel
import gurobipy as gp
from gurobipy import GRB

file_state_data = "../data/State_Roraima/Municipios_RR.xlsx"
nrows_file_state_data = 16
file_state_map_json = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.json"
object_name = "RR_Municipios_2022"
file_state_map_shapefile = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)
NUM_UNITS = len(municipalities)
NUM_OF_REGIONS = 2

m = gp.Model("mapa")

# número de municípios
x = m.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub=1
              , vtype=GRB.INTEGER
              )

w = m.addVars([(i, c) for i in municipalities for c in range(NUM_OF_REGIONS)], lb=0, ub = 1,
              vtype=GRB.INTEGER)

y = m.addVars([(i, j, c) for i in municipalities for j in municipalities for c in range(NUM_OF_REGIONS)], lb=0)

tam = m.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)
pop = m.addVars([c for c in range(NUM_OF_REGIONS)],lb=0)

f1 = m.addVar()

m.addConstrs((gp.quicksum(x[i,c] for c in range(NUM_OF_REGIONS)) == 1 for i in municipalities),name="1")
m.addConstrs((gp.quicksum(w[i,c] for i in municipalities) == 1 for c in range(NUM_OF_REGIONS)), name="2")
m.addConstrs((w[i,c] <= x[i,c] for i in municipalities for c in range(NUM_OF_REGIONS)),name="3")

#Contiguity constraints
M = NUM_UNITS - NUM_OF_REGIONS + 1
m.addConstrs((y[i,j,c] <= x[i,c] * M for i in municipalities for j in municipalities[i]["vizinhos"] for c in range(NUM_OF_REGIONS)), name="flow1")
m.addConstrs((y[i,j,c] <= x[j,c] * M for i in municipalities for j in municipalities[i]["vizinhos"] for c in range(NUM_OF_REGIONS)), name="flow2")
m.addConstrs((gp.quicksum(y[i,j,c] for j in municipalities[i]["vizinhos"]) - gp.quicksum(y[j,i,c] for j in municipalities[i]["vizinhos"]) >= x[i,c] - M * w[i,c] for i in municipalities for c in range(NUM_OF_REGIONS)), name="flow3")

#Constraints to work with area and population
m.addConstrs(tam[c] == gp.quicksum(x[i,c] * municipalities[i]["AREA_KM2"] for i in municipalities) for c in range(NUM_OF_REGIONS))
m.addConstrs(pop[c] == gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) for c in range(NUM_OF_REGIONS))

#Objective function f1 - area equality
m.addConstr(f1 >= (tam[0] - tam[1])/stateArea)
m.addConstr(f1 >= (tam[1] - tam[0])/stateArea)

#Treat population equality as a constraint
DESV_MAX_EQPOP = 0.2
MEAN_POP = statePop/NUM_OF_REGIONS
LOWER_BOUND = MEAN_POP - (statePop*DESV_MAX_EQPOP)
UPPER_BOUND = MEAN_POP + (statePop*DESV_MAX_EQPOP)
constrUpper = m.addConstrs((gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) <= UPPER_BOUND for c in range(NUM_OF_REGIONS)), name="upperBound")
constrLower = m.addConstrs((gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) >= LOWER_BOUND for c in range(NUM_OF_REGIONS)), name="lowerBound")

#Optimization tolerance 0.2
m.setObjective(f1, GRB.MINIMIZE)
m.optimize()

plotRegionalizationLabelingModel(x, w, stateMap, municipalities, showCentroids=False, showRegionCentroids=False, showBoundary=True)

#Optimization tolerance 0.25
DESV_MAX_EQPOP = 0.25
MEAN_POP = statePop/NUM_OF_REGIONS
LOWER_BOUND = MEAN_POP - (statePop*DESV_MAX_EQPOP)
UPPER_BOUND = MEAN_POP + (statePop*DESV_MAX_EQPOP)

m.remove(constrUpper)
m.remove(constrLower)
constrUpper = m.addConstrs((gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) <= UPPER_BOUND for c in range(NUM_OF_REGIONS)), name="upperBound")
constrLower = m.addConstrs((gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) >= LOWER_BOUND for c in range(NUM_OF_REGIONS)), name="lowerBound")

m.setObjective(f1, GRB.MINIMIZE)
m.optimize()

plotRegionalizationLabelingModel(x, w, stateMap, municipalities, showCentroids=False, showRegionCentroids=False, showBoundary=True)

#Optimization tolerance 0.3
DESV_MAX_EQPOP = 0.3
MEAN_POP = statePop/NUM_OF_REGIONS
LOWER_BOUND = MEAN_POP - (statePop*DESV_MAX_EQPOP)
UPPER_BOUND = MEAN_POP + (statePop*DESV_MAX_EQPOP)

m.remove(constrUpper)
m.remove(constrLower)
constrUpper = m.addConstrs((gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) <= UPPER_BOUND for c in range(NUM_OF_REGIONS)), name="upperBound")
constrLower = m.addConstrs((gp.quicksum(x[i,c] * municipalities[i]["NUM_HABITANTES"] for i in municipalities) >= LOWER_BOUND for c in range(NUM_OF_REGIONS)), name="lowerBound")

m.setObjective(f1, GRB.MINIMIZE)
m.optimize()

plotRegionalizationLabelingModel(x, w, stateMap, municipalities, showCentroids=False, showRegionCentroids=False, showBoundary=True)