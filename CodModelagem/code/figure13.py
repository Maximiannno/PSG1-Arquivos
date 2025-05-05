from utils import workWithStateData, computeDistanceMatrix, plotRegionalizationHessModel
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

file_state_data = "../data/State_Roraima/Municipios_RR.xlsx"
nrows_file_state_data = 16
file_state_map_json = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.json"
object_name = "RR_Municipios_2022"
file_state_map_shapefile = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

edges = []

#Make the edges list
for i in municipalities:
    for j in municipalities[i]["vizinhos"]:
        if(int(i) < int(j)):
            edge = {}
            edge["v1"] = i
            edge["v2"] = j
            edge["weight"] = municipalities[i]["vizinhos"][j]["dist_linha"]
            edges.append(edge)
            
fringe = edges.copy()

### Prim's Algorithm to find the Minimum Spanning Tree
mst = []
mst_units = []
while(len(fringe) > 0):
    
    #Select the minimum weighted edge in the fringe
    minEdgeCost = float("inf")
    for i in fringe:
        if i["weight"] < minEdgeCost:
            minEdgeCost = i["weight"]
            minEdge = i
            
    #Add the minEdge to the mst
    mst.append(minEdge)
    
    #Populate the mst units
    if minEdge["v1"] not in mst_units:
        mst_units.append(minEdge["v1"])
    if minEdge["v2"] not in mst_units:
        mst_units.append(minEdge["v2"])
    
    #Redraw the fringe
    newFringe = []
    for i in edges:
        #Just one of the vertices should stay in the mst
        if((i["v1"] in mst_units) ^ (i["v2"] in mst_units)):
            newFringe.append(i)
            
    fringe = newFringe
    
### Plot the map
base = stateMap.plot(figsize=(10, 10), edgecolor="white", cmap=ListedColormap(["#80c1ff"]))

largest_dist = 0

for i in mst:
    r0 = i["v1"]
    r1 = i["v2"]
    plt.plot([municipalities[r0]["centroide"].x, municipalities[r1]["centroide"].x], [municipalities[r0]["centroide"].y, municipalities[r1]["centroide"].y], 'ro-')
    
    if(i["weight"] > largest_dist):
        largest_dist = i["weight"]
        largest_edge = i
        
r0 = largest_edge["v1"]
r1 = largest_edge["v2"]
plt.plot([municipalities[r0]["centroide"].x, municipalities[r1]["centroide"].x], [municipalities[r0]["centroide"].y, municipalities[r1]["centroide"].y], 'go-')

### Select largest edge as the cut edge
reg_tree = mst.copy()
reg_tree.remove(largest_edge)

#Plot the regionalization induced by the MST
region = [0] * len(municipalities)
stateMap["REGION"] = region

next_edge = reg_tree[0]
region_units = []
while(next_edge != False):
    #Assign to the region 1
    stateMap.loc[stateMap["CD_MUN"] == next_edge["v1"], "REGION"] = 1
    stateMap.loc[stateMap["CD_MUN"] == next_edge["v2"], "REGION"] = 1
    #Assign to the region units
    if next_edge["v1"] not in region_units:
        region_units.append(next_edge["v1"])
    if next_edge["v2"] not in region_units:
        region_units.append(next_edge["v2"])
    
    #Remove to the list of edges
    reg_tree.remove(next_edge)
    
    #Find the next edge
    new_next_edge = False
    for i in reg_tree:
        if(i["v1"] in region_units or i["v2"] in region_units):
            new_next_edge = i
            break
        
    next_edge = new_next_edge
    
stateMap.plot(column="REGION",figsize=(10,10),edgecolor="#000000",categorical=True,cmap="tab20")

### Model to find the best regionalization considering pairwise distances

# Number of regions
NUM_OF_REGIONS = 2
# number of units
NUM_UNITS = len(municipalities)
### Compute matrix of distance between units
distance = computeDistanceMatrix(municipalities)

m2 = gp.Model("map")

#Decision Variable
x2 = m2.addVars([(i, j) for i in municipalities for j in municipalities], lb=0, ub=1
              , vtype=GRB.INTEGER
              )
#New decision variables for contiguity constraints
y = m2.addVars([(a, i) for a in arcs for i in municipalities], lb=0)

#Constraints
m2.addConstrs((gp.quicksum(x2[i,j] for j in municipalities) == 1 for i in municipalities),name="atribuicao unica")
m2.addConstr((gp.quicksum(x2[j,j] for j in municipalities) == NUM_OF_REGIONS), name="numero_de_distritos")
m2.addConstrs(x2[i,j] <= x2[j,j] for i in municipalities for j in municipalities)

#Constraints for contiguity guarantee
m2.addConstrs((gp.quicksum(y[(j,i),a] for j in municipalities[i]["vizinhos"]) - gp.quicksum(y[(i,j),a] for j in municipalities[i]["vizinhos"]) == x2[i,a] for i in municipalities for a in municipalities if i != a), name="flow1")
m2.addConstrs((gp.quicksum(y[(j,i),a] for j in municipalities[i]["vizinhos"]) <= (NUM_UNITS - NUM_OF_REGIONS) * x2[i,a] for i in municipalities for a in municipalities if a != i), name="flow2")
m2.addConstrs((gp.quicksum(y[(j,a),a] for j in municipalities[a]["vizinhos"]) == 0 for a in municipalities), name="flow3")

#Objective function
m2.setObjective(gp.quicksum(x2[i,k] * x2[j,k] * distance[(i,j)] for i in municipalities for j in municipalities for k in municipalities), GRB.MINIMIZE)

m2.optimize()

### Generate figure
plotRegionalizationHessModel(x2, stateMap, municipalities, showCentroids=False, showRegionCentroids=False, showBoundary=True)