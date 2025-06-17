from utils import workWithStateData
import matplotlib.pyplot as plt
import math

file_state_data = "../data/State_Roraima/Municipios_RR.xlsx"
nrows_file_state_data = 16
file_state_map_json = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.json"
object_name = "RR_Municipios_2022"
file_state_map_shapefile = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.shp"
NUM_OF_REGIONS = 2

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)
NUM_UNITS = len(municipalities)

# Auxiliary arrays
areas = []
for i in municipalities:
    areas.append(municipalities[i]["AREA_KM2"])
    
ids = []
for i in range(NUM_UNITS):
    ids.append(int(math.pow(2,i)))
    
populations = []
for i in municipalities:
    populations.append(municipalities[i]["NUM_HABITANTES"])
    
seqs = []
for i in range(NUM_UNITS):
    seqs.append(i)
    
stateMap["id"] = ids
stateMap["seq"] = seqs
    
#Biggest possible combination
MAXIMUM = int(math.pow(2,NUM_UNITS))-1

# Objective Functions
def calculateF1(finsol):
    reg1 = sum(list(map(lambda x, y: x * y, finsol["atribuicao"], areas)))
    reg0 = stateArea-reg1
    total = abs(reg0 - reg1)
    finsol["f1"] = total

def calculateF2(finsol):
    reg1 = sum(list(map(lambda x, y: x * y, finsol["atribuicao"], populations)))
    reg0 = statePop-reg1
    total = abs(reg0 - reg1)
    finsol["f2"] = total

def isFeasible(solucao, num_regioes):
    if(solucao["tupla"][0]==0 or solucao["tupla"][1]==0):
        return False

    i = 1
    q = []
    genX = [0 for i in range(NUM_UNITS)]
    
    for ind, cid in stateMap.iterrows():
        if(genX[ind] != 0):
            continue
        
        genX[ind] = i;
        q.append(cid)
        
        while(len(q) > 0):
            cidU = q.pop(0)
            vizU = municipalities[cidU["CD_MUN"]]["vizinhos"]
            
            for cidV2 in vizU:
                cidV = stateMap.loc[stateMap[stateMap["CD_MUN"]==cidV2].index[0]]
                if(genX[cidV["seq"]] != 0):
                    continue
                
                if(solucao["atribuicao"][cidU["seq"]] == solucao["atribuicao"][cidV["seq"]]):
                    genX[cidV["seq"]] = genX[cidU["seq"]]
                    q.append(cidV)
                    
        i = i + 1
        
    if((i-1) <= num_regioes):
        return True
    else:
        return False


#Generate all possible solutions
def genRec(ant, pos, solucoes):
    
    for i in range(NUM_OF_REGIONS):
        sol = ant.copy()
        sol.append(i)
        if(pos == NUM_UNITS - 1):
            reg1 = sum(list(map(lambda x, y: x * y, sol, ids)))
            reg0 = MAXIMUM-reg1
            tupla = (reg0, reg1)
            finSol = {}
            finSol["atribuicao"] = sol
            finSol["tupla"] = tupla
            finSol["contiguo"] = isFeasible(finSol, NUM_OF_REGIONS)
            if(finSol["contiguo"] == True):
                calculateF1(finSol)
                calculateF2(finSol)
            solucoes.append(finSol)
        else:
            genRec(sol, pos+1, solucoes)
            
        if len(solucoes)%1000 == 0:
            print(len(solucoes), "solutions generated of", NUM_OF_REGIONS ** NUM_UNITS)

solucoes = []
ant = []
genRec(ant, 0, solucoes)

#Once the solutions are generated, plot the map
x_cont = []
y_cont = []
x_ncont = []
y_ncont = []

#Plot all solutions
for i in solucoes:
    if i["contiguo"]:
        x_cont.append(i["tupla"][0])
        y_cont.append(i["tupla"][1])
    else:
        x_ncont.append(i["tupla"][0])
        y_ncont.append(i["tupla"][1])

plt.figure(figsize=(10,10),dpi=100)
plt.xlabel("Region 1")
plt.ylabel("Region 2")
plt.scatter(x_ncont, y_ncont, color="red")
plt.scatter(x_cont, y_cont, color="blue")

#Plot only solutions where x in [14400,14450]
x_cont = []
y_cont = []
x_ncont = []
y_ncont = []
for i in solucoes:
    if i["tupla"][0] >= 14400 and i["tupla"][0] <= 14450:
        if i["contiguo"]:
            x_cont.append(i["tupla"][0])
            y_cont.append(i["tupla"][1])
        else:
            x_ncont.append(i["tupla"][0])
            y_ncont.append(i["tupla"][1])

plt.figure(figsize=(7,7),dpi=100)
plt.xlabel("Region 1")
plt.ylabel("Region 2")
plt.scatter(x_ncont, y_ncont, color="red")
plt.scatter(x_cont, y_cont, color="blue")
#highlight chosen solution in green
plt.scatter(14432, 18335, color="green")

### Plot the objective space
x_obj = []
y_obj = []
contiguous_solutions = []
for i in solucoes:
    if i["contiguo"]:
        contiguous_solutions.append(i)
        x_obj.append(i["f1"]/stateArea)
        y_obj.append(i["f2"]/statePop)

plt.figure(figsize=(7,7),dpi=100)
plt.xlim(-0.01,1.02)
plt.ylim(-0.01,1.01)
plt.xlabel("$f_1(K)$")
plt.ylabel("$f_2(K)$")
plt.scatter(x_obj, y_obj)

### Generate the pareto-front
pareto_front = []
for i in contiguous_solutions:
    existe_melhor = False
    for j in contiguous_solutions:
        if j["f1"] <= i["f1"] and j["f2"] <= i["f2"] and (j["f1"] < i["f1"] or j["f2"] < i["f2"]):
            existe_melhor = True
            break
    if not existe_melhor:
        pareto_front.append(i)
        
pf_x = []
pf_y = []
for i in pareto_front:
    pf_x.append(i["f1"]/stateArea)
    pf_y.append(i["f2"]/statePop)
    
#Find ideal point = the one with the best value in all functions
ideal_x = min(x_obj)
ideal_y = min(y_obj)

#Find nadir point = the worst values from pareto-front
nadir_x = max(pf_x)
nadir_y = max(pf_y)
    
plt.figure(figsize=(7,7),dpi=100)
plt.xlim(-0.01,1.02)
plt.ylim(-0.01,1.01)
plt.xlabel("$f_1(K)$")
plt.ylabel("$f_2(K)$")
plt.scatter(x_obj, y_obj)
plt.scatter(pf_x, pf_y, color="green")
plt.scatter(ideal_x, ideal_y, color="red")
plt.annotate("ideal", (ideal_x+0.01, ideal_y))
plt.scatter(nadir_x, nadir_y, color="red")
plt.annotate("nadir", (nadir_x+0.01, nadir_y))
plt.scatter(0, 0, color="red")
plt.annotate("utopia", (0.01, 0))

### Find the best candidate based on euclidean distance
min_dist = float("inf")
best_candidate_x = 0
best_candidate_y = 0
for i in range(len(pf_x)):
    dist_atu = math.sqrt(pf_x[i]**2 + pf_y[i]**2)
    if(dist_atu < min_dist):
        min_dist = dist_atu
        best_candidate_x = pf_x[i]
        best_candidate_y = pf_y[i]
        
plt.figure(figsize=(7,7),dpi=100)
plt.xlim(-0.01,1.02)
plt.ylim(-0.01,1.01)
plt.xlabel("$f_1(K)$")
plt.ylabel("$f_2(K)$")
plt.scatter(x_obj, y_obj)
plt.scatter(pf_x, pf_y, color="green")
plt.scatter(ideal_x, ideal_y, color="red")
plt.annotate("ideal", (ideal_x+0.01, ideal_y))
plt.scatter(nadir_x, nadir_y, color="red")
plt.annotate("nadir", (nadir_x+0.01, nadir_y))
plt.scatter(0, 0, color="red")
plt.annotate("utopia", (0.01, 0))
plt.scatter(best_candidate_x, best_candidate_y, color="yellow")
plt.arrow(0.02,0.02,best_candidate_x-0.03, best_candidate_y-0.05,linestyle=':',width=0.0005, edgecolor="grey", head_width=0.01)
