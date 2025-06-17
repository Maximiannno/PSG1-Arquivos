from utils import workWithStateData
import geopandas
import pandas as pd
from matplotlib.colors import ListedColormap
import locale
import matplotlib.pyplot as plt

def find_largest_similarity(regions):
    lowest = float("inf")
    lowest_pair = ()
    
    for i in regions:
        for j in regions[i]["NEIGHBORHOOD"]:
            diff = abs(regions[i]["AVERAGE_INCOME"]-regions[j]["AVERAGE_INCOME"])
            if(diff < lowest):
                lowest = diff
                lowest_pair = (i,j)
                
    return lowest_pair

def merge_regions(regions, pair, municipalities, regions_memory):
    receiver = pair[0]
    donor = pair[1]
    
    #0 add a edge between the most proximous neighboring units between donor and receiver
    min_dist = float("inf")
    lowest_pair = ()
    for i in regions[donor]["municipalities"]:
        for j in municipalities[i]["vizinhos"]:
            if municipalities[j]["REGION_ID"] == receiver:
                if(municipalities[i]["vizinhos"][j]["dist_linha"] < min_dist):
                    min_dist = municipalities[i]["vizinhos"][j]["dist_linha"]
                    lowest_pair = (i,j)
    merge_map.append(lowest_pair)
    
    #1 add municipalities from donor to receiver
    regions[receiver]["municipalities"].extend(regions[donor]["municipalities"])
    
    #2 remove the items from neighbors list
    regions[receiver]["NEIGHBORHOOD"].remove(donor)
    regions[donor]["NEIGHBORHOOD"].remove(receiver)
    
    #3 Change region id in municipalities
    for i in regions[donor]["municipalities"]:
        municipalities[i]["REGION_ID"] = receiver
        
    #4 Change all neighbors from the donor, to refer to the receiver
    for i in regions[donor]["NEIGHBORHOOD"]:
        regions[i]["NEIGHBORHOOD"].remove(donor)
        if(receiver not in regions[i]["NEIGHBORHOOD"]):
            regions[i]["NEIGHBORHOOD"].append(receiver)
    
    #5 Recalculate average income weighted by population
    totalPop = 0
    for i in regions[receiver]["municipalities"]:
        totalPop = totalPop + municipalities[i]["NUM_HABITANTES"]
    
    avgInc = 0
    for i in regions[receiver]["municipalities"]:
        avgInc = avgInc + (municipalities[i]["NUM_HABITANTES"]/totalPop)*municipalities[i]["AVERAGE_INCOME"]
    regions[receiver]["AVERAGE_INCOME"] = avgInc
    
    #6 Assign the donor neighborhood to the receiver
    for i in regions[donor]["NEIGHBORHOOD"]:
        if(i not in regions[receiver]["NEIGHBORHOOD"]):
            regions[receiver]["NEIGHBORHOOD"].append(i)
            
    #7 Change the regions memory
    if(donor in regions_memory):
        regions_memory[receiver] = regions_memory[donor]
        del regions_memory[donor]
    
    #8 Remove the donor region
    del regions[donor]
    
    #In the last removal, remove the region memory
    if len(regions) == 1:
        del regions_memory[receiver]


def generate_color_map(regions_memory):
    num_entries = len(regions_memory)
    
    if num_entries == 0:
        return ListedColormap(["white"])
    if num_entries == 1:
        if next_free_region[0] == 6:
            return ListedColormap(["#80c1ff"])
        else:
            return ListedColormap(["white","#80c1ff"])
    if num_entries == 2:
        if next_free_region[0] == 5:
            return ListedColormap(["white","#80c1ff","#ff80b5"])
        else:
            return ListedColormap(["white","#80c1ff","#80ffa0"])
    if num_entries == 3:
        return ListedColormap(["white","#80c1ff","#80ffa0","#ffdd80"])


def plot_regions_map(stateMap, regions, regions_memory, next_free_region):
    
    num_municipalities = len(stateMap.index)
    regions_list = [0] * num_municipalities
    
    stateMap["REGION"] = regions_list
    
    #If the region has more then 1 unit, attribute another id
    for i in regions:
        if(len(regions[i]["municipalities"])) > 1:
            if i in regions_memory:
                regionId = regions_memory[i]
            else:
                regionId = next_free_region[0]
                next_free_region[0] = next_free_region[0] + 1
                regions_memory[i] = regionId
            
            for j in regions[i]["municipalities"]:
                stateMap.loc[stateMap["CD_MUN"] == j, "REGION"] = regionId

    #Plot the map
    base = stateMap.plot(column="REGION",figsize=(10,10),edgecolor="#d4d4d4",categorical=True,cmap=generate_color_map(regions_memory))
    #Plot the average income
    for i in regions:
        base.annotate(locale.format('%.2f', regions[i]["AVERAGE_INCOME"], True), (regions[i]["CENTROID"].x,regions[i]["CENTROID"].y))

file_state_data = "../data/State_Roraima/Municipios_RR.xlsx"
nrows_file_state_data = 16
file_state_map_json = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.json"
object_name = "RR_Municipios_2022"
file_state_map_shapefile = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

### Prepare initial regions -- each unit represents a single region
regions = {}
id = 1
for i in municipalities:
    region = {}
    region["id"] = id
    region["municipalities"] = []
    region["municipalities"].append(i)
    region["AVERAGE_INCOME"] = municipalities[i]["AVERAGE_INCOME"]
    region["CENTROID"] = municipalities[i]["centroide"]
    region["NEIGHBORHOOD"] = []
    regions[id] = region
    municipalities[i]["REGION_ID"] = id
    id = id + 1
    
for i in regions:
    for j in regions[i]["municipalities"]:
        for k in municipalities[j]["vizinhos"]:
            regions[i]["NEIGHBORHOOD"].append(municipalities[k]["REGION_ID"])

NUM_OF_REGIONS = len(regions)

#Control variables, used to maintain the colors
regions_memory = {}
next_free_region = [1]

#Plot first map with initial regions and average income
plot_regions_map(stateMap, regions, regions_memory, next_free_region)

#Control variable, to print the edges of the resulting spanning tree
merge_map = []

NUM_OF_REGIONS = len(regions)
### Hierarchical Clustering - Merging most similar regions
while(NUM_OF_REGIONS > 1):
    #Find two neighbor regions with largest similarity
    pair = find_largest_similarity(regions)
    #Merge the regions
    merge_regions(regions, pair, municipalities, regions_memory)
    #Plot the map
    plot_regions_map(stateMap, regions, regions_memory, next_free_region)
    
    NUM_OF_REGIONS = NUM_OF_REGIONS - 1 
    
#Print the edges of resulting spannig tree
base = stateMap.plot(figsize=(10,10),edgecolor="#d4d4d4",cmap=ListedColormap(["#80c1ff"]))
for i in merge_map:
    r0 = i[0]
    r1 = i[1]
    plt.plot([municipalities[r0]["centroide"].x, municipalities[r1]["centroide"].x], [municipalities[r0]["centroide"].y, municipalities[r1]["centroide"].y], 'ro-')