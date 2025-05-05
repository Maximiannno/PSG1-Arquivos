from utils import workWithStateData
from matplotlib.colors import ListedColormap
import locale


def get_map_colors():
    
    if MUN_IN_NO_REGION > 0:
        return ListedColormap(["white","#80ffa0","#80c1ff"])
    else:
        return ListedColormap(["#80ffa0","#80c1ff"])

def plot_regionalization_map(stateMap, regions, municipalities):
    region_list = [0] * len(stateMap.index)
    
    stateMap["REGION"] = region_list
    
    units = list(municipalities.keys())
    
    #Print the region area, in square kilometers
    for i in regions:
        for j in regions[i]["municipalities"]:
            units.remove(j)
            stateMap.loc[stateMap["CD_MUN"] == j, "REGION"] = regions[i]["id"]
    
    base = stateMap.plot(column="REGION",figsize=(10,10),edgecolor="#d4d4d4",categorical=True,cmap=get_map_colors())
    #For the units outside an region, print the area, in square kilometers
    for i in units:
        base.annotate(locale.format('%.2f', municipalities[i]["AREA_KM2"], True), (municipalities[i]["centroide"].x,municipalities[i]["centroide"].y))
    for i in regions:
        mun = regions[i]["municipalities"][0]
        base.annotate(locale.format('%.2f', regions[i]["area"], True), (municipalities[mun]["centroide"].x,municipalities[mun]["centroide"].y))

file_state_data = "../data/State_Roraima/Municipios_RR.xlsx"
nrows_file_state_data = 16
file_state_map_json = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.json"
object_name = "RR_Municipios_2022"
file_state_map_shapefile = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

#Initiate a dictionary of regions
MUN_IN_NO_REGION = 15
regions = {}

#Plot the state map, with territorial areas, with no regions
plot_regionalization_map(stateMap, regions, municipalities)

#Choose the most southern and most northern unit as roots
#lat is between -90 and +90
most_north = -91
most_south = 91

for i in municipalities:
    if(municipalities[i]["centroide"].y < most_south):
        most_south = municipalities[i]["centroide"].y
        most_south_unit = i
    if(municipalities[i]["centroide"].y > most_north):
        most_north = municipalities[i]["centroide"].y
        most_north_unit = i
    municipalities[i]["REGION_ID"] = 0
        
regions[1] = {}
regions[1]["id"] = 1
regions[1]["municipalities"] = [most_south_unit]
regions[1]["area"] = municipalities[most_south_unit]["AREA_KM2"]
municipalities[most_south_unit]["REGION_ID"] = 1

regions[2] = {}
regions[2]["id"] = 2
regions[2]["municipalities"] = [most_north_unit]
regions[2]["area"] = municipalities[most_north_unit]["AREA_KM2"]
municipalities[most_north_unit]["REGION_ID"] = 2

#Plot the state map, with territorial areas, with initial regions
plot_regionalization_map(stateMap, regions, municipalities)

MUN_IN_NO_REGION = len(municipalities) - 2

while MUN_IN_NO_REGION > 0:
    #Greedy strategy: choose the region with the minimum area to expand
    min_area = float("inf")
    for i in regions:
        if regions[i]["area"] < min_area:
            min_area = regions[i]["area"]
            min_area_region = i
            
    #Greedy strategy: chose the neighboring unit not assigned with the minimum area to expand
    min_area = float("inf")
    for i in regions[min_area_region]["municipalities"]:
        for j in municipalities[i]["vizinhos"]:
            if municipalities[j]["REGION_ID"] == 0:
                if municipalities[j]["AREA_KM2"] < min_area:
                    min_area = municipalities[j]["AREA_KM2"]
                    min_area_neigh = j
    
    #Update the region
    regions[min_area_region]["municipalities"].append(min_area_neigh)
    regions[min_area_region]["area"] = regions[min_area_region]["area"] + municipalities[min_area_neigh]["AREA_KM2"]
    
    #Update the municipality
    municipalities[min_area_neigh]["REGION_ID"] = min_area_region
    
    MUN_IN_NO_REGION = MUN_IN_NO_REGION - 1
    
    #Plot the map
    plot_regionalization_map(stateMap, regions, municipalities)