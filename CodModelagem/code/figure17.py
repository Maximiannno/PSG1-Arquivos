from utils import workWithStateData, plotRegionalization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

file_state_data = "../data/State_Roraima/Municipios_RR.xlsx"
nrows_file_state_data = 16
file_state_map_json = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.json"
object_name = "RR_Municipios_2022"
file_state_map_shapefile = "../data/State_Roraima/RR_Municipios_2022/RR_Municipios_2022.shp"

### Load state data
stateMap, municipalities, mun_list, stateArea, statePop, arcs = workWithStateData(file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)

#Attribute binary ids to the units
ind = 0
for i in municipalities:
    municipalities[i]["ID"] = (2 ** ind)
    ind = ind + 1

#Plot the map with the ids
base = stateMap.plot(figsize=(10,10),edgecolor="#d4d4d4",cmap=ListedColormap(["#D9F3FF"]))
for i in municipalities:
    base.annotate(municipalities[i]["ID"], (municipalities[i]["centroide"].x,municipalities[i]["centroide"].y))

### Plot a regionalization with the ids
regionalization = [1,1,1,1,1,0,0,1,1,1,1,0,0,0,1]
base = plotRegionalization(stateMap, regionalization, showCentroids=False, showBoundary=True)
for i in municipalities:
    base.annotate(municipalities[i]["ID"], (municipalities[i]["centroide"].x,municipalities[i]["centroide"].y))
    
# Adicionando o quadrado no canto inferior esquerdo
# Definindo as coordenadas e o tamanho do quadrado
quadrado = Rectangle((-65, -1), 1.5, 1, color="white", ec="black", lw=1)

# Adicionando o quadrado ao gráfico
plt.gca().add_patch(quadrado)

# Adicionando o texto dentro do quadrado
plt.text(-64.25, -0.5, "(14432, 18335)\nor\n(18335, 14432)", fontsize=12, ha='center', va='center', color='black')

# Mostrar o gráfico com o quadrado e texto
plt.show()