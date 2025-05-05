import numpy
from numpy import sin, cos, arccos, pi
import geopandas
import pandas as pd
import json
import math
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = numpy.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= numpy.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = numpy.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)


def rad2deg(radians):
    degrees = radians * 180 / pi
    return degrees


def deg2rad(degrees):
    radians = degrees * pi / 180
    return radians


def getDistanceBetweenPointsNew(latitude1, longitude1, latitude2, longitude2, unit='miles'):

    if (latitude1 == latitude2 and longitude1 == longitude2):
        return 0

    theta = longitude1 - longitude2

    distance = 60 * 1.1515 * rad2deg(
        arccos(
            (sin(deg2rad(latitude1)) * sin(deg2rad(latitude2))) +
            (cos(deg2rad(latitude1)) * cos(deg2rad(latitude2)) * cos(deg2rad(theta)))
        )
    )

    if unit == 'miles':
        return numpy.round(distance, 5).item()
    if unit == 'kilometers':
        return numpy.round(distance * 1.609344, 5).item()
    

def workWithStateData(arquivo_dados_municipio, nrows_dados_municipio, arquivo_json_mapa, nome_objeto, arquivo_shapefile_mapa):
    
    # Ler o mapa no GeoPandas
    mapa = geopandas.read_file(arquivo_shapefile_mapa)
    mapa["centroid"] = mapa["geometry"].centroid
    
    # Ler dados populacionais dos municipios
    dados_mun = pd.read_excel(arquivo_dados_municipio, header=0,
                              index_col=1, skiprows=2, nrows=nrows_dados_municipio)
    
    # Ler dados dos mapas
    with open(arquivo_json_mapa, "r") as f:
        data = json.load(f)

    # Obter os valores de transformação (escala)
    scaleX = data["transform"]["scale"][0]
    scaleY = data["transform"]["scale"][1]
    transformX = data["transform"]["translate"][0]
    transformY = data["transform"]["translate"][1]

    # Carregar os tamanhos reais dos arcos --> Para posterior cálculo de perímetro
    geoArcs = data["arcs"]
    tamArcs = []

    for geoArc in geoArcs:
        geoPos = geoArc[0]
        xIni = geoPos[0]
        yIni = geoPos[1]
        xIniScale = (xIni * scaleX) + transformX
        yIniScale = (yIni * scaleY) + transformY
        distArc = {}

        distArc["arcos"] = []
        distArc["distancia"] = 0

        for j in range(1, len(geoArc)):
            geoPos1 = geoArc[j]
            x = xIni + geoPos1[0]
            y = yIni + geoPos1[1]
            xScale = (x * scaleX) + transformX
            yScale = (y * scaleY) + transformY

            # Cálculo de distância (real) entre os pontos
            # distancia = math.sqrt( math.pow((xScale - xIniScale),2) + math.pow((yScale-yIniScale),2) )
            distancia = getDistanceBetweenPointsNew(
                yIniScale, xIniScale, yScale, xScale, unit='kilometers')
            distancia = round(distancia, 5)

            props = {}
            props["distancia"] = distancia
            props["lat_ini"] = yIniScale
            props["long_ini"] = xIniScale
            props["lat_fim"] = yScale
            props["long_fim"] = xScale
            distArc["arcos"].append(props)
            distArc["distancia"] = distArc["distancia"] + distancia

            # Renovar as variáveis para a próxima iteração
            xIni = x
            yIni = y
            xIniScale = xScale
            yIniScale = yScale

        distArc["distancia"] = round(distArc["distancia"], 5)
        tamArcs.append(distArc)

    # Carregar as cidades do estado
    cidades = data["objects"][nome_objeto]["geometries"]
    municipios = {}
    mapaBord = {}
    passarDepois = {}
    arcos = []
    arestas = []
    areaEstado = 0
    habitantesEstado = 0
    divisasEstaduais = []
    lista_mun = []

    # Iterar pelos municípios processando os dados
    for cidade in cidades:

        # Carregar informacoes do municipio
        municipio = cidade["properties"]
        cod_mun_atu = municipio["CD_MUN"]
        municipio["vizinhos"] = {}
        municipio["arestas"] = {}
        municipio["NUM_HABITANTES"] = int(
            dados_mun.loc[int(municipio["CD_MUN"]), "População residente - pessoas [2022]"])
        municipio["NM_MUN"] = dados_mun.loc[int(
            municipio["CD_MUN"]), "Município [-]"]
        arcs = cidade["arcs"][0]
        municipio["PERIMETRO_DIVISA"] = 0
        municipio["AVERAGE_INCOME"] = dados_mun.loc[int(municipio["CD_MUN"]), "PIB per capita - R$ [2021]"]
        municipio["centroide"] = mapa.loc[mapa[mapa["CD_MUN"] == cod_mun_atu].index[0], "centroid"]
        
        #Totalizadores
        areaEstado = areaEstado + municipio["AREA_KM2"]
        habitantesEstado = habitantesEstado + municipio["NUM_HABITANTES"]

        if cidade["type"] == "MultiPolygon":
            arcs = []
            for i in range(len(cidade["arcs"])):
                arcs.extend(cidade["arcs"][i][0])

        # Passar pelos arcos que formam o desenho do município
        for border in arcs:
            if border < 0:
                b2 = (border + 1) * -1
                divisasEstaduais.remove(b2)
        
                if b2 in mapaBord:
                    cidFronteira = mapaBord[b2]
                    distancia = tamArcs[b2]["distancia"]
                    dist_linha = getDistanceBetweenPointsNew(
                        municipio["centroide"].y, municipio["centroide"].x, municipios[cidFronteira]["centroide"].y, municipios[cidFronteira]["centroide"].x, unit='kilometers')
                    dist_linha = round(dist_linha, 5)
                    obj = {"distancia": distancia,
                           "dist_linha": dist_linha}

                    municipio["vizinhos"][municipios[cidFronteira]
                                          ["CD_MUN"]] = obj
                    municipios[cidFronteira]["vizinhos"][cod_mun_atu] = obj

                    tuplaI = (cidFronteira, cod_mun_atu)
                    tuplaJ = (cod_mun_atu, cidFronteira)

                    if tuplaI not in arcos:
                        arcos.append(tuplaI)

                    if tuplaJ not in arcos:
                        arcos.append(tuplaJ)

                    if tuplaI not in arestas and tuplaJ not in arestas:
                        arestas.append(tuplaI)
                        #tamArestas[tuplaI] = distancia
                        
                    if tuplaI in arestas:
                        municipio["arestas"][tuplaI] = distancia
                        municipios[cidFronteira]["arestas"][tuplaI] = distancia
                        
                    if tuplaJ in arestas:
                        municipio["arestas"][tuplaJ] = distancia
                        municipios[cidFronteira]["arestas"][tuplaJ] = distancia

                else:
                    passarDepois[b2] = cod_mun_atu

            else:
                mapaBord[border] = cod_mun_atu
                divisasEstaduais.append(border)

        municipios[cod_mun_atu] = municipio
        lista_mun.append(cod_mun_atu)

    #Atribuir o perímetro de divisa de cada município
    for i in divisasEstaduais:
        munDiv = mapaBord[i]
        municipios[munDiv]["PERIMETRO_DIVISA"] = municipios[munDiv]["PERIMETRO_DIVISA"] + tamArcs[i]["distancia"]
        
    return mapa, municipios, lista_mun, areaEstado, habitantesEstado, arcos


def computeDistanceMatrix(municipios):
    
    distancia = {}

    #Calular a matriz completa de distâncias
    for i in municipios:
        for j in municipios:
            if(i==j):
                distancia[(i,j)] = 0
            else:
                dist_linha = getDistanceBetweenPointsNew(
                    municipios[i]["centroide"].y, municipios[i]["centroide"].x, municipios[j]["centroide"].y, municipios[j]["centroide"].x, unit='kilometers')
                dist_linha = round(dist_linha, 5)
                distancia[(i,j)] = dist_linha
                
    return distancia

def plotRegionalizationHessModel(x, mapa, municipios, showCentroids, showRegionCentroids, showBoundary):
    
    solucao = {}
    centros = {}
    list_centros = []
    contCentros = 0;
    
    for val in x:
        #print(f"{val} {x[val].x}")
        if(x[val].x == 1):
            if(val[1] in centros):
                solucao[val[0]] = centros[val[1]]
            else:
                centros[val[1]] = contCentros
                contCentros = contCentros + 1
                solucao[val[0]] = centros[val[1]]
                list_centros.append(val[1])
                
    #Plotar a solução
    regioes = []
    for i in municipios:
        regioes.append(solucao[i])
        
    mapa = mapa.copy()
    mapa.insert(2, "REGIAO", regioes, True)
    
    #Plotar o mapa
    base = mapa.plot(column="REGIAO", figsize=(10, 10), cmap="tab20", categorical=True)
    
    if showBoundary:
        #Acentuar as fronteiras
        base = mapa.boundary.plot(ax=base, color="black")
    
    if showCentroids:
        #Printar os centroides
        base = mapa.centroid.plot(ax=base, color="blue", markersize=7)
        
    if showRegionCentroids:
        #Printar os centroides das regiões
        base = mapa.loc[mapa["CD_MUN"].isin(list_centros)].centroid.plot(ax=base, color="red", markersize=14)
        
    return base

def findRegionsHessModel(x, municipios):
    
    solucao = {}
    centros = {}
    list_centros = []
    contCentros = 0;
    
    for val in x:
        #print(f"{val} {x[val].x}")
        if(x[val].x == 1):
            if(val[1] in centros):
                solucao[val[0]] = centros[val[1]]
            else:
                centros[val[1]] = contCentros
                contCentros = contCentros + 1
                solucao[val[0]] = centros[val[1]]
                list_centros.append(val[1])
                
    #Plotar a solução
    regioes = []
    for i in municipios:
        regioes.append(solucao[i])
        
    return regioes, list_centros
    

def plotRegionalizationHessModelMulticolors(x, mapa, municipios, showCentroids, showRegionCentroids, showBoundary, numColors):
    
    regioes, list_centros = findRegionsHessModel(x, municipios)
        
    mapa = mapa.copy()
    mapa.insert(2, "REGIAO", regioes, True)
    
    #Plotar o mapa
    base = mapa.plot(column="REGIAO", figsize=(10, 10), cmap=generate_colormap(numColors), categorical=True)
    
    if showBoundary:
        #Acentuar as fronteiras
        base = mapa.boundary.plot(ax=base, color="black")
    
    if showCentroids:
        #Printar os centroides
        base = mapa.centroid.plot(ax=base, color="blue", markersize=7)
        
    if showRegionCentroids:
        #Printar os centroides das regiões
        base = mapa.loc[mapa["CD_MUN"].isin(list_centros)].centroid.plot(ax=base, color="red", markersize=14)
        
    return base


def plotRegionalizationLabelingModel(x, w, mapa, municipios, showCentroids, showRegionCentroids, showBoundary):
    
    solucao = {}
    list_centros = []
    
    for val in x:
        #print(f"{val} {x[val].x}")
        if(x[val].x >= 0.99):
            solucao[val[0]] = val[1]
            
    for val in w:
        if(w[val].x == 1):
            list_centros.append(val[0])
    
    #Plotar a solução
    regioes = []
    for i in municipios:
        regioes.append(solucao[i])
        
    mapa = mapa.copy()
    mapa.insert(2, "REGIAO", regioes, True)
    
    #Plot map of regions
    base = mapa.plot(column="REGIAO", figsize=(10, 10), cmap="tab20", categorical=True)
   
    if showBoundary:
        #Acentuar as fronteiras
        base = mapa.boundary.plot(ax=base, color="black")
    
    if showCentroids:
        #Printar os centroides
        base = mapa.centroid.plot(ax=base, color="blue", markersize=7)
    
    if showRegionCentroids:
        #Printar os centroides
        base = mapa.loc[mapa["CD_MUN"].isin(list_centros)].centroid.plot(ax=base, color="red", markersize=14)
        
    return base


def plotRegionalization(mapa, regioes, showCentroids, showBoundary):
    
    mapa = mapa.copy()
    mapa.insert(2, "REGIAO", regioes, True)
    
    #Plot map of regions
    base = mapa.plot(column="REGIAO", figsize=(10, 10), cmap="tab20", categorical=True)
   
    if showBoundary:
        #Acentuar as fronteiras
        base = mapa.boundary.plot(ax=base, color="white")
    
    if showCentroids:
        #Printar os centroides
        base = mapa.centroid.plot(ax=base, color="blue", markersize=7)
            
    return base


def djikstraDistance(municipios, distancia):
    
    distancia_d = {}
    for orig in municipios:
        distancias_d = {}
        nao_visitados = []
        
        for dest in municipios:
            distancias_d[dest] = {}
            if orig == dest:
                distancias_d[dest]["dis"] = 0
                distancias_d[dest]["vertices"] = []
            else:
                distancias_d[dest]["dis"] = float('inf')
                distancias_d[dest]["vertices"] = []
            nao_visitados.append(dest)
            
        while(len(nao_visitados) > 0):
            
            #escolher o municipio não-visitado com a menor distancia
            choose_dest = "";
            dist_dst = float('inf')
            for dest in nao_visitados:
                if(distancias_d[dest]["dis"] < dist_dst):
                    choose_dest = dest
                    dist_dst = distancias_d[dest]["dis"]
            
            nao_visitados.remove(choose_dest)
            for viz in municipios[choose_dest]["vizinhos"]:
                if(distancias_d[viz]["dis"] > distancias_d[choose_dest]["dis"]+distancia[choose_dest,viz]):
                    distancias_d[viz]["dis"] = distancias_d[choose_dest]["dis"]+distancia[choose_dest,viz]
                    distancias_d[viz]["vertices"] = []
                    distancias_d[viz]["vertices"].extend(distancias_d[choose_dest]["vertices"])
                    distancias_d[viz]["vertices"].append(choose_dest)
        
        #Pós-processamento: retirar a origem dos nós da lista
        for dest in distancias_d:
            if(orig in distancias_d[dest]["vertices"]):
                distancias_d[dest]["vertices"].remove(orig)
        
        distancia_d[orig] = distancias_d
        
    return distancia_d