"""
This code contains some functions used in various approaches
"""
import numpy
from numpy import sin, cos, arccos, pi
import geopandas
import pandas as pd
import json
import random

"""
This functions returns the distance, in kilometers, between two coordinates
"""
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

"""
This function loads a graph of a territory
"""
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
    seq_atu = 0

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
        municipio["seq"] = seq_atu
        
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
        seq_atu = seq_atu + 1

    #Atribuir o perímetro de divisa de cada município
    for i in divisasEstaduais:
        munDiv = mapaBord[i]
        municipios[munDiv]["PERIMETRO_DIVISA"] = municipios[munDiv]["PERIMETRO_DIVISA"] + tamArcs[i]["distancia"]
        
    return mapa, municipios, lista_mun, areaEstado, habitantesEstado, arcos

"""
This function adds medical procedures to state data
"""
def add_medical_procedures(graph, file_medical_procedures, nrows_file):
    # Read medical_procedure_files
    med_proc = pd.read_excel(file_medical_procedures, header=0, nrows=nrows_file)
    med_proc['MUNIC_MOV'] = med_proc['MUNIC_MOV'].astype(str)
    
    for i in graph:
        graph[i]["procedimentos"] = med_proc.loc[med_proc['MUNIC_MOV']==i[:6], 'PROC_REA'].tolist()

"""
This function checks the feasibility of a solution, and returns the constraint violation number
"""
def checkFeasibility(solution, num_regions, graph):
    
    #Test if there's an empty region
    empty_regions = list(range(num_regions))
    k = 0
    
    while len(empty_regions) > 0 and k < len(solution["atribuicao"]):
        if(solution["atribuicao"][k] in empty_regions):
            empty_regions.remove(solution["atribuicao"][k])
        k = k+1
    if len(empty_regions) > 0:
        return False, len(empty_regions)

    NUM_UNITS = len(graph)
    i = 1
    q = []
    genX = [0 for ik in range(NUM_UNITS)]
    
    for indCid in graph:
        
        cid = graph[indCid]
        ind = cid["seq"]
        if(genX[ind] != 0):
            continue
        
        genX[ind] = i;
        q.append(cid)
        
        while(len(q) > 0):
            cidU = q.pop(0)
            vizU = graph[cidU["CD_MUN"]]["vizinhos"]
            
            for cidV2 in vizU:
                cidV = graph[cidV2]
                if(genX[cidV["seq"]] != 0):
                    continue
                
                if(solution["atribuicao"][cidU["seq"]] == solution["atribuicao"][cidV["seq"]]):
                    genX[cidV["seq"]] = genX[cidU["seq"]]
                    q.append(cidV)
                    
        i = i + 1
        
    if((i-1) <= num_regions):
        return True, 0
    else:
        return False, (i-1-num_regions)


"""
This function tests if a solution is feasible
"""
def isFeasible(solution, num_regions, graph):
    is_feasible, num = checkFeasibility(solution, num_regions, graph)
    return is_feasible

"""
This function computes a distance matrix between all units
"""
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

"""
This function evaluates if solution a is dominated by solution b
list types is an dict of types of functions (min for minimization and max for maximization)
"""
def is_dominated_by(a, b, list_types):
    result = []
    
    for fun in list_types:
        fun_type = list_types[fun]
        
        if fun_type == "min":
            if b["fitness"][fun] < a["fitness"][fun]:
                result.append(1)
            elif b["fitness"][fun] == a["fitness"][fun]:
                result.append(0)
            else:
                result.append(-1)
        
        if fun_type == "max":
            if b["fitness"][fun] > a["fitness"][fun]:
                result.append(1)
            elif b["fitness"][fun] == a["fitness"][fun]:
                result.append(0)
            else:
                result.append(-1)
    
    there_is_a_better = False
    for i in result:
        
        if i == -1:
            return False
        
        if i == 1:
            there_is_a_better = True
            
    return there_is_a_better

"""
This function mount a dominance ranking of solutions
"""
def dominance_ranking_old(solutions, list_fun):
    
    non_ranked_solutions = list(range(len(solutions)))
    current_level = 1
    dominance_ranking = [0] * len(solutions)
    
    while(len(non_ranked_solutions) > 0):
        
        for i in range(len(solutions)):
            #If the solution is not in a front
            if(dominance_ranking[i] == 0):
                sol_i = solutions[i]
                better_sol = False
                for j in range(len(solutions)):
                    if(dominance_ranking[j] == 0):
                        sol_j = solutions[j]
                        if is_dominated_by(sol_i, sol_j, list_fun):
                            better_sol = True
                            break
                
                if not better_sol:
                    dominance_ranking[i] = current_level
                    non_ranked_solutions.remove(i)
        
        current_level = current_level + 1
        
    return dominance_ranking

def dominance_ranking(solutions, list_fun):
    non_ranked_solutions = list(range(len(solutions)))
    current_level = 1
    dominance_ranking = [0] * len(solutions)

    while non_ranked_solutions:
        current_front = []
        
        for i in non_ranked_solutions:
            sol_i = solutions[i]
            dominated = False

            for j in non_ranked_solutions:
                if i == j:
                    continue
                sol_j = solutions[j]
                if is_dominated_by(sol_i, sol_j, list_fun):
                    dominated = True
                    break

            if not dominated:
                current_front.append(i)

        for idx in current_front:
            dominance_ranking[idx] = current_level
            non_ranked_solutions.remove(idx)

        current_level += 1

    return dominance_ranking

"""
This function calculate the crowding distances for all solutions
"""
def calculate_crowding_distance(solutions, dominance_rank, fitness_keys):
    num_solutions = len(solutions)
    crowding_distances = [0.0] * num_solutions

    # Agrupa soluções por nível de dominância
    fronts = {}
    for i, rank in enumerate(dominance_rank):
        fronts.setdefault(rank, []).append(i)

    for front in fronts.values():
        if len(front) <= 2:
            for i in front:
                crowding_distances[i] = float('inf')
            continue

        for key in fitness_keys:
            # Ordena os índices do front pelo valor da função objetivo atual
            sorted_front = sorted(front, key=lambda i: solutions[i]["fitness"][key])
            f_min = solutions[sorted_front[0]]["fitness"][key]
            f_max = solutions[sorted_front[-1]]["fitness"][key]
            range_f = f_max - f_min if f_max != f_min else 1e-9

            # Borda recebe infinito
            crowding_distances[sorted_front[0]] = float('inf')
            crowding_distances[sorted_front[-1]] = float('inf')

            # Interiores recebem a soma das distâncias normalizadas
            for j in range(1, len(sorted_front) - 1):
                prev_f = solutions[sorted_front[j - 1]]["fitness"][key]
                next_f = solutions[sorted_front[j + 1]]["fitness"][key]
                dist = (next_f - prev_f) / range_f
                if crowding_distances[sorted_front[j]] != float('inf'):
                    crowding_distances[sorted_front[j]] += dist

    return crowding_distances

"""
This function makes the crowded tournament selection operator
"""
def crowded_binary_tournament(sol_a, sol_b, sol_ranking, diversity_ranking):
    rank_a = sol_ranking[sol_a]
    rank_b = sol_ranking[sol_b]
    diversity_a = diversity_ranking[sol_a]
    diversity_b = diversity_ranking[sol_b]

    if rank_a < rank_b:
        return sol_a
    elif rank_b < rank_a:
        return sol_b
    else:  # rank igual → decide por diversidade
        if diversity_a > diversity_b:
            return sol_a
        elif diversity_b > diversity_a:
            return sol_b
        else:
            # Empate total: sorteia um dos dois
            return random.choice([sol_a, sol_b])
        
"""
This function removes equal solutions from a population
"""
def remove_equal_solutions(population):
    curr_ids = []
    pop_ret = []
    
    for i in population:
        if i["id"] not in curr_ids:
            pop_ret.append(i)
            curr_ids.append(i["id"])
            
    return pop_ret

"""
This function guarantees a normal atribution without symmetric solutions
"""
def normalize_atribuicao(atribuicao):
    """
    Renomeia os grupos conforme ordem de primeira aparição.
    Ex: [2, 2, 0, 0, 1, 1] → [0, 0, 1, 1, 2, 2]
    """
    mapa = {}
    proximo_grupo = 0
    atribuicao_normalizada = []

    for g in atribuicao:
        if g not in mapa:
            mapa[g] = proximo_grupo
            proximo_grupo += 1
        atribuicao_normalizada.append(mapa[g])
    
    return atribuicao_normalizada

"""
This function generates an unique id from an genotype
"""
def generate_id(atribuicao):
    """Gera um ID em string binária a partir da atribuição."""
    return ''.join(str(g) for g in atribuicao)

"""
This function removes symmetric solutions from a population
"""
def remove_symmetric_solutions(population):
    """
    Remove soluções simétricas de uma população para k grupos (k ≥ 2).
    Mantém apenas uma versão de cada particionamento, independentemente do rótulo dos grupos.
    """
    seen = set()
    unique_population = []

    for sol in population:
        atrib = sol['atribuicao']
        atrib_norm = normalize_atribuicao(atrib)
        id_norm = generate_id(atrib_norm)

        if id_norm not in seen:
            seen.add(id_norm)

            sol_normalizada = sol.copy()
            sol_normalizada['atribuicao'] = atrib_norm
            sol_normalizada['id'] = id_norm

            unique_population.append(sol_normalizada)

    return unique_population