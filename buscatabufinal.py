import random
from collections import deque
import numpy as np
import os
import pprint
import heapq
import utils.utils as utils
import matplotlib.pyplot as plt
import geopandas as gpd
import json



#termos a serem mudados para busca mais/menos profunda:
#alternancia
#tamanho da lista tabu
#n de reinicios



#Carregar dados

base_dir = os.path.dirname(__file__)


''' RORAIMA 
file_state_data = os.path.join(base_dir, "data", "State_Roraima", "Municipios_RR.xlsx")
nrows_file_state_data = 16
file_state_map_json = os.path.join(base_dir, "data", "State_Roraima", "RR_Municipios_2022", "RR_Municipios_2022.json")
object_name = "RR_Municipios_2022"
file_state_map_shapefile = os.path.join(base_dir, "data", "State_Roraima", "RR_Municipios_2022", "RR_Municipios_2022.shp")
file_medical_procedures = os.path.join(base_dir, "data", "State_Roraima", "Procedimentos_RR.xlsx")
nrows_file_medical_procedures = 964
NUM_REGIONS = 2  '''

""" AMAZONAS 
file_state_data = os.path.join(base_dir, "data", "State_Amazonas", "Municipios_AM.xlsx")
nrows_file_state_data = 63
file_state_map_json = os.path.join(base_dir, "data", "State_Amazonas", "AM_Municipios_2022", "AM_Municipios_2022.json")
object_name = "AM_Municipios_2022"
file_state_map_shapefile = os.path.join(base_dir, "data", "State_Amazonas", "AM_Municipios_2022", "AM_Municipios_2022.shp")
file_medical_procedures = os.path.join(base_dir, "data", "State_Amazonas", "Procedimentos_AM.xlsx")
nrows_file_medical_procedures = 6890
NUM_REGIONS = 9  """ 

""" RIO GRANDE DO NORTE 

file_state_data = os.path.join(base_dir, "data", "State_RioGrandeDoNorte", "Municipios_RN.xlsx")
nrows_file_state_data = 168
file_state_map_json = os.path.join(base_dir, "data", "State_RioGrandeDoNorte", "RN_Municipios_2022", "RN_Municipios_2022.json")
object_name = "RN_Municipios_2022"
file_state_map_shapefile = os.path.join(base_dir, "data", "State_RioGrandeDoNorte", "RN_Municipios_2022", "RN_Municipios_2022.shp")
file_medical_procedures = os.path.join(base_dir, "data", "State_RioGrandeDoNorte", "Procedimentos_RN.xlsx")
nrows_file_medical_procedures = 5345
NUM_REGIONS = 8 """


""" MINAS GERAIS 

file_state_data = os.path.join(base_dir, "data", "State_MinasGerais", "Municipios_MG.xlsx")
nrows_file_state_data = 854
file_state_map_json = os.path.join(base_dir, "data", "State_MinasGerais", "MG_Municipios_2022", "MG_Municipios_20221.json")
object_name = "MG_Municipios_2022"
file_state_map_shapefile = os.path.join(base_dir, "data", "State_MinasGerais", "MG_Municipios_2022", "MG_Municipios_2022.shp")
file_medical_procedures = os.path.join(base_dir, "data", "State_MinasGerais", "Procedimentos_MG.xlsx")
nrows_file_medical_procedures = 58458
NUM_REGIONS = 89 """


""" SAO PAULO """

file_state_data = os.path.join(base_dir, "data", "State_SaoPaulo", "Municipios_SP.xlsx")
nrows_file_state_data = 646
file_state_map_json = os.path.join(base_dir, "data", "State_SaoPaulo", "SP_Municipios_2022", "SP_Municipios_2022.json")
object_name = "SP_Municipios_2022"
file_state_map_shapefile = os.path.join(base_dir, "data", "State_SaoPaulo", "SP_Municipios_2022", "SP_Municipios_2022.shp")
file_medical_procedures = os.path.join(base_dir, "data", "State_SaoPaulo", "Procedimentos_SP.xlsx")
nrows_file_medical_procedures = 58458
NUM_REGIONS = 62 


stateMap, municipalities, mun_list, stateArea, statePop, arcs = utils.workWithStateData(
    file_state_data, nrows_file_state_data, file_state_map_json, object_name, file_state_map_shapefile)
distances = utils.computeDistanceMatrix(municipalities)
utils.add_medical_procedures(municipalities, file_medical_procedures, nrows_file_medical_procedures)




#funções objetivo

def f1_homogeneity_inhabitants(graph, sol, num_regions, totalPop):
    units_list = list(graph.keys())
    pops = {i: 0 for i in range(num_regions)}
    for i in range(len(sol)):
        pops[sol[i]] += graph[units_list[i]]['NUM_HABITANTES']
    mean_pop = totalPop / num_regions
    desvio_total = sum(abs(pops[i] - mean_pop) for i in pops)
    return desvio_total / num_regions

def f2_variety_medical_procedures(graph, sol, num_regions):
    units_list = list(graph.keys())
    procedures = {i: set() for i in range(num_regions)}
    for i in range(len(sol)):
        for j in graph[units_list[i]]['procedimentos']:
            procedures[sol[i]].add(j)
    sum_procs_reg = sum(len(procedures[i]) for i in range(num_regions))
    return sum_procs_reg / num_regions

def f3_intra_regional_traveling_distance(graph, sol, num_regions, distances):
    city_codes = list(graph.keys())
    f3 = 0.0
    for region in range(num_regions):
        for i, city_i in enumerate(city_codes):
            if sol[i] != region:
                hi = graph[city_i]['NUM_HABITANTES']
                min_distance = float('inf')
                for j, city_j in enumerate(city_codes):
                    if sol[j] == region:
                        dij = distances.get((city_i, city_j), distances.get((city_j, city_i), None))
                        if dij is not None:
                            weighted_distance = hi * dij
                            min_distance = min(min_distance, weighted_distance)
                if min_distance != float('inf'):
                    f3 += min_distance
    return f3

def avaliar_multiobjetivo(sol):
    f1 = f1_homogeneity_inhabitants(municipalities, sol, NUM_REGIONS, statePop)
    f2 = f2_variety_medical_procedures(municipalities, sol, NUM_REGIONS)
    f3 = f3_intra_regional_traveling_distance(municipalities, sol, NUM_REGIONS, distances)
    return {"f1": f1, "f2": f2, "f3": f3}

def domina(a, b):
    melhor = False
    if a['f1'] > b['f1']: return False
    if a['f3'] > b['f3']: return False
    if a['f2'] < b['f2']: return False
    if a['f1'] < b['f1']: melhor = True
    if a['f2'] > b['f2']: melhor = True
    if a['f3'] < b['f3']: melhor = True
    return melhor


#geração inical (p-dispersion)




def encontrar_componentes_conexos():
    from collections import deque

    cidades = list(municipalities.keys())
    visitados = set()
    componentes = []

    for cidade in cidades:
        if cidade in visitados:
            continue
        componente = []
        fila = deque([cidade])
        while fila:
            atual = fila.popleft()
            if atual in visitados:
                continue
            visitados.add(atual)
            componente.append(atual)
            vizinhos = municipalities[atual]['vizinhos'].keys()
            for viz in vizinhos:
                if viz not in visitados:
                    fila.append(viz)
        componentes.append(componente)
    return componentes

def gerar_solucao_inicial():
    componentes = encontrar_componentes_conexos()

    seeds = []
    usados = set()

    print ('sol ini')

    #garante um seed por componente
    for comp in componentes:
        escolha = random.choice(comp)
        seeds.append(escolha)
        usados.add(escolha)

    #se faltarem regiões, sorteia seeds extras
    while len(seeds) < NUM_REGIONS:
        candidatos = list(set(municipalities.keys()) - usados)
        escolha = random.choice(candidatos)
        seeds.append(escolha)
        usados.add(escolha)

    atribuicao = {s: i for i, s in enumerate(seeds)}
    fronteiras = deque([(s, i) for i, s in enumerate(seeds)])
    visitados = set(seeds)

    while fronteiras:
        cidade_atual, regiao = fronteiras.popleft()
        vizinhos = list(municipalities[cidade_atual]['vizinhos'].keys())
        random.shuffle(vizinhos)

        for viz in vizinhos:
            if viz not in visitados:
                atribuicao[viz] = regiao
                fronteiras.append((viz, regiao))
                visitados.add(viz)

    #preenche regiões desconexas
    restantes = set(municipalities.keys()) - set(atribuicao.keys())
    for restante in restantes:
        #escolhe a região mais próxima
        min_dist = float("inf")
        melhor_regiao = random.randint(0, NUM_REGIONS-1)  # fallback
        for s in seeds:
            dist = utils.getDistanceBetweenPointsNew(
                municipalities[restante]['centroide'].y,
                municipalities[restante]['centroide'].x,
                municipalities[s]['centroide'].y,
                municipalities[s]['centroide'].x, unit='kilometers')
            if dist < min_dist:
                min_dist = dist
                melhor_regiao = atribuicao[s]
        atribuicao[restante] = melhor_regiao

    solucao = [atribuicao[m] for m in municipalities.keys()]

    #checa se é factível
    if utils.isFeasible({'atribuicao': solucao}, NUM_REGIONS, municipalities):

        print ("achou factivel")

        return solucao
    else:
        #se não for factível, tenta de novo


        print ('nao ahcou factivel')
        return gerar_solucao_inicial()


#erturbação
#troca municípios de regiões e checa se é factível, tentando perturbar a solução para descobrir melhores

def perturbar(atribuicao, intensidade=0.3):
    nova = atribuicao.copy()
    num_alteracoes = int(len(nova) * intensidade)
    for _ in range(num_alteracoes):
        i = random.randint(0, len(nova)-1)
        nova[i] = random.randint(0, NUM_REGIONS-1)
    if utils.isFeasible({'atribuicao': nova}, NUM_REGIONS, municipalities):
        return nova
    return atribuicao



#método de escape de platôs, chamado quando o algoritmo estagna


def sp_escape(atribuicao, max_tries=3):
    city_codes = list(municipalities.keys())
    for _ in range(max_tries):
        i = random.choice(range(len(atribuicao)))
        reg_i = atribuicao[i]
        visited = set()
        queue = [(0, i, [])]
        while queue:
            dist_acum, atual, caminho = heapq.heappop(queue)
            if atual in visited:
                continue
            visited.add(atual)
            caminho = caminho + [atual]
            vizinhos = municipalities[city_codes[atual]]['vizinhos'].keys()
            for viz in vizinhos:
                idx_viz = city_codes.index(viz)
                if atribuicao[idx_viz] != reg_i:
                    for node in caminho:
                        atribuicao[node] = atribuicao[idx_viz]
                    if utils.isFeasible({'atribuicao': atribuicao}, NUM_REGIONS, municipalities):
                        return atribuicao
                    else:
                        for node in caminho:
                            atribuicao[node] = reg_i
                else:
                    dist = municipalities[city_codes[atual]]['vizinhos'][viz]['dist_linha']
                    heapq.heappush(queue, (dist_acum + dist, idx_viz, caminho))
    return atribuicao


#busca tabu

def busca_tabu():
    pareto = []
    melhor_global = None
    reinicio = 0
    max_reinicios = 200

    while reinicio < max_reinicios:

        #gera a solução inicial ou perturba a melhor
        if reinicio == 0 or melhor_global is None:
            atribuicao = gerar_solucao_inicial()
        elif reinicio % 3 == 0:
            atribuicao = gerar_solucao_inicial()
        else:
            atribuicao = perturbar(melhor_global, intensidade=0.2)

        melhor_local = atribuicao
        melhor_raw = avaliar_multiobjetivo(atribuicao)
        estagnado = 0


        #altera entre compactar e balancear as regiões
        for alternancia in range(min(200, len(mun_list) * 2)):
            print(f"Alternância {alternancia+1}")
            atribuicao = otimizar_compactacao(atribuicao)
            atribuicao = otimizar_balanceamento(atribuicao, alpha=0.2)
            score = avaliar_multiobjetivo(atribuicao)

            if domina(score, melhor_raw):
                melhor_local = atribuicao
                melhor_raw = score
                estagnado = 0
            else:
                estagnado += 1

            if estagnado >= 5:
                atribuicao = sp_escape(atribuicao)
                estagnado = 0

        atribuicao_norm = utils.normalize_atribuicao(melhor_local)
        id_norm = utils.generate_id(atribuicao_norm)

        #checa se é factível
        if not utils.isFeasible({'atribuicao': atribuicao_norm}, NUM_REGIONS, municipalities):
            #não conta como reinício, só gera outra
            continue

        novo = {"atribuicao": atribuicao_norm, "fitness": melhor_raw, "id": id_norm}

        #checa se a solução achada é dominada pela fronteira de pareto e adiciona à fronteira se não
        nao_dominado = []
        dominado = False
        for s in pareto:
            if domina(novo['fitness'], s['fitness']):
                continue
            elif domina(s['fitness'], novo['fitness']):
                dominado = True
                break
            else:
                nao_dominado.append(s)
        if not dominado:
            nao_dominado.append(novo)
            pareto = nao_dominado

        melhor_global = melhor_local

        print(f"Reinício {reinicio+1}: f1={melhor_raw['f1']:.1f}, f2={melhor_raw['f2']:.1f}, f3={melhor_raw['f3']:.1f}")

        reinicio += 1

    return pareto


#balanceamento e compactação


#balanceamento 

def otimizar_balanceamento(atribuicao, alpha=0.2):
    tabu = deque(maxlen=int(len(mun_list) / 4))
    melhor = atribuicao.copy()
    melhor_eval = avaliar_multiobjetivo(melhor)
    city_codes = list(municipalities.keys())

    for _ in range(100):
        vizinhos = []

        indices_amostra = random.sample(range(len(atribuicao)), k=min(len(atribuicao), 10))

        for i in indices_amostra:
            atual = atribuicao[i]
            cidade_i = city_codes[i]
            vizinhos_i = municipalities[cidade_i]['vizinhos'].keys()

            if not any(atribuicao[city_codes.index(v)] != atual for v in vizinhos_i):
                continue

            regioes_candidatas = random.sample(range(NUM_REGIONS), k=min(NUM_REGIONS, 3))

            for nova_regiao in regioes_candidatas:
                if nova_regiao != atual:
                    nova = atribuicao.copy()
                    nova[i] = nova_regiao

                    if not utils.isFeasible({'atribuicao': nova}, NUM_REGIONS, municipalities):
                        continue

                    nova_eval = avaliar_multiobjetivo(nova)
                    # foco no f1 (população), mas mantendo f3 sob controle
                    if nova_eval["f1"] < melhor_eval["f1"] and nova_eval["f3"] <= (1+alpha)*melhor_eval["f3"] and tuple(nova) not in tabu:
                        vizinhos.append((nova_eval, nova))

        if not vizinhos:
            break

        vizinhos.sort(key=lambda x: x[0]["f1"])
        melhor_viz = vizinhos[0]
        atribuicao = melhor_viz[1]
        tabu.append(tuple(atribuicao))

        if domina(melhor_viz[0], melhor_eval):
            melhor = atribuicao
            melhor_eval = melhor_viz[0]

    return melhor

#compactação

def otimizar_compactacao(atribuicao):
    tabu = deque(maxlen=int(len(mun_list) / 4))
    melhor = atribuicao.copy()
    melhor_eval = avaliar_multiobjetivo(melhor)
    city_codes = list(municipalities.keys())

    for _ in range(100):
        vizinhos = []

        indices_amostra = random.sample(range(len(atribuicao)), k=min(len(atribuicao), 10))

        for i in indices_amostra:
            atual = atribuicao[i]
            cidade_i = city_codes[i]
            vizinhos_i = municipalities[cidade_i]['vizinhos'].keys()

            if not any(atribuicao[city_codes.index(v)] != atual for v in vizinhos_i):
                continue

            regioes_candidatas = random.sample(range(NUM_REGIONS), k=min(NUM_REGIONS, 3))

            for nova_regiao in regioes_candidatas:
                if nova_regiao != atual:
                    nova = atribuicao.copy()
                    nova[i] = nova_regiao

                    if not utils.isFeasible({'atribuicao': nova}, NUM_REGIONS, municipalities):
                        continue

                    nova_eval = avaliar_multiobjetivo(nova)
                    #foco total em f3 (compactação), tolerando variações de f1
                    if nova_eval["f3"] < melhor_eval["f3"] and tuple(nova) not in tabu:
                        vizinhos.append((nova_eval, nova))

        if not vizinhos:
            break

        vizinhos.sort(key=lambda x: x[0]["f3"])
        melhor_viz = vizinhos[0]
        atribuicao = melhor_viz[1]
        tabu.append(tuple(atribuicao))

        if domina(melhor_viz[0], melhor_eval):
            melhor = atribuicao
            melhor_eval = melhor_viz[0]

    return melhor


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def salvar_solucoes_pareto(pareto_solucoes, nome_estado):
    os.makedirs("resultados", exist_ok=True)
    caminho_arquivo = os.path.join("resultados", f"{nome_estado}_resultados.json")
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        json.dump(pareto_solucoes, f, indent=4, ensure_ascii=False, cls=NpEncoder)
    print(f"Soluções de Pareto salvas em: {caminho_arquivo}")



#execucao

pareto = busca_tabu()

#limpeza
pareto = utils.remove_symmetric_solutions(pareto)
pareto = utils.remove_equal_solutions(pareto)

print("\nPareto Final:")
for sol in pareto:
    output = {
        "atribuicao": sol["atribuicao"],
        "contiguo": True,
        "fitness": {
            "f1": round(sol["fitness"]["f1"], 4),
            "f2": round(sol["fitness"]["f2"], 4),
            "f3": round(sol["fitness"]["f3"], 4)
        },
        "id": sol["id"]
    }
    pprint.pprint(output, sort_dicts=False)

salvar_solucoes_pareto(pareto, 'spcerto')

#visualiza
sol = pareto[0]
atribuicao = sol["atribuicao"]
stateMap['grupo'] = 0
mun_keys = list(municipalities.keys())
for idx, cd_mun in enumerate(mun_keys):
    stateMap.loc[stateMap['CD_MUN'] == cd_mun, 'grupo'] = atribuicao[idx]
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
stateMap.plot(column='grupo', cmap='tab20', legend=True, edgecolor='black', ax=ax)
plt.title("Partição obtida pela busca tabu")
plt.show()


