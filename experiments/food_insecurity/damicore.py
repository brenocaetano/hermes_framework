#!/usr/bin/env python
# coding: utf-8

# In[15]:

import numpy as np
import pandas as pd
# import cupy as cp
# from tqdm import tqdm
import os
# from ncdlib import compute_ncd, available_compressors
import multiprocessing as mp

from multiprocessing import Pool
from skbio import DistanceMatrix
from skbio.tree import nj
import igraph as ig

from shutil import copyfile
import shutil as shutil

# bibliotecas privadas -  Breno Caetano
import utils
import dados_twomoons
import newick2AdjMatrix

import ete3
from ete3 import Tree, TreeStyle, NodeStyle, TextFace



def layout_123LLGC(node):
    numbers = set(range(len(tnewick)))
    numbers = [str(i) for i in numbers]
#     grupo = set(gruposS[0])
#     for grupo in grupos:
    for i,grupo in zip(range(len(gruposS)),gruposS):
        cor = cores[i]
        if node.name in grupo:
            node.img_style["size"] = 200
#             node.img_style['fgcolor'] = 'white'
            node.img_style['bgcolor'] = cor
    
    # rotulo0 = [str(i) for i in result0]


def compute_costs_vect(dist_matrix, n):
    r = cp.sum(dist_matrix, axis=1)
    R = cp.tile(r, (n, 1))
    costs = (n-2) * dist_matrix - R - R.T
    cp.fill_diagonal(costs, 0)
    return costs



def find_minimum(costs):
    min_idx = cp.argmin(costs)
    min_pos = (min_idx // costs.shape[0], min_idx % costs.shape[0])
    min_cost = costs[min_pos[0], min_pos[1]]
    return int(min_pos[0]), int(min_pos[1]), min_cost

def update_matrix(dist_matrix, i, j, n):
    new_row = (dist_matrix[i, :] + dist_matrix[j, :] - dist_matrix[i, j]) / 2
    new_row = cp.expand_dims(new_row, axis=0)

    indices = [x for x in range(n) if x != i and x != j]
    reduced_matrix = dist_matrix[indices, :][:, indices]

    reduced_matrix = cp.vstack((reduced_matrix, new_row[:, indices]))
    new_col = cp.append(new_row[0, indices], 0).reshape(-1, 1)
    reduced_matrix = cp.hstack((reduced_matrix, new_col))

    return reduced_matrix

def neighbor_joining_gpu(dist_matrix):
    n = dist_matrix.shape[0]
    dist_matrix = cp.array(dist_matrix)
    nodes = [str(i) for i in range(n)]
    limite = len(dist_matrix)
    count = 0
    # with tqdm(total=limite) as pbar:
    while n > 1:
        costs = compute_costs_vect(dist_matrix, n)
        i, j, min_cost = find_minimum(costs)

        dij = dist_matrix[i, j]
        ri = cp.sum(dist_matrix[i, :]) - dij
        rj = cp.sum(dist_matrix[j, :]) - dij
        u = (dij + ri - rj) / 2
        v = dij - u

        dist_matrix = update_matrix(dist_matrix, i, j, n)

        new_node = f"({nodes[i]}:{u:.2f},{nodes[j]}:{v:.2f})"
        nodes = [nodes[k] for k in range(n) if k != i and k != j] + [new_node]

        n -= 1
        if count % 500 == 0:
            print(f'Iteração {count}')

        count += 1

    return nodes[0]

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import itertools

# def calculate_q(params):
#     distances, i, j, n = params
#     return (i, j, (n-2) * distances[i][j] - np.sum(distances[i]) - np.sum(distances[j]))
#
# from concurrent.futures import ThreadPoolExecutor
#
# def find_min_pair(distances):
#     n = len(distances)
#     args = [(distances, i, j, n) for i in range(n) for j in range(i+1, n)]
#     num_cores = os.cpu_count()  # Obtém o número de núcleos da CPU
#
#     with ThreadPoolExecutor(max_workers=num_cores) as executor:
#         results = list(executor.map(calculate_q, args))
#     min_result = min(results, key=lambda x: x[2])
#     return min_result[0], min_result[1]


def calculate_q_batch(params_batch, distances, n):
    result_batch = []
    for params in params_batch:
        i, j = params
        q_value = (n - 2) * distances[i][j] - np.sum(distances[i]) - np.sum(distances[j])
        result_batch.append((i, j, q_value))
    return result_batch


def find_min_pair(distances):
    n = len(distances)
    num_cores = os.cpu_count()  # Assume que queremos utilizar todos os núcleos
    batch_size = max(1, (n * (n - 1) // 2) // num_cores)  # Dividindo as tarefas em lotes
    # batch_size = 100
    # Criação de todos os pares possíveis e divisão em lotes
    args = [(i, j) for i in range(n) for j in range(i + 1, n)]
    batches = [args[i:i + batch_size] for i in range(0, len(args), batch_size)]

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(lambda batch: calculate_q_batch(batch, distances, n), batches))

    # Achatar a lista de resultados e encontrar o par com a menor distância corrigida
    flat_results = [item for sublist in results for item in sublist]
    min_result = min(flat_results, key=lambda x: x[2])
    return min_result[0], min_result[1]


# Continuação do código com a função de atualização e construção da árvore.


def update_distances(distances, i, j):
    n = len(distances)
    new_row = [(distances[i][k] + distances[j][k] - distances[i][j]) / 2 for k in range(n) if k != i and k != j]
    new_distances = np.delete(distances, (i, j), axis=0)
    new_distances = np.delete(new_distances, (i, j), axis=1)
    new_distances = np.append(new_distances, np.array([new_row]).T, axis=1)
    new_row.append(0)
    new_distances = np.append(new_distances, [new_row], axis=0)
    return new_distances

def build_newick_tree(indices, names):
    def get_name(index):
        if isinstance(index, int):
            return names[index]
        else:
            return "(" + ",".join(get_name(subindex) for subindex in index) + ")"
    tree = get_name(indices[0]) + "," + get_name(indices[1])
    return "(" + tree + ");"

def neighbor_joining_parallel(distances, names):
    indices = list(range(len(names)))
    while len(indices) > 2:
        i, j = find_min_pair(distances)
        distances = update_distances(distances, i, j)
        # Cria um novo índice representando a fusão dos nós i e j
        new_index = [indices[i], indices[j]]
        indices[i] = new_index
        if i < j:
            indices.pop(j)
        else:
            indices.pop(j)
            indices.pop(i)

        print(len(indices))
        if (len(indices) % 1000 == 0):
            print(len(indices))
    return build_newick_tree(indices, names)



# In[]
def clusterizacaoHierarquica(matrizDistancia, nomesArquivos, metodo = 'nj'):
#     print('\nMETODO DE CLUSTERIZACAO HIERARQUICA\n')
    if metodo == 'nj':
#         print(matrizDistancia)
#         print(nomesArquivos)
        
        matrizDistancia = DistanceMatrix(matrizDistancia, nomesArquivos)
        tree = nj(matrizDistancia)
        # print('treeNJ %s' % tree)
        # print(tree.ascii_art())
        # print(type(tree))
        return tree, matrizDistancia

    elif metodo == 'nj-cuda':
        tree = neighbor_joining_gpu(matrizDistancia)
        return tree, matrizDistancia

    elif metodo == 'nj-parallel':
        tree = neighbor_joining_parallel(matrizDistancia, nomesArquivos)
        return tree, matrizDistancia





def estilos():
    visual_style = {}
    visual_style['vertex_size'] = (50)
    visual_style['bbox'] = (4000,4000)
    visual_style['margin'] = 100
    visual_style['vertex_label_family']  = 'Calibri'
    visual_style['vertex_label_size'] = '24'
#     print(visual_style)
    return visual_style


estilo = estilos()


# In[21]:
def deteccaoComunidade(grafo, metodo = 'fastgreedy', optimal_count = 0):
    com = ig.Graph()
    if metodo == 'fastgreedy':
        # estilo = lerParametros(diretorio = config)
        com = grafo.community_fastgreedy()
        if optimal_count != 0:
            com.optimal_count = optimal_count
        com = com.as_clustering()
        
    elif metodo == 'edge_betweenness':
        # estilo = lerParametros(diretorio = config)
        com = grafo.community_edge_betweenness()
        if optimal_count != 0:
            com.optimal_count = optimal_count
        com = com.as_clustering()
        
    elif metodo == 'optimal_modularity':
        # estilo = lerParametros(diretorio = config)
        com = grafo.community_optimal_modularity()
        if optimal_count != 0:
            com.optimal_count = optimal_count
            
    elif metodo == 'infomap':
        # estilo = lerParametros(diretorio = config)
        com = grafo.community_infomap()
        if optimal_count != 0:
            com.optimal_count = optimal_count
    
    elif metodo == 'label_propagation':
        # estilo = lerParametros(diretorio = config)
        com = grafo.community_label_propagation()
        if optimal_count != 0:
            com.optimal_count = optimal_count
            
    elif metodo == 'spinglass':
        # estilo = lerParametros(diretorio = config)
        com = grafo.community_spinglass()
        if optimal_count != 0:
            com.optimal_count = optimal_count
    
    elif metodo == 'walktrap':
        # estilo = lerParametros(diretorio = config)
        com  = grafo.community_walktrap()
        if optimal_count != 0:
            com.optimal_count = optimal_count
        com = com.as_clustering()
            
    elif metodo == 'multilevel':
        # estilo = lerParametros(diretorio = config)
        com = grafo.community_multilevel()
        if optimal_count != 0:
            com.optimal_count = optimal_count
            
    elif metodo == 'leading_eigenvector':
        # estilo = lerParametros(diretorio = config)
        com  = grafo.community_leading_eigenvector()
        if optimal_count != 0:
            com.optimal_count = optimal_count
            
        
    return com

# In [.]

def damicore(indir='/home/brenocaetano/teste1',
             outdir='',
             listaArquivos = ' ',
             optimal_count = ' ',
             qntTotalElementos = 12,
             algo_deteccao = 'fastgreedy', 
             matrizDistancia = ''):
    
    

    tamanhoMatriz = len(matrizDistancia)
    
    tree, mdGrafico = clusterizacaoHierarquica(matrizDistancia=matrizDistancia,
                                                   nomesArquivos=listaArquivos)
        
#     print(tree.ascii_art())
    grafo, newick = newick2AdjMatrix.main(str(tree)[:-2], indir + '.png')
    print('dentro da bib damicore, linha 168')
    print(newick)

#     grafo.vs['label'] = listaArquivos
#     grafo.vs['name'] = listaArquivos
    grafoDetCom = deteccaoComunidade(grafo, metodo = algo_deteccao, optimal_count = optimal_count, config = config)
    
    
    

    return grafo, grafoDetCom, tree, newick, matrizDistancia 

# In[]

def permutacaoMatrizAdjacencia(grafo, listaArquivos, numReamostragem):
    nome_nos = grafo.vs['name']
    nome_nos = np.array(nome_nos)
#     print(nome_nos)
#     print(listaArquivos)
    matadj = grafo.get_adjacency().data
    matadj = np.array(matadj)
    matrizesAdjReamostradas, listaArquivosReamostrados = utils.reamostragemMatrizAdj(matadj, numReamostragem, nome_nos)
    
    return matrizesAdjReamostradas, listaArquivosReamostrados
    
    
    # In[]
def geracaoEdetComaPartirdasMatrizesDeAdjReamostradas(listaArquivosReamostrados, matrizesAdjReamostradas, grafo, metodo = 'fastgreedy', optimal_count = ' '):
    
    grafosReamostradosAposNj = []
    grafoDetComReamostrados = []
    # print('metodo')
    # print(metodo)
    for i in range(len(matrizesAdjReamostradas)):
        novoG = ig.Graph.Adjacency(matrizesAdjReamostradas[i].tolist())
        novoG.to_undirected()
        novoG.vs['label'] = listaArquivosReamostrados[i]
        novoG.vs['name'] = listaArquivosReamostrados[i]
#         ig.plot(novoG, target = os.join.path(outdir, 'novoGrafo'+str(i))
        grafoDetCom = deteccaoComunidade(grafo, metodo = metodo, optimal_count = optimal_count)
        grafoDetComReamostrados.append(grafoDetCom)
        grafosReamostradosAposNj.append(novoG)
        
        
    
        
        
        
    return grafosReamostradosAposNj, grafoDetComReamostrados

# def damicoreComPermutacaoMatrizAdjComPeso(indir='/home/brenocaetano/teste1',outdir='',config='',listaArquivos = ' ',optimal_count = ' ', qntTotalElementos = 12, algo_deteccao = 'fastgreedy', matrizDistancia = '', qntMatrizAdj = 2, pesos = []):
    
    

#     tamanhoMatriz = len(matrizDistancia)
#     tree, mdGrafico = clusterizacaoHierarquica(matrizDistancia=matrizDistancia,
#                                                    nomesArquivos=listaArquivos)
        

#     grafo, newick = newick2AdjMatrix.main(str(tree)[:-2], indir + '.png')
    
    
    
# #     matrizesAdj = particionamentoEmFolhas(grafoDetCom, listaArquivos)
#     matrizesAdjReamostradas, listaArquivosReamostrados = permutacaoMatrizAdjacencia(grafo, listaArquivos, qntMatrizAdj)
    
    
    
    
#     grafosReamostradosAposNj,grafoDetComReamostrados=geracaoEdetComaPartirdasMatrizesDeAdjReamostradas(listaArquivosReamostrados,
#                                                                                                        matrizesAdjReamostradas,
#                                                                                                        grafo,
#                                                                                                        metodo = algo_deteccao,
#                                                                                                        optimal_count = optimal_count,
#                                                                                                        config = config)
    
    
    

    
#     return grafosReamostradosAposNj, grafoDetComReamostrados, tree, newick, matrizDistancia


#     damicoreComPermutacaoMatrizAdj
#   damicoreComPermutacaoMatrizAdj

# In[]

def damicoreComPermutacaoMatrizAdj(indir='/home/brenocaetano/teste1',
                                   outdir='',
                                   listaArquivos = ' ',
                                   optimal_count = ' ',
                                   qntTotalElementos = 12,
                                   algo_deteccao = 'fastgreedy',
                                   matrizDistancia = '',
                                   qntMatrizAdj = 2,
                                   metodo = 'nj'):
    
    

    tamanhoMatriz = len(matrizDistancia)
    # necessario usar a simetrizacao pois por problemas de arredondamento
    # do python, a matriz de distancia pode nao ser simetrica.
    # print(type(matrizDistancia))
    # matrizDistancia = utils.simetrizacao(matrizDistancia)
    matrizDistancia = 0.5 * (matrizDistancia + matrizDistancia.T)
    print('###################')
    print('\nCOMECOU O NJ\n')
    print('###################')
    tree, mdGrafico = clusterizacaoHierarquica(matrizDistancia=matrizDistancia,
                                                   nomesArquivos=listaArquivos,
                                               metodo = metodo)
        

    grafo, newick = newick2AdjMatrix.main(str(tree)[:-2], indir + '.png')
    # print('dentro da bib damicore, linha 261')
    # print(newick)
    
    
#     matrizesAdj = particionamentoEmFolhas(grafoDetCom, listaArquivos)
    matrizesAdjReamostradas, listaArquivosReamostrados = permutacaoMatrizAdjacencia(grafo, listaArquivos, qntMatrizAdj)
    grafosReamostradosAposNj,grafoDetComReamostrados=geracaoEdetComaPartirdasMatrizesDeAdjReamostradas(listaArquivosReamostrados,
                                                                                                       matrizesAdjReamostradas,
                                                                                                       grafo,
                                                                                                       metodo = algo_deteccao,
                                                                                                       optimal_count = optimal_count)
    
    
    

    
    return grafosReamostradosAposNj, grafoDetComReamostrados, tree, newick, matrizDistancia
# In[]


# In[]

def consenso(listaDeGrafosDetCom, listaDeGrafos, listaArquivosOriginais, metodo = 'helson'):
    
    if metodo == 'helson':
        listaArquivosStr = np.array([str(item) for item in listaArquivosOriginais])
        particionamentoSoDeFolhasTotal = []
        col = len(listaDeGrafos[0])
#       col é o numero de elementos em uma posicao da lista lista de grafos. 
#         print('listaDeGrafos')
#         print(listaDeGrafos)
#         print(len(listaDeGrafos))
#         print(len(listaDeGrafos[0]))
        for grafos_linha in range(len(listaDeGrafos)):
#             print('\ngrafos_linha %i' % grafos_linha)
            for grafos_col in range(col):
#                 print('grafos_col %i\n' % grafos_col)
                grafoEmFoco = listaDeGrafos[grafos_linha][grafos_col]
                grafoDetComEmFoco = listaDeGrafosDetCom[grafos_linha][grafos_col]
                
#               PRA CADA GRAFO
                particionamentoIntermediario = []
                for i in range(len(grafoDetComEmFoco.sizes())):
                    subgrafo = grafoDetComEmFoco.subgraph(i)
#                     print(subgrafo.vs['name'])

                    particionamentoSoDeFolhas_aux = []                    
#                   PRA CADA SUB-GRAFO
                    for no in subgrafo.vs['name']:
                        if no in listaArquivosStr:
                #           aqui são usados dois appends, para guargar o nó e a qual cluster ele pertence.
                            particionamentoSoDeFolhas_aux.append(no)
                #             particionamentoSoDeFolhas_aux.append(i)
#                     print('particionamentoSoDeFolhas_aux')
#                     print(particionamentoSoDeFolhas_aux)
                    particionamentoIntermediario.append(particionamentoSoDeFolhas_aux)
#                 print('particionamentoIntermediario')
#                 print(particionamentoIntermediario)
                    
                    
                particionamentoSoDeFolhasTotal.append(particionamentoIntermediario)
        
#         print('particionamento so de folhas total')
#         print(particionamentoSoDeFolhasTotal)
#         print('particionamento so de folhas total')
#         print(len(particionamentoSoDeFolhasTotal))
#         print(len(particionamentoSoDeFolhasTotal[0]))
# AGORA SIM É O CONSENSO. O TRECHO ACIMA ESTAVA RETIRANDO O NÓS INTERMEDIARIOS            
        consensoMatriz = np.zeros(len(listaArquivosOriginais)*len(listaArquivosOriginais)).reshape(len(listaArquivosOriginais),len(listaArquivosOriginais))
    
    
        matrizDf = pd.DataFrame(consensoMatriz, columns = listaArquivosStr, index = listaArquivosStr)
                                  
                                  
        for w in range(len(particionamentoSoDeFolhasTotal)):
            for z in range(len(particionamentoSoDeFolhasTotal[w])):
                particao = particionamentoSoDeFolhasTotal[w]                  
                for i in range(len(particao[z])+1):
                    for j in range(i+1, len(particao[z])):
            #             print('%i e %i' % (i,j))
                        matrizDf.loc[particao[z][i]][particao[z][j]] += 1
                        matrizDf.loc[particao[z][j]][particao[z][i]] += 1

            
    return particionamentoSoDeFolhasTotal, matrizDf
                

# def rbf_damicore(dadosZ, dadosZ_teste, y, y_teste, num_neuron, otimizador, numero_entradas, epocas, pesos = 'aleatorio'):
#     saida = []
    

#     print('num_neuron')
#     print(num_neuron)
#     print('otimizador')
#     print(otimizador)
    
#     if pesos == 'constante':
#         initializer = tf.keras.initializers.GlorotNormal(0)
#     else:
#         initializer = tf.keras.initializers.GlorotNormal()

#     model = tf.keras.models.Sequential()

#     if num_neuron ==1 :
#             # PARA USO SE FOR UTILIZAR 1 NEURONIOS NA CAMADA DE SAIDA - USANDO O ONEHOTENCODER
#         y1 = y.reshape(-1,1)
#         y1_teste = y_teste.reshape(-1,1)

#     else:
#             # PARA USO SE FOR UTILIZAR 2 NEURONIOS NA CAMADA DE SAIDA - USANDO O ONEHOTENCODER            
#         enc = OneHotEncoder()
#         y_fit = enc.fit(y.reshape(-1,1))
#         y2 = y_fit.transform(y.reshape(-1,1)).toarray()

#         y_fit_teste = enc.fit(y_teste.reshape(-1,1))
#         y2_teste = y_fit_teste.transform(y_teste.reshape(-1,1)).toarray()
        
#     model.add(tf.keras.layers.Dense(units=num_neuron, activation = 'linear', input_shape=(numero_entradas,),
#                                     kernel_initializer = initializer))
#             # model.add(tf.keras.layers.Dropout(0.2))
#     model.summary()
#     model.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

    
#     if num_neuron == 1:
#         history = model.fit(dadosZ,y1, epochs=epocas, verbose = 0)
#         test_loss, test_accuracy = model.evaluate(dadosZ_teste,y1_teste)
#     else:
#         history = model.fit(dadosZ,y2, epochs=epocas, verbose = 0)
#         test_loss, test_accuracy = model.evaluate(dadosZ_teste,y2_teste)


#     print('test_accuracy: {}'.format(test_accuracy))

#     print('test_loss: {}'.format(test_loss))

#     saida.append([num_neuron, otimizador, test_accuracy, history])
            
#     return saida
        
# In[]
# if __name__ == "__main__":
#     print('ok')
    
#     X3 = dados_twomoons.X3
#     y3 = dados_twomoons.y3
#     X, y = X3, y3
    
#     matrix_dist_euclidiana = utils.matrizDistancia(X, 'euclidiana')
#     # matrix_dist_euclidiana = ncd...
    
# # In['configuracao']    
#     indir = os.getcwd()
#     outdir_principal =   os.path.join(indir,'output')        
#     config = os.path.join(indir,'config')
    
#     outdir = os.path.join(outdir_principal, "experimentosKennedy")
    
#     pontos = X
#     n_samples = len(X)
#     pontos_str = range(n_samples)
#     pontos_str = np.array([str(item) for item in pontos_str])
    
#     numMatrizDistancia = 1
#     tamMatrizDistancia = len(matrix_dist_euclidiana)
#     elementos = (np.arange(tamMatrizDistancia))
    
#     numMatrizAdjacencia = [1]
    
#     elementosReamostrados = []
#     matrizesDeDistanciaReamostradas = []
#     sequenciaReamostragem = []
            
#     elementosReamostrados, matrizesDeDistanciaReamostradas, sequenciaReamostragem = utils.reamostragemMatrizDistancia(matrix_dist_euclidiana, numMatrizDistancia, pontos, n_samples, elementos)

# # In[EXECUCAO DAMICORE]
#     algo_deteccao = ['fastgreedy']
#     grafosReamostradosTotais = []
#     grafosDetComReamostradosTotais = []
#     for algo_detec in algo_deteccao:   
#         print(algo_detec)
#         numOptimal_count = ' '
#         grafo, grafoDetCom, tree, newick, md = damicoreComPermutacaoMatrizAdj(indir=indir,
#                                                                                         outdir=outdir_principal,
#                                                                                         config = config,
#                                                                                         optimal_count = numOptimal_count,
#                                                                                         qntTotalElementos = len(pontos_str),
#                                                                                         listaArquivos=sequenciaReamostragem[0],
#                                                                                         algo_deteccao = algo_detec,
#                                 #                                                       para cada matriz de distancia, tem-se uma sequencia dos pontos
#                                                                                         matrizDistancia = matrizesDeDistanciaReamostradas[0],
#                                                                                         qntMatrizAdj = numMatrizAdjacencia[0])
#         grafosReamostradosTotais.append(grafo)
#         grafosDetComReamostradosTotais.append(grafoDetCom)

# # In[]





# # In[OPCIONAL]
#     particionamentoFolhas, consensoTotal_interacaoDf = consenso(grafosDetComReamostradosTotais,
#                                                                           grafosReamostradosTotais, elementos, 'helson')
    
#     # consensoTotal_interacao = np.where(consensoTotal_interacaoDf >= len(algo_deteccao)/2,
#     #                                    consensoTotal_interacaoDf, 0)
#     print('corte {}'.format(numMatrizAdjacencia[0]*len(algo_deteccao)/2))
#     consensoTotal_interacao = np.where(consensoTotal_interacaoDf > numMatrizAdjacencia[0]*len(algo_deteccao)/2 ,
#                                         consensoTotal_interacaoDf, 0)
    
    
#     grafoConsenso = ig.Graph.Adjacency(consensoTotal_interacao.tolist())
    
#     grafoConsenso.to_undirected()
#     grafoConsenso.vs['label'] = consensoTotal_interacaoDf.columns.values.tolist()
#     grafoConsenso.vs['name'] = consensoTotal_interacaoDf.columns.values.tolist()
    
#     # comunidadesConsenso_com_optimal_count_automatico = grafoConsenso.community_fastgreedy().as_clustering()
#     comunidadesConsenso = grafoConsenso.community_fastgreedy().as_clustering()
    
    
#     # In[SALVANDO OS RESULTADOS]
#     if os.path.exists(outdir):
#         #         print('pasta existe')
#         utils.apagaConteudoDiretorio(indir = outdir)
#         #         print('deve ter sido apagada')
#     else:
#         os.makedirs(outdir)
    
#             #SALVA O FORMATO NEWICK JA RETIRANDO O CARACTERE DE NEWLINE
#     utils.salvaFormatoNewick(outdir, tree)
    
#             #SALVA OS CLUSTERS EM UM ARQUIVO SEM OS NÓS INTERMEDIARIOS, APENAS AS FOLHAS. 
#     destinoArqClusters = os.path.join(outdir,'clusters.out')
#     utils.salvaClustersSemNosIntermediarios(outdir, destinoArqClusters, comunidadesConsenso, pontos_str)
    
#     grupos  = utils.getGruposInfile(destinoArqClusters)
#             #PLOTA UM GRAFICO CONTENDO OS PONTOS E SEUS RESPECTIVOS GRUPOS
#     centroDosGrupos, varianciaDosGrupos, desvioPadraoDosGrupos = utils.getCentroDosGrupos(grupos, pontos, plot = False, annotate = False)    
    
    
#     # In[]    
    
#     newickS = ''
#     for i in range(len(newick)):
#         newickS += newick[i]
#     newickS += ';'
#     newickS
    
    
#     tnewick = Tree(newickS)
#     tnewick
    
#     # In[]
#     # cores = ['#1C1C1C','#FFFF00','green', 'red', 'blue', '#00FF00','#8B4513','#9370DB','#4B0082',' 	#BA55D3',
#     #          '#FF00FF','#FF1493','#800000','#FF0000','#B0E0E6','#FFA500','#FFA07A','#DC143C','#A020F0','#8B008B',
#     #         '#BC8F8F','#D2B48C','#F4A460']
    
    
#     # Chartreuse/verde - #7FFF00
#     # amarelo          - #FFFF00
#     # Aqua / Cyan      - #00FFFF
#     # LawnGreen        - #7CFC00 - muito parecido com chartreuse
#     # deeppink         - #FF1493
#     # Lime             - #00FF00
#     # SaddleBrown      - #8B4513
#     # MediumPurple     - #9370DB
#     # Indigo           - #4B0082
#     # MediumOrchid     - #BA55D3
#     # Fuchsia / Magenta- #FF00FF
#     # Orchid           - #DA70D6
#     # maroon           - #800000 -> #F0F8FF aliceblue
#     # red              - #FF0000
#     # PowderBlue       - #B0E0E6
#     # orange           - #FFA500
#     # LightSalmon      - #FFA07A
#     # Crimson          - #DC143C
#     # purple           - #A020F0 - > #4F4F4F (cinza)
#     # darkmangeta      - #8B008B
#     # RosyBrown        - #BC8F8F
#     # tan              - #D2B48C
#     cores = ['#008B8B','#FFFF00','#00FFFF', '#7CFC00', '#FF1493', '#00FF00','#8B4513','#9370DB','#4B0082','#BA55D3',
#              '#FF00FF','#DA70D6','#F0F8FF','#FF0000','#B0E0E6','#FFA500','#FFA07A','#DC143C','#4F4F4F','#8B008B',
#             '#BC8F8F','#D2B48C','#F4A460']#FF8C00
    
#     # pega os nome dos nós dos grupos e coloca no formado str. isso é necessario para usar o ETE
#     gruposS = []
#     for g in grupos:
#         grupoS = [str(no) for no in g]
#         gruposS.append(grupoS)
#     # gruposS
    
    
#     # In[]
#     # Basic tree style
#     ts = TreeStyle()
#     ts.layout_fn = layout_123LLGC
#     ts.show_leaf_name = True
#     ts.scale = 150
#     ts.mode = 'c'
#     ts.branch_vertical_margin = 10 # 10 pixels between adjacent branches
    
#     # # Add two text faces to different columns
#     # # tnewick.add_face(TextFace("hola "), column=0, position = "branch-right")
#     # # tnewick.add_face(TextFace("mundo!"), column=1, position = "branch-right")
#     tnewick.show(tree_style=ts)
    
#     # #     for i in rotulo0:
#     # #         cor = cores[i]
#     #     if node.name in rotulo0:
#     #         node.img_style['fgcolor'] = 'white'
#     #     else:
#     #         node.img_style['fgcolor'] = 'black'