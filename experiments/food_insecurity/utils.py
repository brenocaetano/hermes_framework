#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
# from sklearn.utils.graph import graph_shortest_path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import shortest_path
import collections
from random import randint

import utils # explicitly import the utils module

import tempfile
from joblib import Parallel, delayed
import subprocess
# comment if dont using google colab
# from google.colab import files 
import tempfile
import zipfile
import Levenshtein
from itertools import combinations_with_replacement


from scipy.sparse import csr_matrix, csgraph

from skbio import DistanceMatrix
from skbio.tree import nj
from skbio import TreeNode

from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from Bio import Phylo

import tqdm
from tqdm import tqdm

from ete3 import Tree


# Função para converter a árvore em um grafo do igraph
def tree_to_igraph(tree):
    edges = []
    node_counter = 0
    clade_to_id = {}
    clade_names = {}
    weights = []
    node_map = {}

    for clade in tree.traverse("levelorder"):
        # for clade in tree.find_clades(order="level"):
        if clade not in clade_to_id:
            clade_name = clade.name if clade.name else f"node_{node_counter}"
            clade_to_id[clade] = node_counter
            clade_names[node_counter] = clade_name
            node_counter += 1
        for child in clade.children:
            # for child in clade.clades:
            if child not in clade_to_id:
                child_name = child.name if child.name else f"node_{node_counter}"
                clade_to_id[child] = node_counter
                clade_names[node_counter] = child_name
                node_counter += 1
            edges.append((clade_to_id[clade], clade_to_id[child]))

            weights.append(1 / child.dist if child.dist is not None and child.dist > 0.0 else 0.0)
            # weights.append(1 / child.dist if child.dist is not None and child.dist != 0.0 else 0.0)

            # try:
            #     weights.append(child.branch_length)
            # except:
            #     weights.append(0.0)

    # print("Edges:", edges)  # Adiciona instrução de depuração
    # print("Clade Names:", clade_names)  # Adiciona instrução de depuração
    g = ig.Graph(edges=edges)
    g.vs["name"] = [clade_names[i] for i in range(len(clade_names))]
    g.vs["label"] = [clade_names[i] for i in range(len(clade_names))]
    g.es["weights"] = weights
    return g

def damicore(X, y, distancias, n_neighbors=6, algo_deteccao='fastgreedy', resolution=1):
# def damicore(X, distancias, n_neighbors, algo_deteccao, resolution=1):
    # COMO AS DISTANCIAS JA ESTAO CALCULADAS, O PARAMETRO N_NEIGHBOR É DESNECESSARIO
    # Filograma

    names = [str(i) for i in range(X.shape[0])]

    matriz = distancias

    # Extrair a triangular inferior e a diagonal
    matrix = []
    for i in range(matriz.shape[0]):
        linha = []
        for j in range(i + 1):
            linha.append(matriz[i, j])
        matrix.append(linha)
    # Crie uma matriz de distância no formato aceito pelo Biopython
    dm = _DistanceMatrix(names, matrix)
    # print(dm)

    # Construa a árvore usando o algoritmo Neighbor Joining
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    # # Exiba a árvore
    # Phylo.draw(tree)

    # Salve a árvore no formato Newick
    Phylo.write(tree, os.path.join(os.getcwd(),"nj_tree.newick"), "newick")

    from ete3 import Tree, TreeStyle

    # Ler o arquivo Newick
    tree = Tree(os.path.join(os.getcwd(),"nj_tree.newick"), format=1)

    # Configurar o estilo da árvore (opcional)
    ts = TreeStyle()
    ts.show_leaf_name = True  # Mostrar os nomes das folhas
    ts.show_branch_length = True  # Mostrar os comprimentos dos ramos
    ts.show_branch_support = True  # Mostrar os valores de suporte dos ramos

    # Plotar a árvore
    # tree.show(tree_style=ts)

    # Detecção de comunidades e Carregar a árvore Newick
    # tree = Phylo.read("nj_tree.newick", "newick")

    # Converter a árvore em um grafo do igraph
    g = tree_to_igraph(tree)

    # Aplicar o algoritmo de detecção de comunidades (Louvain, neste caso)
    # communities = g.community_multilevel(weights=weights)

    # Aplicar o algoritmo de detecção de comunidades (fastgreedy, neste caso)
    if algo_deteccao == 'louvain':
        communities = g.community_multilevel(weights=g.es['weights'], resolution=resolution)
    elif algo_deteccao == 'fastgreedy':
        communities = g.community_fastgreedy(weights=g.es['weights']).as_clustering()

    # Obter a associação de cada vértice
    membership = communities.membership

    # Mostrar os resultados
    # print("Comunidades detectadas:", communities)
    # print("Associação dos vértices:", membership)

    grupos = []
    grupos_all_nodes = []
    subgraphs = []
    for sub in communities.subgraphs():
        subgraphs.append(sub.vs['name'])
        # pode ocorrer de haver uma comunidade formada apenas por nós internos.
        subgraph_only_leafs = [int(elem) for elem in sub.vs["name"] if not elem.startswith("Inner")]
        subgraph_all_nodes = [elem for elem in sub.vs["name"]]
        grupos_all_nodes.append(subgraph_all_nodes)
        if subgraph_only_leafs != []:
            grupos.append(subgraph_only_leafs)

    return grupos, grupos_all_nodes, tree

def community_mapping(original_membership, g):
    original_member = original_membership.copy()
    # define the map of elements belong to one communities
    # obtendo os nós de cada comunidade antes da agregação

    # obtem os nomes de todos os nós no grafo
    nos_no_grafo = g.vs['name'].copy()

    # Filtrando os elementos que não começam com 'No' ou seja, não são elementos reais da base de dados.
    posicoes = [i for i, elemento in enumerate(nos_no_grafo) if elemento.startswith('No')]

    # Remover os nós internos do dendrograma nas posições encontradas, iterando em ordem decrescente
    for posicao in sorted(posicoes, reverse=True):
        del nos_no_grafo[posicao]
        del original_member[posicao]

    # Dicionário para mapear comunidades contraídas a vértices originais
    community_map = {}

    # Preencher o dicionário com os vértices originais correspondentes a cada comunidade
    for vertex_index, community_id in enumerate(original_member):
        if community_id not in community_map:
            community_map[community_id] = []
        community_map[community_id].append(vertex_index)

    # Agora você pode imprimir ou processar o dicionário 'community_map'
    # Cada chave é um ID de comunidade contraída, e os valores são os vértices originais que a formam
    for community_id, vertices in community_map.items():
        print(f"Comunidade {community_id}: {vertices}")

    return community_map

# In[imprimindo a agregacao]
def aggregation_of_communities(membership, grafo):
    g = grafo.copy()
    # Contrair o grafo baseado nas comunidades
    # Importante: contract_vertices modifica o grafo original
    g.contract_vertices(membership, combine_attrs="ignore")
    g.simplify(multiple=True, loops=True)

    # Verificar e definir o atributo 'name' se não estiver definido
    if 'name' not in g.vs.attributes():
        g.vs['name'] = [str(i) for i in range(len(g.vs))]

    return g

# In[]
def community_detection_aggregation(g_agg):
    communities_agg = g_agg.community_fastgreedy()
    return communities_agg
# In[obtendo os elementos com maior centralidade nas comunidades]


def get_elementos_maior_centralidade_agg(membership_agg, community_map, X, grafo, type='degree_centrality'):
    community_map_agg = {}
    grupos_agg = []
    # mapeando o membership de cada amostra de uma community of community (agg)
    for comm_of_comm in list(set(membership_agg)):
        comm_agg = np.where(np.array(membership_agg) == comm_of_comm)
        elements = []
        for comm in comm_agg[0]:
            elements += community_map[comm]
        community_map_agg[comm_of_comm] = elements
        grupos_agg.append(elements)

    if type =='communities_centers':
        #obtendo o centro das comunidades na agregação

        centroDasAgregacoes = []

        for comm in range(len(community_map_agg)):
            centroDasAgregacoes.append(X[community_map_agg[comm]].mean(axis=0))

        # obtendo as amostras proximas ao centro das comunidades das agregações
        elementos_prox_centros_agg = get_elementos_prox_centroides(grupos_agg, centroDasAgregacoes, X)

        return elementos_prox_centros_agg

    elif type =='degree_centrality':
        # Calculando a centralidade de grau
        degree_centrality = grafo.degree()

        # Inicializando uma lista para armazenar o nó mais central de cada comunidade
        central_nodes = []

        # Iterando sobre cada comunidade para encontrar o nó mais central
        for community in grupos_agg:
            print(community)
            print(degree_centrality)
            community_centralities = [degree_centrality[node] for node in community]
            central_node = community[community_centralities.index(max(community_centralities))]
            central_nodes.append(central_node)

        return central_nodes


from ete3 import Tree


def get_closest_to_root_in_subtrees(newick_str, grupos):
    tree = Tree(newick_str, format=1)
    names_in_sub_arvores = []

    for num, grupo in enumerate(grupos):
        # Conjunto de nós para encontrar o ancestral comum
        conjunto_de_nos = grupo

        # Encontrando os nós na árvore
        nodes = [tree.search_nodes(name=str(name))[0] for name in conjunto_de_nos]

        # Encontrando o ancestral comum mais recente
        ancestral_comum = tree.get_common_ancestor(nodes)

        # Salvando a árvore em um arquivo PDF
        output_file = f"tree_{num}.pdf"
        ancestral_comum.render(output_file, w=183, units="mm")

        # Obtendo as sub-árvores que derivam do ancestral comum
        sub_arvores = ancestral_comum.get_children()

        for sub_arvore in sub_arvores:
            # Obtendo os nomes dos descendentes da sub-árvore que fazem parte da comunidade
            descendant_names = [int(leaf) for leaf in sub_arvore.get_leaf_names() if int(leaf) in conjunto_de_nos]

            if descendant_names:
                # Encontrando o nó mais próximo da raiz dentro dos descendentes
                closest_to_root = min(descendant_names, key=lambda leaf: tree.get_distance(ancestral_comum,
                                                                                           tree.search_nodes(
                                                                                               name=str(leaf))[0]))
                names_in_sub_arvores.append(closest_to_root)

    return names_in_sub_arvores


def get_central_names_of_clades(newick_str, grupos):
    # este metodo obtém os elementos que formam a comunidade distribuidos nos diversos ramos a partir do ancestral comum
    tree = Tree(newick_str, format=1)
    samples_to_labeled = []
    names_in_sub_arvores = []
    # all_descendant_names = []
    for num, grupo in enumerate(grupos):

        # Conjunto de nós para encontrar o ancestral comum
        conjunto_de_nos = grupo

        # Encontrando os nós na árvore
        nodes = [tree.search_nodes(name=str(name))[0] for name in conjunto_de_nos]

        # Encontrando o ancestral comum mais recente
        ancestral_comum = tree.get_common_ancestor(nodes)
        # print(f"qnts filhos tem o ancestral comum? {len(ancestral_comum.get_children())}")

        # Exibindo o ancestral comum
        # print(f"Ancestral comum mais recente para {conjunto_de_nos}:")

        # Salvando a árvore em um arquivo PDF
        output_file = f"tree_{num}.pdf"
        ancestral_comum.render(output_file, w=183, units="mm")

        # Obtendo as sub-árvores que derivam do ancestral comum
        sub_arvores = ancestral_comum.get_children()

        # print("\nSub-árvores que derivam do ancestral comum:")
        for i, sub_arvore in enumerate(sub_arvores, start=1):
            # print(f"Sub-árvore {i}:")
            # print(sub_arvore)

            # print(f'conjunto {conjunto_de_nos}')
            # Obtendo os nomes dos descendentes da sub-árvore que fazem parte da comunidade
            descendant_names = [int(leaf) for leaf in sub_arvore.get_leaf_names() if int(leaf) in conjunto_de_nos]
            # print(f'descententes {descendant_names}')
            # pode ocorrer do grupo ter apenas 2 elementos e o ancestral comum ter 3 filhos. assim, um [] seria introdu
            # zido e causaria erro
            if descendant_names != []:
                names_in_sub_arvores.append(descendant_names)


            # print(f"Nomes dos descendentes da Sub-árvore {i}: {descendant_names}")
            # print()

    return names_in_sub_arvores


# Função para remover nós que sejam folhas e comecem com 'Node'
def remove_leaves_starting_with_node(tree, prefix='node'):
    # Percorre todos os nós da árvore
    for node in tree.iter_leaves():
        # Verifica se o nome começa com 'Node'
        if node.name.startswith(prefix):
            print(f"Removendo folha: {node.name}")
            node.delete()

    return tree


# Função para iterativamente remover a raiz se tiver apenas um filho que começa com 'Node'
def iteratively_remove_root_if_single_child(tree, prefix='node'):
    while len(tree.children) == 1 and tree.children[0].name.startswith(prefix):
        child = tree.children[0]
        print(f"Removendo raiz e promovendo {child.name} a nova raiz.")
        # Substitui a árvore pela subárvore do filho
        tree = child.detach()  # Detach retorna a subárvore a partir do filho

    tree.dist = 0.0
    return tree


from ete3 import Tree, TreeStyle, NodeStyle, TextFace

# from ete3.coretype.tree import TreeStyle
# from ete3 import Tree, NodeStyle, TextFace

    # ... your other code ...

def extract_subtree(tree, target_node_name):
    """
    Extrai a subárvore com a raiz no nó especificado por target_node_name.
    """
    target_node = tree & target_node_name
    if not target_node:
        raise ValueError(f"Nó {target_node_name} não encontrado na árvore.")
    subtree = target_node.detach()
    return subtree


def filter_subtree(subtree, community_nodes):
    """
    Remove folhas ou nós internos da subárvore que não estão na lista community_nodes,
    mas mantém nós internos necessários.
    """
    community_nodes_set = set(community_nodes)
    for node in subtree.traverse(strategy='postorder'):
        # print(f'node {node.name}')
        if node.name not in community_nodes_set:
            # print(f'removendo {node.name}')
            node.delete()

    return subtree.copy()


def find_median_node(distance_dict):
    # Ordenar os nós de acordo com suas distâncias
    sorted_nodes = sorted(distance_dict.items(), key=lambda x: x[1])

    # Calcular o índice da mediana
    n = len(sorted_nodes)
    median_index = (n - 1) // 2  # Índice do nó com a mediana

    # Retornar o nó cuja distância é a mediana
    median_node, median_distance = sorted_nodes[median_index]
    return median_node, median_distance

# apaga um grupo de folhas de uma arvore
def del_group_of_leaves(tree, group):
    rotulos = [str(item) for item in group]
    for leaf in tree.iter_leaves():
        if leaf.name in rotulos:
            leaf.delete(preserve_branch_length=True)
    return tree

# Função para calcular as distâncias até a raiz
def get_distances_to_root(tree, community_analisada, topology_only = False):
    distances = {}
    for leaf in tree.iter_leaves():
        if leaf.name in community_analisada:
            distances[leaf.name] = tree.get_distance(leaf, tree, topology_only)
    return distances

def get_middle_of_filogram(grupos, tree, topology_only=False):
    node_keys = []
    for i, grupo in enumerate(grupos):
        grupo = list(map(str, grupo))
        distances_to_root = get_distances_to_root(tree, grupo, topology_only)
        if len(distances_to_root) != 0:
            # print(distances_to_root)
            median_node = find_median_node(distances_to_root)
            node_keys.append(median_node)

    node_keys = [int(node) for node, _ in node_keys]
    return node_keys


# Função para encontrar o nó mais distante do grupo, baseado na topologia
def find_farthest_node(distance_dict):
    # Ordenar os nós de acordo com suas distâncias em ordem decrescente
    sorted_nodes = sorted(distance_dict.items(), key=lambda x: x[1], reverse=True)

    # O nó mais distante será o primeiro da lista ordenada
    farthest_node, farthest_distance = sorted_nodes[0]
    return farthest_node, farthest_distance

# Função para obter o nó mais distante baseado na topologia da árvore
def get_farthest_node_in_tree(grupos, tree, topology_only = False):
    node_keys = []
    for i, grupo in enumerate(grupos):
        grupo = list(map(str, grupo))  # Certifique-se de que os nomes estão em formato de string
        distances_to_root = get_distances_to_root(tree, grupo, True)  # Usar a topologia
        farthest_node = find_farthest_node(distances_to_root)
        node_keys.append(farthest_node)
    node_keys = [int(node) for node, _ in node_keys]
    return node_keys


def chooseSamplesToLabelBackup(grupos, X, tree, newick_str, type):
    # 'communities_centers' returns the central sample from all community
    if type == 'communities_centers':
        centroDosGrupos, varianciaDosGrupos, desvioPadraoDosGrupos = utils.getCentroDosGrupos(grupos, X, plot=False,
                                                                                              annotate=False, dim1=0,
                                                                                              dim2=0)
        return get_elementos_prox_centroides(grupos, centroDosGrupos, X)
    # 'nearestRoot' returns the  nearest sample of root
    elif type == 'nearestRoot':
        return get_amostra_prox_raiz(grupos, tree)
    # 'clades_centers' returns the samples that are central of each sub-tree from community
    elif type == 'clades_centers':
        # names_in_ramos representa uma lista de lista onde cada elemento representa os pontos de uma comunidade distri-
        # buidos atraves dos ramos que partem do ancestral comum dos elementos da comunidade.
        names_in_ramos = get_central_names_of_clades(newick_str, grupos)
        print(names_in_ramos)
        centroDosGrupos, varianciaDosGrupos, desvioPadraoDosGrupos = getCentroDosGrupos(names_in_ramos, X, plot=False,
                                                                                              annotate=False,
                                                                                              dim1=0,dim2=0)
        return get_elementos_prox_centroides(names_in_ramos, centroDosGrupos, X)
    elif type == 'clades_nearestRoot':
        return get_closest_to_root_in_subtrees(newick_str, grupos)
    elif type == 'fartherFromRoot':
        return get_amostra_mais_distante_da_raiz(grupos, tree)
    elif type == 'node_in_filogram_center':
        return get_middle_of_filogram(grupos, tree, topology_only=False)
    elif type == 'fartherFromRootCofenetic':
        return get_farthest_node_in_tree(grupos, tree, topology_only=True)
    elif type == 'node_in_filogram_centerCofenetic':
        return get_middle_of_filogram(grupos, tree, topology_only=True)



def chooseSamplesToLabel(grupos, X, tree, newick_str, type):
    # 'communities_centers' returns the central sample from all community
    if type == 'communities_centers':
        centroDosGrupos, varianciaDosGrupos, desvioPadraoDosGrupos = utils.getCentroDosGrupos(grupos, X, plot=False,
                                                                                              annotate=False, dim1=0,
                                                                                              dim2=0)
        return get_elementos_prox_centroides(grupos, centroDosGrupos, X)
    # 'nearestRoot' returns the  nearest sample of root
    elif type == 'nearestRoot':
        return get_amostra_prox_raiz(grupos, tree)
    # 'clades_centers' returns the samples that are central of each sub-tree from community
    elif type == 'clades_centers':
        # names_in_ramos representa uma lista de lista onde cada elemento representa os pontos de uma comunidade distri-
        # buidos atraves dos ramos que partem do ancestral comum dos elementos da comunidade.
        names_in_ramos = get_central_names_of_clades(newick_str, grupos)
        print(names_in_ramos)
        centroDosGrupos, varianciaDosGrupos, desvioPadraoDosGrupos = getCentroDosGrupos(names_in_ramos, X, plot=False,
                                                                                              annotate=False,
                                                                                              dim1=0,dim2=0)
        return get_elementos_prox_centroides(names_in_ramos, centroDosGrupos, X)
    elif type == 'clades_nearestRoot':
        return get_closest_to_root_in_subtrees(newick_str, grupos)
    elif type == 'fartherFromRoot':
        return get_amostra_mais_distante_da_raiz(grupos, tree)
    elif type == 'NodeInFilogramCenter':
        return get_middle_of_filogram(grupos, tree, topology_only=False)
    elif type == 'fartherFromRootCofenetic':
        return get_farthest_node_in_tree(grupos, tree, topology_only=True)
    elif type == 'NodeInFilogramCenterCofenetic':
        return get_middle_of_filogram(grupos, tree, topology_only=True)


def get_amostra_prox_raiz(grupos, tree):
    # retorna todos os nos de um grupo que seja mais proximo da raiz, ou seja, retorna a raiz da subarvore onde a comu-
    # nidade esta inserida.
    # Supondo que a raiz seja o nó ancestral da árvore inteira
    raiz = tree.get_tree_root()
    elementos_mais_proximos = []

    for grupo in grupos:
        menor_distancia = float('inf')
        folha_mais_proxima = None

        for folha_nome in grupo:
            # print((folha_nome))
            folha = tree.search_nodes(name=str(folha_nome))[0]
            distancia = tree.get_distance(raiz, folha)

            if distancia < menor_distancia:
                menor_distancia = distancia
                folha_mais_proxima = folha_nome

        elementos_mais_proximos.append(folha_mais_proxima)

    return elementos_mais_proximos

def get_amostra_mais_distante_da_raiz(grupos, tree):
    # Supondo que a raiz seja o nó ancestral da árvore inteira
    raiz = tree.get_tree_root()
    elementos_mais_distantes = []

    for grupo in grupos:
        maior_distancia = -float('inf')  # Iniciar com -infinito para encontrar o máximo
        folha_mais_distante = None

        for folha_nome in grupo:
            folha = tree.search_nodes(name=str(folha_nome))[0]
            distancia = tree.get_distance(raiz, folha)

            if distancia > maior_distancia:
                maior_distancia = distancia
                folha_mais_distante = folha_nome

        elementos_mais_distantes.append(folha_mais_distante)

    return elementos_mais_distantes


def get_elementos_prox_centroides(grupos, centroids, X):
#     esta funcao obtem os elementos de cada grupo que estao mais proximos do centro do grupo
    index_min_dist = []
    for grupo, centro in zip(grupos, centroids):
        min_dist = []
        for ele in grupo:
            min_dist.append(euclidean_distances(X[ele].reshape(1,-1), centro.reshape(1,-1)))
        index_min_dist.append(grupo[min_dist.index(min(min_dist))])

    return index_min_dist

def get_elementos_from_communities(grupos, centroids, X, grafo, comunidades,
                                   membership=None, type='communities_centers', aggregation = False,
                                   community_map=None):
    if aggregation == False:
        if type == 'communities_centers':
            return get_elementos_prox_centroides(grupos, centroids, X)
        elif type == 'degree_centrality':
            # Calculando a centralidade de grau
            degree_centrality = grafo.degree()

            # Inicializando uma lista para armazenar o nó mais central de cada comunidade
            central_nodes = []

            # Iterando sobre cada comunidade para encontrar o nó mais central
            for community in comunidades:
                community_centralities = [degree_centrality[node] for node in community]
                central_node = community[community_centralities.index(max(community_centralities))]
                central_nodes.append(central_node)

            return central_nodes
    else:
        return get_elementos_maior_centralidade_agg(membership, community_map, X, grafo, type=type)
        # if type == 'communities_centers':
        #     return get_elementos_maior_centralidade_agg(membership_agg, community_map, X, grafo, type=type)
        #     # return get_elementos_prox_centroides_agg(membership_agg, community_map, X)
        # elif type == 'degree_centrality':
        #     # # Calculando a centralidade de grau
        #     # degree_centrality = grafoConseso.degree()
        #     #
        #     # # Inicializando uma lista para armazenar o nó mais central de cada comunidade
        #     # central_nodes = []
        #     #
        #     # # Iterando sobre cada comunidade para encontrar o nó mais central
        #     # for community in comunidadesConsenso:
        #     #     community_centralities = [degree_centrality[node] for node in community]
        #     #     central_node = community[community_centralities.index(max(community_centralities))]
        #     #     central_nodes.append(central_node)
        #
        #     return get_elementos_maior_centralidade_agg(membership_agg, community_map, X, grafo, type=type)


def retira_outliers_in_Test(outliers, labels, test_idx):
    out = []
    label_ele = []
    test_list = test_idx
#   informa os outliers que estao no conjunto de teste.
    for i in range(len(outliers)):
        if outliers[i] in test_list:
            print('\n ####################\n Outlier {} faz parte do conjunto de teste. Retirado da lista outliers!\n\n'.format(outliers[i]))
        else:
            out.append(outliers[i])
            label_ele.append(labels[i])
    
    
    return out, label_ele

def verifica_outliers_in_test(outliers, test_idx, train_idx, label_ind, unlab_ind):
# faz a verificação se há outliers no conjunto de teste, se sim, uma nova amostra do conjunto de treinamento é escolhida
# para fazer parte do conjunto de teste
    
    out = outliers.copy()
    teste_ind = test_idx.copy()
    train_ind = train_idx.copy()
    labeled = np.array(label_ind.index.copy())
    unlabeled = np.array(unlab_ind.index.copy())
    
    for elemento in out:
        flag = True
        if elemento in teste_ind:
#           identifica o outlier que esta no conjunto de teste, encontra sua posicao, faz sua retirada e o coloca no
#           conjunto de treinamento
            ind_ele = np.where(teste_ind == elemento)

            teste_ind = np.delete(teste_ind, ind_ele)
            train_ind = np.append(train_ind, elemento)
            
#           faz-se a escolha de um novo elemento que nao seja outlier para substituir o elemento retirado do conj. de teste
            
            while flag == True:
                idx_novo_ele = np.random.randint(len(train_ind))
                novo_elemento = train_ind[idx_novo_ele]
                if novo_elemento not in out:
                    
                    if novo_elemento in labeled:
#                       só quero se for do conjunto unlabeled.  
                        continue            
                    elif novo_elemento in unlabeled:
                        teste_ind = np.append(teste_ind, novo_elemento)
                        train_ind = np.delete(train_ind, idx_novo_ele)
                        
                        idx_unlabeled = np.where(unlabeled == novo_elemento)
                        unlabeled = np.delete(unlabeled, idx_unlabeled)
                        unlabeled = np.append(unlabeled, elemento)
                        
                        flag = False            
    
    return teste_ind, train_ind, np.sort(labeled), np.sort(unlabeled), test_idx.copy(), train_idx.copy(), np.array(label_ind.index.copy()), np.array(unlab_ind.index.copy())

def defineOutliersAposCluster(outliers, grupos, qntClusters):
    out = []


    outliers = np.array(outliers)
    grupos = np.array(grupos)
    for clus in range(qntClusters):
#         print(clus)
        amostrasClus = np.where(grupos == clus)
#         print(amostrasClus)
        outs = np.array(outliers)[amostrasClus]
#         print(outs)
        outAleatorio = np.random.randint(len(amostrasClus[0]))
#         print(outAleatorio)
        out.append(outs[outAleatorio])

    return out

def novasPermutacoes(opt_k, train_index, y):
#     faz novas permutacoes do label_idx, caso as amostras iniciais sejam, TODAS, da mesma classe.
    while (True):
        randpermu = np.random.permutation(train_index)[:opt_k]
        if (len(np.unique(y[randpermu])) > 1):
            return randpermu


def getOutliers(nos_mais_longe, nos_mais_proximo, lenArrayRepeticoes, viz):

    nos_mais_longeViz = nos_mais_longe[nos_mais_longe['vizinho'] == viz]
    counterLonge = collections.Counter(nos_mais_longeViz['ponto+longe'])

    repeticoesLonge = np.ones(lenArrayRepeticoes)
    repeticoesLongebase = np.zeros(lenArrayRepeticoes)

    nos_mais_proximoViz = nos_mais_proximo[nos_mais_proximo['vizinho'] == viz]
    counterProx = collections.Counter(nos_mais_proximoViz['ponto+proximo'])


    repeticoesProx = np.ones(lenArrayRepeticoes)
    repeticoesProxbase = np.zeros(lenArrayRepeticoes)


    for key in counterLonge.keys():
        repeticoesLonge[int(key)] = counterLonge.get(key)
        repeticoesLongebase[int(key)] = counterLonge.get(key)

    for key in counterProx.keys():
        repeticoesProx[int(key)] = counterProx.get(key)
        repeticoesProxbase[int(key)] = counterProx.get(key)


        # CONSTRUINDO O DATAFRAME
    repDf = pd.DataFrame((repeticoesLongebase - repeticoesProxbase), columns = ['rep'])
    repDf['repeticoesLongeBase'] = repeticoesLongebase
    repDf['repeticoesProxBase'] = repeticoesProxbase
    repDf['repeticoesLonProx'] = repeticoesLongebase + repeticoesProxbase
    repDf['rep_abs'] = abs(repeticoesLongebase - repeticoesProxbase)
    repDf['rep_semZeros'] = repDf['rep']
    repDf['rep_semZeros'] = repDf['rep_semZeros'].replace([0.0], np.nan)
    repDf['repeticoesLonProx_semZeros'] = repDf['repeticoesLonProx']
    repDf['repeticoesLonProx_semZeros'] = repDf['repeticoesLonProx_semZeros'].replace([0.0], np.nan)
    repDf


    #CALCULANDO OS OUTLIERS INFERIORES E SUPERIORES PARA REP SEM ZEROS - whis 1.5
    outliersSupIdx15, outliersSupRepValues15, outliersInfIdx15, outliersInfRepValues15 = outliersSupInf(repDf['rep_semZeros'], 0.25, 0.75, whis = 1.5)
    #CALCULANDO OS OUTLIERS INFERIORES E SUPERIORES PARA REP SEM ZEROS - whis 1
    outliersSupIdx10, outliersSupRepValues10, outliersInfIdx10, outliersInfRepValues10 = outliersSupInf(repDf['rep_semZeros'], 0.25, 0.75, whis=1)
    #CALCULANDO OS OUTLIERS INFERIORES E SUPERIORES PARA REP SEM ZEROS - whis 0.8
    outliersSupIdx08, outliersSupRepValues08, outliersInfIdx08, outliersInfRepValues08 = outliersSupInf(repDf['rep_semZeros'], 0.25, 0.75, whis=0.8)
    
    return outliersSupIdx15, outliersInfIdx15, outliersSupIdx10, outliersInfIdx10, outliersSupIdx08, outliersInfIdx08

def outliersSupInf(df, quartil1, quartil3, whis = 1.5):
    q1, q3 = df.quantile([quartil1, quartil3]).values
#     print(q1,q3)
    iqr = q3 - q1
#     print(iqr)

    lower_bound = q1 - (whis * iqr) 
    upper_bound = q3 + (whis * iqr) 
#     print(lower_bound, upper_bound)
    
    repeticoesAcimaUpperBound = df > upper_bound
    repeticoesAcimaUpperBound = repeticoesAcimaUpperBound[repeticoesAcimaUpperBound == True]
    repeticoesAbaixoLowerBound = df < lower_bound
    repeticoesAbaixoLowerBound = repeticoesAbaixoLowerBound[repeticoesAbaixoLowerBound == True]
    repeticoesAcimaUpperBound, repeticoesAbaixoLowerBound

    outliersSupIdx = repeticoesAcimaUpperBound.index.to_list()
    outliersSupRepValues = abs(df[outliersSupIdx].values)
    outliersInfIdx = repeticoesAbaixoLowerBound.index.to_list()
    outliersInfRepValues = abs(df[outliersInfIdx].values)
    
    return outliersSupIdx, outliersSupRepValues, outliersInfIdx, outliersInfRepValues

def contaRepeticoes(nos_Df, viz, tipo, lenArrayRepeticoes):
    # faz a contagem de vezes que uma amostra aparece no dataframe
    # assim sabemos qnts vezes uma amostra ocorreu como mais proxima ou
    # mais distante da raiz
    nos_mais_proximoViz = nos_Df[nos_Df['vizinho'] == viz]

    counter = collections.Counter(nos_mais_proximoViz['ponto+'+tipo])

    repeticoes = np.zeros(lenArrayRepeticoes)

    for key in counter.keys():
        repeticoes[int(key)] = counter.get(key)
    
    return repeticoes


def get_nos_mais(file, diretorioTrabalho, base, tipo ='proximo', executarTudo = False):
    
    #esta funcao obtem os pontos mais proximos ou distante da raiz para um conjunto
    #de arvores no formato newick
    arquivos = os.listdir(diretorioTrabalho)
    arquivos = [arq for arq in arquivos if arq.endswith('newick.out')]
    if (not os.path.exists(file)) or executarTudo:
        nos_mais = pd.DataFrame([], columns=['amostragem', 'vizinho', 'ponto+'+tipo])
        vizinho = 1
        for arq in tqdm(arquivos):
        #     print(arq)
            amostragem = arq.split(sep='_')[2]
            vizinho = arq.split(sep='_')[3]
            arq = os.path.join(diretorioTrabalho, arq)
        #     print(arq, amostragem, vizinho)
            tree = Tree(arq)
            
            if tipo == 'proximo':
                closest_leaf = tree.get_closest_leaf()
                name = closest_leaf[0].name
            elif tipo == 'longe':
                farthest_leaf = tree.get_farthest_leaf()
                name = farthest_leaf[0].name
            
            ponto_mais = name
        #     print(ponto_mais_proximo)

            ponto_aux = pd.DataFrame([amostragem,vizinho,ponto_mais], index=['amostragem', 'vizinho', 'ponto+'+tipo])
        #     print(ponto_aux)
            nos_mais = pd.concat([nos_mais,ponto_aux.transpose()], axis=0, ignore_index= True)

        nos_mais[['amostragem', 'vizinho']] = nos_mais[['amostragem', 'vizinho']].astype(int)
        # print('nos_mais_'+ tipo + base + '.csv')
        nos_mais.to_csv(file, sep=';', decimal = ',', index= False)

    else:
        nos_mais = pd.read_csv(file, sep=';', decimal=',')
    
    return nos_mais, arquivos



def salvaResults(nomeDestino, results, num_splits, num_queries):
    dados = []
    dados_aux = []
    for i in range(num_splits):
        dados_aux.append(results[i].initial_point)
    
            # salva a primeira lista que tem o resultado para os pontos iniciais de todos os rounds
    dados.append(dados_aux)
    df = pd.DataFrame(np.array(dados).T, columns=['pontos iniciais'])
    
    
    dados = []
    
        # agora pega o resultados de todos os rounds para cada query
    for j in range(num_queries):
        dados_aux = []
        for i in range(num_splits):
            # print("\n\ni {} e j==state {}\n\n".format(i,j))
            dados_aux.append(results[i].get_state(j)['performance'])
        df['query {}'.format(j)] = (np.array(dados_aux).T)
    
    df.name = 'Round'
    
    df.to_csv(nomeDestino, decimal = ',', sep=';')

def plot_results_df(df_own, df_original, num_colunas, num_linhas,loc_legenda = 'upper left',
                 titulo = 'teste', nome_arquivo = '', saving_path = '.', tipo_ci = 'std_padrao'):
    #funcao utilizada para plotar graficos de dois df
    #     PARTE 1 - PLOTAGEM DOS DADOS

    sns.reset_defaults()
    # Calcular a média e o desvio padrão de cada coluna de cada dataframe
    means_own = df_own.mean()
    std_devs_own = df_own.std()
    if tipo_ci == 'std_padrao':
        conf_int_low_own =  means_own - std_devs_own
        conf_int_high_own = means_own + std_devs_own
    elif tipo_ci == 'intervalo_confianca':
        conf_int_low_own = means_own - 1.96 * (std_devs_own / (len(dadosDf_own) ** 0.5))
        conf_int_high_own = means_own + 1.96 * (std_devs_own / (len(dadosDf_own) ** 0.5))
    
    means_originais = df_original.mean()
    std_devs_originais = df_original.std()
    if tipo_ci == 'std_padrao':
        conf_int_low_originais =  means_originais - std_devs_originais
        conf_int_high_originais = means_originais + std_devs_originais
    elif tipo_ci == 'intervalo_confianca':
        conf_int_low_originais = means_originais - 1.96 * (std_devs_originais / (len(dadosDf_originais) ** 0.5))
        conf_int_high_originais = means_originais + 1.96 * (std_devs_originais / (len(dadosDf_originais) ** 0.5))
    
    
    # Calcular os intervalos de confiança (por exemplo, usando 95% de confiança)
    
    
    
        # Concatenar os DataFrames
    df_concat = pd.concat([means_own, means_originais], axis=1)
    df_concat.columns = ['minha', 'original']
    
    # Preparar os dados para o gráfico
    x = df_own.columns
    
    
    plt.figure(figsize=(10, 6))
    
    
    sns.lineplot(data=df_concat, x=x, y='original', label='Original', marker='o')
    plt.fill_between(x, conf_int_low_originais, conf_int_high_originais, alpha=0.4)
    
    sns.lineplot(data=df_concat,x=x, y='minha', label = 'Minha', marker='o')
    plt.fill_between(x, conf_int_low_own, conf_int_high_own, alpha=0.4)
    
    
    # Configurar o estilo do seaborn (opcional)
    sns.set(style="whitegrid")
    # Ajustar os rótulos do eixo x para exibir apenas alguns números
    # x_labels = [0]
    # for i in range(1,len(dadosDf_own.columns)):
    #     if (i % 5) == 0:
    #         x_labels.append(i)
    
    
    plt.xticks(range(num_linhas))
    plt.ylabel('taxa de acerto')
    #     plt.xlabel('queries')
    plt.legend(loc= loc_legenda)
    plt.title(titulo)
    plt.savefig(os.path.join(saving_path,nome_arquivo))
    plt.show()

    

def plot_results(results_own, results_originais, numero_splits, num_query, loc_legenda = 'upper left',
                 titulo = 'teste', nome_arquivo = '', saving_path = '.', tipo_ci = 'std_padrao'):
# essa funcao é utilizada para plotar os dados obtidos utilizando a biblioteca
# alipy
#     PARTE 1 - OBTENÇÃO DOS DADOS
#     obtem um datagrama com os resultados obtidos pela minha proposta
    dados = []
    dados_aux = []
    for i in range(numero_splits):
        dados_aux.append(results_own[i].initial_point)

        # salva a primeira lista que tem o resultado para os pontos iniciais de todos os rounds
    dados.append(dados_aux)
    dadosDf = pd.DataFrame(np.array(dados).T, columns=['pontos iniciais'])


    dados = []
    dados_aux = []
    # agora pega o resultados de todos os rounds para cada query
    for j in range(num_query):
        dados_aux = []
        for i in range(numero_splits):
            dados_aux.append(results_own[i].get_state(j)['performance'])
        
        dadosDf['query {}'.format(j)] = (np.array(dados_aux).T)

    dadosDf_own = dadosDf.copy()
    
#     obtem um datagrama com os resultados obtidos pela proposta original, sem alteracao minha
    dados = []
    dados_aux = []
    for i in range(numero_splits):
        dados_aux.append(results_originais[i].initial_point)

        # salva a primeira lista que tem o resultado para os pontos iniciais de todos os rounds
    dados.append(dados_aux)
    dadosDf = pd.DataFrame(np.array(dados).T, columns=['pontos iniciais'])


    dados = []
    dados_aux = []
    # agora pega o resultados de todos os rounds para cada query
    for j in range(num_query):
        dados_aux = []
        for i in range(numero_splits):
            dados_aux.append(results_originais[i].get_state(j)['performance'])
    
        dadosDf['query {}'.format(j)] = (np.array(dados_aux).T)
        
    dadosDf_originais = dadosDf.copy()
    
#     PARTE 2 - PLOTAGEM DOS DADOS

    sns.reset_defaults()
    # Calcular a média e o desvio padrão de cada coluna de cada dataframe
    means_own = dadosDf_own.mean()
    std_devs_own = dadosDf_own.std()
    if tipo_ci == 'std_padrao':
        conf_int_low_own =  means_own - std_devs_own
        conf_int_high_own = means_own + std_devs_own
    elif tipo_ci == 'intervalo_confianca':
        conf_int_low_own = means_own - 1.96 * (std_devs_own / (len(dadosDf_own) ** 0.5))
        conf_int_high_own = means_own + 1.96 * (std_devs_own / (len(dadosDf_own) ** 0.5))
    
    means_originais = dadosDf_originais.mean()
    std_devs_originais = dadosDf_originais.std()
    if tipo_ci == 'std_padrao':
        conf_int_low_originais =  means_originais - std_devs_originais
        conf_int_high_originais = means_originais + std_devs_originais
    elif tipo_ci == 'intervalo_confianca':
        conf_int_low_originais = means_originais - 1.96 * (std_devs_originais / (len(dadosDf_originais) ** 0.5))
        conf_int_high_originais = means_originais + 1.96 * (std_devs_originais / (len(dadosDf_originais) ** 0.5))


    # Calcular os intervalos de confiança (por exemplo, usando 95% de confiança)
    

    
        # Concatenar os DataFrames
    df_concat = pd.concat([means_own, means_originais], axis=1)
    df_concat.columns = ['minha', 'original']
    
# Preparar os dados para o gráfico
    x = dadosDf_own.columns
    
    
    plt.figure(figsize=(10, 6))
    
    
    sns.lineplot(data=df_concat, x=x, y='original', label='Original', marker='o')
    plt.fill_between(x, conf_int_low_originais, conf_int_high_originais, alpha=0.4)
    
    sns.lineplot(data=df_concat,x=x, y='minha', label = 'Minha', marker='o')
    plt.fill_between(x, conf_int_low_own, conf_int_high_own, alpha=0.4)
    

    # Configurar o estilo do seaborn (opcional)
    sns.set(style="whitegrid")
    # Ajustar os rótulos do eixo x para exibir apenas alguns números
    x_labels = [0]
    for i in range(1,len(dadosDf_own.columns)):
        if (i % 5) == 0:
            x_labels.append(i)
    
    
    plt.xticks(x_labels)
    plt.ylabel('taxa de acerto')
#     plt.xlabel('queries')
    plt.legend(loc= loc_legenda)
    plt.title(titulo)
    plt.savefig(os.path.join(saving_path,nome_arquivo))
    plt.show()



# Function to calculate silhouette score for a range of K values
def calculate_silhouette_score(data, k_range):
    silhouette_scores = []
    tam_data = len(data)

    for k in k_range:
        
        if tam_data - 1 < k:
            # print('valor de k {}'.format(k))
            # print('Valor de K é maior ou igual à quantidade de elementos na base de dados.')
            break
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        # print('valor de k {}, tam {}, labels {}'.format(k, tam_data, labels))
    
        
        silhouette_scores.append(silhouette_score(data, labels))
    # Find the optimal K value that maximizes the silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores
# In[EXEMPLO DE USO DO METODO CALCULATE_SILHOUETTE_SCORE]
# Specify the range of K values to test
    k_values = range(2, 11)
    
# Calculate silhouette scores for each K value
    iris = load_iris()
    print(iris.data[:4])
    print(type(iris.data[:4]))
    optimal_k,silhouette_scores = calculate_silhouette_score(iris.data[:4], k_values)


# Plot the silhouette scores
    plt.plot(range(2,len(silhouette_scores)+2), silhouette_scores, marker='o')
    plt.title('Silhouette Score For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.show()
    print(f'Optimal number of clusters (K): {optimal_k} e scores {silhouette_scores}')
    
# In[]


def adaptaNewick(newickS, X, y_total):
    newickS_aux = newickS
    for i in range(len(X)):
#   PARENTESES
        if y_total[i] == 0:
            if i < 10:
#                 Para antes da virgula, ou seja, comeca com (
                newickS_aux = newickS_aux.replace('(00'+str(i)+',','(B00'+str(i)+',')
            elif i >=10 and i < 100:
                newickS_aux = newickS_aux.replace('(0'+str(i)+',','(B0'+str(i)+',')
            else:
                newickS_aux = newickS_aux.replace('('+str(i)+',','(B'+str(i)+',')
        else:
            if i < 10:
                newickS_aux = newickS_aux.replace('(00'+str(i)+',','(P00'+str(i)+',')
            elif i >=10 and i < 100:
                newickS_aux = newickS_aux.replace('(0'+str(i)+',','(P0'+str(i)+',')
            else:
                newickS_aux = newickS_aux.replace('('+str(i)+',','(P'+str(i)+',')
        

    for i in range(len(X)):
#   VIRGULA
        if y_total[i] == 0:
            if i < 10:
#                                 Para depois da virgula, ou seja, comeca com ,
                newickS_aux = newickS_aux.replace(',00'+str(i)+')',',B00'+str(i)+')')
            elif i >=10 and i < 100:
                newickS_aux = newickS_aux.replace(',0'+str(i)+')',',B0'+str(i)+')')
            else:
                newickS_aux = newickS_aux.replace(','+str(i)+')',',B'+str(i)+')')
                
        else:
            if i < 10:
                newickS_aux = newickS_aux.replace(',00'+str(i)+')',',P00'+str(i)+')')
            elif i >=10 and i < 100:
                newickS_aux = newickS_aux.replace(',0'+str(i)+')',',P0'+str(i)+')')
            else:
                newickS_aux = newickS_aux.replace(','+str(i)+')',',P'+str(i)+')')
                
    return newickS_aux


# def adequaFormatoNewick(newick):
#     # Esta funcao recebe o vetor newick original e coloca os nomes do nós no 
#     # no formato a ser utilizado, com 3 caracteres. ex: 002, 024, 435
    
#     #A VARIAVEL NEWICK É UM VETOR DE CHARS. O CODIGO ABAIXO 
#     newickS = '('
#     for i in range(len(newick)):
#         newickS += newick[i]
#     newickS += ');'
#     # print(newickS)
    
#     import re
    
#     newickS2 = newickS.replace(',',',v,')
    
    
#     newickS2.count('(')
#     newickS2.count(')')
#     split_aux = re.split('\)|\(|,',newickS2)
    
#     #AQUI FUNCIONA ATE 999 ELEMENTOS ANALISADOS. PARA MAIS DE 999, UM OUTRO ELIF
#     # DEVERÁ SER INSERIDO POIS TERÁ 4 CARACTERES
    
#     for i in range(len(split_aux)):
#         if split_aux[i] == '':
#             split_aux[i] ='#'
#         elif split_aux[i] == 'v':
#             split_aux[i] = ','
#         elif split_aux[i] == ';':
#             split_aux[i] = ');'# esse parentese eh inserido pois devido ao split, um parente de fechamento é perdido nao sei pq
#         elif int(split_aux[i]) < 10:
#             split_aux[i] = '00'+split_aux[i]
#         elif int(split_aux[i]) >= 10 and int(split_aux[i]) < 100:
#             split_aux[i] = '0'+split_aux[i]
#         else:
#             pass
    
    
#     # AQUI É FEITO A SUBSTITUICAO DOS # POR PARENTESES. ASSIM, O NOME DOS NÓS FICA COM
#     # 3 CARACTERES, EX: 003, 300, 024
    
#     split_aux2 = ''.join(split_aux)
#     # print(split_aux2)
    
#     sequenciaParenteses = []
#     # se for 1 é ( se for 2 é )
#     for i in newickS:
#         if i == '(':
#             sequenciaParenteses.append('(')
#         elif i == ')':
#             sequenciaParenteses.append(')')
#     # print(len(sequenciaParenteses))
    
#     parentesesUsados = 0
#     str_aux = ''
#     for i in range(len(split_aux2)):
#         if split_aux2[i] == '#':
#             split_aux2 = split_aux2[:i] + sequenciaParenteses[parentesesUsados] + split_aux2[i + 1:]
#             parentesesUsados += 1
    
#     newickS2 = split_aux2
    
#     return newickS2, newickS

def avaliaErrosDAMICORE(grupos, y):
    erros = 0
    gruposComErros = []
    qntElementos = 0
    erroPorGrupo = []
    for i, grupo in enumerate(grupos):
             
#             print(grupo)
#             print('tam grupo atual {}'.format(len(grupo)))
            qntElementos += len(grupo)
#             print('classes reais do grupo atual {}'.format(y[grupo]))
                    #se tiver mais elementos da classe 1
            gruposComErros.append(grupo)
                    # comeca aqui o calculo do erro
            counter = collections.Counter(y[grupo])
#             print(counter)
            counter_list = np.array(counter.most_common())
            max_class = counter_list[0][0]
            count_max_class = counter_list[0][1]
#                     todos_elementos = counter_list[:,1]
            erros_grupoAtual = (len(grupo) - count_max_class)
#             print('erros grupo atual {} de {} -> em % {:.3f}'.format(erros_grupoAtual,
#                                                                          len(grupo),
#                                                                          erros_grupoAtual/len(grupo)))
            erroPorGrupo.append(erros_grupoAtual/len(grupo))
            erros += erros_grupoAtual
#             print('erros totais ate o momento {} e por grupo atual {:.3f}\n\n'.format(erros, erroPorGrupo[i]))

#     print('erros totais {} em {}'.format(erros, qntElementos))
    
    return erros, gruposComErros, erroPorGrupo


def avaliaErrosDAMICOREpontos_dict(grupos, y, pontos_dict):
    erros = 0
    gruposComErros = []
    qntElementos = 0
    erroPorGrupo = []
    for i, grupo in enumerate(grupos):
            
            # print('grupo atual {}'.format((grupo)))
            posicao_pontos = getPontosDict(pontos_dict,grupo)
            
            # print('posicao_pontos atual {}'.format((posicao_pontos)))
            # grupo = posicao_pontos
            # print('y do grupo {}'.format(y[posicao_pontos]))
            qntElementos += len(grupo)
#             print('classes reais do grupo atual {}'.format(y[grupo]))
                    #se tiver mais elementos da classe 1
            gruposComErros.append(grupo)
                    # comeca aqui o calculo do erro
            counter = collections.Counter(y[posicao_pontos])
#             print(counter)
            counter_list = np.array(counter.most_common())
            max_class = counter_list[0][0]
            count_max_class = counter_list[0][1]
#                     todos_elementos = counter_list[:,1]
            erros_grupoAtual = (len(grupo) - count_max_class)
            # print('erros grupo atual {} de {} -> em % {:.3f}'.format(erros_grupoAtual,
                                                                          # len(grupo),
                                                                          # erros_grupoAtual/len(grupo)))
            erroPorGrupo.append(erros_grupoAtual/len(grupo))
            erros += erros_grupoAtual
            # print('erros totais ate o momento {} e por grupo atual {:.3f}\n\n'.format(erros, erroPorGrupo[i]))

#     print('erros totais {} em {}'.format(erros, qntElementos))
    

    return erros , gruposComErros, erroPorGrupo

def getPontosDict(dicionario, keys):
    saida = []
    for k in keys:
        saida.append(dicionario[k])
    
    return np.array(saida)

def avaliaErrosDAMICOREpontos_dictOriginal_apagardepois(grupos, y, pontos_dict):
    erros = 0
    for grupo in grupos:
        posicao_pontos = getPontosDict(pontos_dict,grupo)
        print('grupo')
        print(grupo)
        print('posicao pontos')
        print(posicao_pontos)
        if ( len(y[posicao_pontos]) == int(sum(y[posicao_pontos])) ) or sum(y[posicao_pontos]) == 0.0:
            print('mesma classe')
            continue
        else:
            print('nao mesma classe')
            #se tiver mais elementos da classe 1
            if sum(y[posicao_pontos]) >= len(y[posicao_pontos])/2:
                erros += (len(y[posicao_pontos]) - sum(y[posicao_pontos]))
            #se tiver mais elementos da classe 0
            else:
                erros += len(y[posicao_pontos]) - (len(y[posicao_pontos]) - sum(y[posicao_pontos]))
    return erros


def avaliaErrosDAMICOREantigo(grupos, y):
    erros = 0
    gruposComErros = []
    for grupo in grupos:
    #     print(grupo)
    #     print(y[grupo])
        if len(y[grupo]) == int(sum(y[grupo])) or sum(y[grupo]) == 0.0:
    #         print('mesma classe')
            continue
        else:
    #         print('nao mesma classe')
            #se tiver mais elementos da classe 1
            gruposComErros.append(grupo)
            if sum(y[grupo]) >= len(y[grupo])/2:
                erros += (len(y[grupo]) - sum(y[grupo]))
            #se tiver mais elementos da classe 0
            else:
                erros += len(y[grupo]) - (len(y[grupo]) - sum(y[grupo]))
    return erros, gruposComErros

def apagaConteudoDiretorio(indir):
    
    for the_file in os.listdir(indir):
        file_path = os.path.join(indir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
#                 print('é direitorio')
#                 print(indir)
#                 print(file_path)
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            
def salvaFormatoNewick(outdir, tree):
    with open(outdir+'/newick.out', 'w', encoding='utf-8') as file:
        file.write(str(tree)[:-1])
        
def salvaFormatoNewickOutdir(outdir, tree):
    with open(outdir, 'w', encoding='utf-8') as file:
        file.write(str(tree)[:-1])
        

def salvaClustersSemNosIntermediariosDestino(destinoArqClusters, grafoDetCom, pontos_str):
    destinoArqClusters = destinoArqClusters
    if os.path.exists(destinoArqClusters):
        os.remove(destinoArqClusters)
    for i in range(len(grafoDetCom.sizes())):
        nomes_vertices = grafoDetCom.subgraph(i).vs['name']
        nomes_vertices = retiraNoIntermediario(nomes_vertices, pontos_str)
        with open(destinoArqClusters, 'a') as file:
            file.write(vectorToString(nomes_vertices)+'\n')


def salvaClustersSemNosIntermediarios(outdir, destinoArqClusters, grafoDetCom, pontos_str):
    # destinoArqClusters = os.path.join(outdir,'clusters.out')
    destinoArqClusters = os.path.join(outdir,destinoArqClusters)
    if os.path.exists(destinoArqClusters):
        os.remove(destinoArqClusters)
    for i in range(len(grafoDetCom.sizes())):
        nomes_vertices = grafoDetCom.subgraph(i).vs['name']
        nomes_vertices = retiraNoIntermediario(nomes_vertices, pontos_str)
        with open(destinoArqClusters, 'a') as file:
            file.write(vectorToString(nomes_vertices)+'\n')

def salvaClustersConsenso(destinoArqClusters, grafoDetCom):
   
    if os.path.exists(destinoArqClusters):
        os.remove(destinoArqClusters)
    for i in range(len(grafoDetCom.sizes())):
        nomes_vertices = grafoDetCom.subgraph(i).vs['name']
        with open(destinoArqClusters, 'a') as file:
            file.write(vectorToString(nomes_vertices)+'\n')

# In[27]:


def getGruposInfile(arquivo_clusterizacao):
    
    with open(arquivo_clusterizacao, 'r') as file:
        linhas = file.readlines()
        
    
    grupos = []
    for grupo in linhas: 
#         print(grupo)
        grupo_aux = grupo[:-1]
        grupo_aux = grupo_aux.split(sep=',')
#         print(grupo_aux)
        grupo_aux = [int(item) for item in grupo_aux]
        grupos.append(grupo_aux)
        
    return grupos, len(linhas)


# In[28]:
def calculate_centroid(array1):
    
    centroid = np.mean(array1, axis = 0)

    return centroid


def getCentroDosGrupos(grupos, pontos, plot = True, annotate = True, dim1 = 0, dim2 = 1):
    centroDosGrupos = []
    varianciaDosGrupos = []
    desvioPadraoDosGrupos = []

    for grupo in grupos:
        pontosDoGrupo = pontos[grupo]
        centroDoGrupo = calculate_centroid(pontosDoGrupo)
        centroDosGrupos.append(centroDoGrupo)
        varianciaDosGrupos.append(np.var(pontosDoGrupo))
        desvioPadraoDosGrupos.append(np.std(pontosDoGrupo))
    if plot == True:
        for grupo in grupos:
            x_aux = []
            y_aux = []
            for i in grupo:
                x_aux.append(pontos[i,dim1])
                y_aux.append(pontos[i,dim2])
#     print(grupo)
            plt.scatter(x_aux,y_aux, alpha = 0.5)
            # if annotate == True:
            #     for i in grupo:
            #         plt.annotate(str(i), (pontos[i,0], pontos[i,1]))

        for centroDoGrupo in centroDosGrupos:
            plt.scatter(x=centroDoGrupo[dim1], y = centroDoGrupo[dim2], marker='+', c='black')

        plt.savefig('comunidades.pdf', dpi=500)
        plt.show()

    return centroDosGrupos, varianciaDosGrupos, desvioPadraoDosGrupos



def getCentroDosGruposAntigo(grupos, pontos, plot = True, annotate = True):
    centroDosGrupos = []
    varianciaDosGrupos = []
    desvioPadraoDosGrupos = []

    for grupo in grupos:
        pontosDoGrupo = pontos[grupo]
        centroDoGrupo = [sum(pontosDoGrupo[:,0])/len(pontosDoGrupo), sum(pontosDoGrupo[:,1])/len(pontosDoGrupo)]
        centroDosGrupos.append(centroDoGrupo)
        varianciaDosGrupos.append(np.var(pontosDoGrupo))
        desvioPadraoDosGrupos.append(np.std(pontosDoGrupo))
    if plot == True:
        for grupo in grupos:
            x_aux = []
            y_aux = []
            for i in grupo:
                x_aux.append(pontos[i,0])
                y_aux.append(pontos[i,1])
#     print(grupo)
            plt.scatter(x_aux,y_aux, alpha = 0.5)
            if annotate == True:
                for i in grupo:
                    plt.annotate(str(i), (pontos[i,0], pontos[i,1]))

        for centroDoGrupo in centroDosGrupos:
            plt.scatter(x=centroDoGrupo[0], y = centroDoGrupo[1], marker='+', c='black')

        plt.savefig('comunidades.pdf', dpi=500)
        plt.show()

    return centroDosGrupos, varianciaDosGrupos, desvioPadraoDosGrupos


def plotaGrupos(grupos, pontos, indir, titulo, annotate = False):
    plt.figure()
    for grupo in grupos:
        x_aux = []
        y_aux = []
#         posicao = []
        for i in grupo:
            x_aux.append(pontos[i,0])
            y_aux.append(pontos[i,1])
            
#     print(grupo)
        plt.scatter(x_aux,y_aux)
        if annotate == True:
            for i in grupo:
                plt.annotate(str(i), (pontos[i,0], pontos[i,1]))
            
    plt.title(titulo)
    plt.savefig(os.path.join(indir,'clusters.png'))
       
    plt.show

def retiraNoIntermediario(clusterVetor, listaArquivos):
    lista = []
    for i in clusterVetor:
        if i in listaArquivos:
            lista.append(i)
            
    return lista

def vectorToString(s):
    v=''
    for i in s:
        v=v+i+','
    return v[:-1]

def vectorIntToVectorStr(s):
    v = []
    for i in s:
        v.append(str(i))
    return v


# def simetrizacao(matriz):
#     print('#######################\n')
#     print('COMECOU A SIMETRIZACAO\n')
#     print('#######################\n')
#     for i in tqdm(range(len(matriz))):
#         for j in (range(len(matriz))):
#             if matriz[i][j] != matriz[j][i]:
#                 matriz[j][i] = matriz[i][j]
#     return matriz

def simetrizacao(matriz):
    print('#######################\n')
    print('#######################\n')
    print('COMECOU A SIMETRIZACAO\n')
    print('#######################\n')

    matriz = np.array(matriz)

    # # # Tira o máximo entre a matriz e sua transposta, garantindo a simetria
    # matriz = np.maximum(matriz, matriz.T)

    # return matriz

    return 0.5 * (matriz + matriz.T)



def transformacaoPseudoVariaveis(grupos, pontos, X_teste, centroDosGrupos, varianciaDosGrupos):
    #dadosZ é só uma referencia ao livro do ivan, onde esta variavel contera as pseudoamostras = pg 182
    for i in range(len(grupos)):
        if i == 0:
            dadosZ = np.sum((pontos-centroDosGrupos[i])**2, axis=1)/(2*varianciaDosGrupos[i])
            dadosZ = dadosZ.reshape(-1,1)
        else:
            dadosZ_aux = np.sum((pontos-centroDosGrupos[i])**2, axis=1)/(2*varianciaDosGrupos[i])
            dadosZ_aux = dadosZ_aux.reshape(-1,1)
            dadosZ = np.hstack((dadosZ, dadosZ_aux))

    for i in range(len(grupos)):
        if i == 0:
            dadosZ_teste = np.sum((X_teste-centroDosGrupos[i])**2, axis=1)/(2*varianciaDosGrupos[i])
            dadosZ_teste = dadosZ_teste.reshape(-1,1)
        else:
            dadosZ_aux_teste = np.sum((X_teste-centroDosGrupos[i])**2, axis=1)/(2*varianciaDosGrupos[i])
            dadosZ_aux_teste = dadosZ_aux_teste.reshape(-1,1)
            dadosZ_teste = np.hstack((dadosZ_teste, dadosZ_aux_teste))
    
    return dadosZ, dadosZ_teste

def reamostragemMatrizAdj(matrizAdjOriginal, numReamostragem, listaArquivos):
    matrizesAdjReamostradas = []
    listaArquivosReamostrados = []
#     print(matrizAdjOriginal)
#     print(len(matrizAdjOriginal))
#     print(numReamostragem)
#     print(type(numReamostragem))
    for i in range(numReamostragem):
        if i == 0:
            matrizesAdjReamostradas.append(matrizAdjOriginal)
            listaArquivosReamostrados.append(listaArquivos)
        else:
            permutacao = np.random.permutation(np.arange(len(matrizAdjOriginal)))
            matriz_aux = matrizAdjOriginal[permutacao,:]
            matriz_aux = matriz_aux.transpose()
            matriz_aux = matriz_aux[permutacao,:]
            matriz_aux = matriz_aux.transpose()
            
#             print(listaArquivos)
#             print(permutacao)
            listaArquivos_aux = listaArquivos[permutacao]
            
            matrizesAdjReamostradas.append(matriz_aux)
            listaArquivosReamostrados.append(listaArquivos_aux)
    
    return matrizesAdjReamostradas, listaArquivosReamostrados


def reamostragemMatrizDistancia(matrizDistanciaOriginal, numMatrizDistancia, pontos, n_samples, elementos):
    elementosReamostrados = []
    matrizesDeDistanciaReamostradas = []
    sequenciaReamostragem = []
    
    for i in range(numMatrizDistancia):
        
        if i == 0:
            matrizesDeDistanciaReamostradas.append(matrizDistanciaOriginal)
            elementosReamostrados.append(pontos)
            permutacao = np.arange(n_samples)            
            permutacao = np.array([str(item) for item in permutacao])
            sequenciaReamostragem.append(permutacao)

        else:
        # i+1 pq arange(1) é [0]
            permutacao = np.random.permutation(elementos)
            elementos_aux = pontos[permutacao]
            
            matriz_aux = matrizDistanciaOriginal[permutacao, :]
            matriz_aux = matriz_aux.transpose()
            matriz_aux = matriz_aux[permutacao,:]
            matriz_aux = matriz_aux.transpose()

            matrizesDeDistanciaReamostradas.append(matriz_aux)
            elementosReamostrados.append(elementos_aux)
            permutacao = np.array([str(item) for item in permutacao])
            sequenciaReamostragem.append(permutacao)
            
    return elementosReamostrados, matrizesDeDistanciaReamostradas, sequenciaReamostragem

def bootstrap(X,y):
#     type(X) e type(y) => numpy.ndarray
    X_reamostrado = []
    y_reamostrado = []
    
    num_max = len(X)
    for i in range(num_max):
        num_rand = randint(0,num_max-1)
        X_reamostrado.append(X[num_rand])
        y_reamostrado.append(y[num_rand])
        
    return np.array(X_reamostrado),np.array(y_reamostrado)


def distanciaGaussiana(x1,x2, variancia):
    distancia = np.exp(-1*np.sum((x1-x2)**2)/(2*variancia))
    return distancia


def matrizDistanciaGaussiana(amostras):

    variancia = np.var(amostras, axis=0)
    variancia = variancia.sum()
        
    matrizDistancia = [distanciaGaussiana(amostra1,amostra2,variancia) for amostra1 in amostras for amostra2 in amostras]
    matrizDistancia = np.array(matrizDistancia).reshape(amostras.shape[0],amostras.shape[0])
    np.fill_diagonal(matrizDistancia,0)
    
    return matrizDistancia
    

def matrizDistanciaGeodesicaAntiga(amostras, num_vizinhos):
    n_neighbors = num_vizinhos
    neighbors_algorithm = 'auto'
    n_jobs = None
    n_jobs = -1
    metric = 'minkowski'
    metric_params = None
    p = 2
    path_method = 'auto'
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                          algorithm=neighbors_algorithm,
                                          metric=metric,
                                          p=p,
                                          metric_params=metric_params,
                                          n_jobs=n_jobs)
    
    nbrs.fit(amostras)
    
    knnG = kneighbors_graph(nbrs, n_neighbors=n_neighbors,
                        mode='distance', metric=metric, p=p,
                        metric_params=metric_params, n_jobs=n_jobs)
    # essa funcao nao esta mais disponivel
    dist_matrix_ = graph_shortest_path(knnG, method=path_method, directed=False)
    # o valor 1000000 é colocado pois se trata da matriz de distancia. logo, se ficasse 0, seria como se o elemento fosse extremamente proximo
    # assim, coloca-se um valor muito alto para indicar que esse nó j, que nao faz adjacencia com o nó i.
    dist_matrix_ = np.where(dist_matrix_ == np.inf, 1000000000, dist_matrix_)
    np.fill_diagonal(dist_matrix_, 0.0)
    return simetrizacao(dist_matrix_.round(decimals = 10)), knnG, nbrs


def geodesic_knn_graph(X, n_neighbors=10):
    # Construir grafo k-NN com distâncias euclidianas
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=True)

    # Calcular a matriz de distância geodésica utilizando o algoritmo de Floyd-Warshall
    geodesic_dist_matrix = csgraph.shortest_path(knn_graph, method='auto', directed=False)

    # Construir novo grafo k-NN com base nas distâncias geodésicas
    n_samples = X.shape[0]
    geodesic_knn = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        sorted_neighbors = np.argsort(geodesic_dist_matrix[i])[:n_neighbors]
        for j in sorted_neighbors:
            geodesic_knn[i, j] = geodesic_dist_matrix[i, j]
            geodesic_knn[j, i] = geodesic_dist_matrix[i, j]  # Grafo não direcionado

    # Convertendo para matriz esparsa
    geodesic_knn_sparse = csr_matrix(geodesic_knn)

    return geodesic_knn_sparse, geodesic_knn, geodesic_dist_matrix

def geodesic_knn_from_distance_matrix(dist_matrix, n_neighbors=10):
    """
    Cria um grafo k-NN baseado em uma matriz de distâncias precomputada (ex: NCD),
    e calcula as distâncias geodésicas entre os nós.
    
    Parâmetros:
        dist_matrix (np.ndarray): matriz de distâncias (n x n)
        n_neighbors (int): número de vizinhos mais próximos
    
    Retorna:
        geodesic_knn_sparse: matriz esparsa do grafo k-NN geodésico
        geodesic_knn: matriz densa do grafo k-NN geodésico
        geodesic_dist_matrix: matriz completa de distâncias geodésicas
    """
    n_samples = dist_matrix.shape[0]

    # 1. Inicializa matriz do grafo k-NN com base nas distâncias fornecidas
    knn_matrix = np.full_like(dist_matrix, fill_value=0.0)
    
    for i in range(n_samples):
        # Obtém os índices dos k vizinhos mais próximos (inclusive ele mesmo)
        neighbors = np.argsort(dist_matrix[i])[:n_neighbors]
        for j in neighbors:
            knn_matrix[i, j] = dist_matrix[i, j]
            knn_matrix[j, i] = dist_matrix[i, j]  # Garante simetria

    knn_sparse = csr_matrix(knn_matrix)

    # 2. Calcula a matriz de distâncias geodésicas (menor caminho entre os nós)
    geodesic_dist_matrix = shortest_path(knn_sparse, method='auto', directed=False)

    # 3. Constrói novo grafo k-NN com base nas distâncias geodésicas
    geodesic_knn = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        sorted_neighbors = np.argsort(geodesic_dist_matrix[i])[:n_neighbors]
        for j in sorted_neighbors:
            geodesic_knn[i, j] = geodesic_dist_matrix[i, j]
            geodesic_knn[j, i] = geodesic_dist_matrix[i, j]  # Garante simetria

    geodesic_knn_sparse = csr_matrix(geodesic_knn)

    return geodesic_knn_sparse, geodesic_knn, geodesic_dist_matrix

def matrizDistanciaGeodesica(amostras, num_vizinhos):
    n_neighbors = num_vizinhos
    neighbors_algorithm = 'auto'
    n_jobs = None
    n_jobs = -1
    metric = 'minkowski'
    metric_params = None
    p = 2
    path_method = 'auto'
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                              algorithm=neighbors_algorithm,
                                              metric=metric,
                                              p=p,
                                              metric_params=metric_params,
                                              n_jobs=n_jobs)
        
    nbrs.fit(amostras)
        
    knnG = kneighbors_graph(nbrs, n_neighbors=n_neighbors,
                            mode='distance', metric=metric, p=p,
                            metric_params=metric_params, n_jobs=n_jobs)
    
    for i in tqdm(range(len(amostras))):
        if i == 0:
            matrix = shortest_path(csgraph=knnG, directed=False, indices=i, return_predecessors=False)
        else:
            dist_matrixAux = shortest_path(csgraph=knnG, directed=False, indices=i, return_predecessors=False)
            matrix = np.vstack((matrix, dist_matrixAux))

            

    matrix = np.where(matrix == np.inf, 1000000000, matrix)
    np.fill_diagonal(matrix, 0.0)

    print(f'shape da matriz {matrix.shape}  tipo {type(matrix)}')
    
    # na dist geodesica não será utilizado a simetrizacao
    # pois nos testes realizados a matriz de distancia
    # gerada já é simétrica.
    # return simetrizacao(matrix.round(decimals = 10)), knnG, nbrs
    return matrix.round(decimals = 10), knnG, nbrs


from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import concurrent.futures
import numpy as np
from tqdm import tqdm


def calculate_path(knnG, index):
    return shortest_path(csgraph=knnG, directed=False, indices=index, return_predecessors=False)


def matrizDistanciaGeodesicaMultiThread(amostras, num_vizinhos):
    n_neighbors = num_vizinhos
    neighbors_algorithm = 'auto'
    n_jobs = -1  # Utiliza todos os processadores
    metric = 'minkowski'
    metric_params = None
    p = 2
    path_method = 'auto'

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=neighbors_algorithm,
                            metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
    nbrs.fit(amostras)
    knnG = kneighbors_graph(nbrs, n_neighbors=n_neighbors, mode='distance',
                            metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)

    # Usar ThreadPoolExecutor para paralelizar o cálculo dos caminhos mais curtos
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_path, knnG, i) for i in range(len(amostras))]
        # results = [f.result() for f in futures, total=len(amostras)]
        # Coletando os resultados e imprimindo o progresso
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 100 == 0:  # Imprime a cada 100 tarefas completas
                print(f'Progresso: {i}/{len(amostras)} tarefas completadas.')
            results.append(future.result())

    # Combina todos os resultados em uma única matriz
    matrix = np.vstack(results)
    matrix = np.where(matrix == np.inf, 1000000000, matrix)
    np.fill_diagonal(matrix, 0.0)

    # Não há necessidade de simetrização, conforme indicado
    return matrix.round(decimals=10), knnG, nbrs


# Exemplo de uso
# amostras = seu_data_array
# num_vizinhos = seu_valor_de_vizinhos
# matrix, knnG, nbrs = matrizDistanciaGeodesica(amostras, num_vizinhos)

def matrizDistanciaCofenetica(amostras, num_vizinhos):
    n_amostras = len(amostras)
    
    #construi arvore filogenetiva baseada na
    #distancia geodesica
    label_nos = [str(i) for i in range(n_amostras)]
    matrix_dist, knnG, nbrs = matrizDistancia(amostras, "geodesica", num_vizinhos)
    mdt = DistanceMatrix(matrix_dist, label_nos)
    arvore = nj(mdt)
    
    #converte o formato newick para um objeto Ete3
    tEte3 = Tree(str(arvore)[:-1])
    
    #inicializar uma matriz com zeros
    distanciaTopologica = np.zeros(n_amostras*n_amostras).reshape(n_amostras,n_amostras)
    
    #calcula distancia topologica par a par
    for i in range(n_amostras):
        for j in range(n_amostras):
            if i != j:
                distanciaTopologica[i][j] = tEte3.get_distance(str(i),str(j), topology_only=True)

    return distanciaTopologica

# -------- CONFIGURAR pyncd.py UMA VEZ --------
def setup_pyncd():
    script_dir = os.getcwd()
    pyncd_path = os.path.join(script_dir, 'pyncd.py')
    os.chmod(pyncd_path, 0o755)
    with open(pyncd_path, 'r+') as f:
        content = f.read()
        if not content.startswith('#!'):
            f.seek(0, 0)
            f.write('#!/usr/bin/env python3\n' + content)
    return pyncd_path

# -------- SALVAR LINHAS EM ARQUIVOS --------

def save_dataframe_rows_to_tempfiles(df):
    temp_files = []
    for i, (_, row) in enumerate(df.iterrows()):
        binary_data = b''
        for value in row.values:
            # Serialização de listas numéricas
            if isinstance(value, list):
                try:
                    array = np.array(value, dtype=np.float32)
                except ValueError:
                    array = np.array([float(x) for x in value], dtype=np.float32)
                binary_data += array.tobytes()
            # Serialização de outros tipos
            else:
                try:
                    binary_data += np.array(value, dtype=np.float32).tobytes()
                except:
                    binary_data += str(value).encode('utf-8')
        
        with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=f'_row{i}.bin') as tmp:
            tmp.write(binary_data)
            temp_files.append(tmp.name)
    return temp_files

# def save_dataframe_rows_to_tempfiles(df):
#     temp_files = []
#     for i, (_, row) in enumerate(df.iterrows()):
#         content = '|'.join([str(x) for x in row.values])
#         with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=f'_row{i}.txt') as tmp:
#             tmp.write(content)
#             temp_files.append(tmp.name)
#     return temp_files

# # -------- DISTÂNCIA NCD ENTRE DOIS ARQUIVOS --------
# def ncd_pyncd(file1, file2, pyncd_path):
#     result = subprocess.run(['python3', pyncd_path, file1, file2], capture_output=True, text=True)
#     try:
#         return float(result.stdout.strip().split('\n')[-1])
#     except ValueError:
#         return np.nan

# -------- DISTÂNCIA NCD ENTRE DOIS ARQUIVOS --------
def ncd_pyncd(file1, file2, compress, pyncd_path):
    result = subprocess.run(['python3', pyncd_path, file1, file2, compress], capture_output=True, text=True)
    try:
        return float(result.stdout.strip().split('\n')[-1])
    except ValueError:
        return np.nan


# -------- PARALELIZAÇÃO PARA MATRIZ NCD --------
def calcular_matriz_ncd_parallel(df, compress, n_jobs=-1):

    pyncd_path = setup_pyncd()
    temp_files = save_dataframe_rows_to_tempfiles(df)
    n = len(temp_files)

    index_pairs = [(i, j) for i in range(n) for j in range(i, n)]

    print("⏱️ Calculando pares em paralelo...")
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(ncd_pyncd)(temp_files[i], temp_files[j], pyncd_path)
    #     for i, j in tqdm(index_pairs)
    # )

    from joblib import parallel_backend

    with parallel_backend("threading"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(ncd_pyncd)(temp_files[i], temp_files[j], compress, pyncd_path)
            for i, j in tqdm(index_pairs)
        )

    dist_matrix = np.zeros((n, n))
    for (i, j), dist in zip(index_pairs, results):
        dist_matrix[i][j] = dist
        dist_matrix[j][i] = dist

    
    # Criar arquivo .zip com os arquivos temporários
    zip_path = os.path.join(tempfile.gettempdir(), "arquivos_temporarios.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for fpath in temp_files:
            zipf.write(fpath, arcname=os.path.basename(fpath))

    # Baixar o .zip no Google Colab
    files.download(zip_path)

    for f in temp_files:
        os.remove(f)

    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

# -------- GRAFO GEODÉSICO A PARTIR DE MATRIZ DE DISTÂNCIA --------
def geodesic_knn_from_distance_matrix(dist_matrix, n_neighbors=10):
    n_samples = dist_matrix.shape[0]
    knn_matrix = np.full_like(dist_matrix, fill_value=0.0)

    for i in range(n_samples):
        neighbors = np.argsort(dist_matrix[i])[:n_neighbors]
        for j in neighbors:
            knn_matrix[i, j] = dist_matrix[i, j]
            knn_matrix[j, i] = dist_matrix[i, j]

    knn_sparse = csr_matrix(knn_matrix)
    geodesic_dist_matrix = shortest_path(knn_sparse, method='auto', directed=False)

    geodesic_knn = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        sorted_neighbors = np.argsort(geodesic_dist_matrix[i])[:n_neighbors]
        for j in sorted_neighbors:
            geodesic_knn[i, j] = geodesic_dist_matrix[i, j]
            geodesic_knn[j, i] = geodesic_dist_matrix[i, j]

    geodesic_knn_sparse = csr_matrix(geodesic_knn)
    return geodesic_knn_sparse, geodesic_knn, geodesic_dist_matrix
    


# Função para converter uma linha do DataFrame em uma string
def row_to_string(row):
    return ' '.join(map(str, row.values))

# Função para calcular NLD entre duas strings
def normalized_levenshtein_distance(x: str, y: str) -> float:
    if not x and not y:
        return 0.0
    lev_dist = Levenshtein.distance(x, y)
    max_len = max(len(x), len(y))
    return lev_dist / max_len

# def nld_distance_matrix_parallel(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
#     """
#     Calcula em paralelo a matriz de distância normalizada de Levenshtein
#     entre todas as linhas de um DataFrame.
#
#     Parâmetros
#     ----------
#     df : pandas.DataFrame
#         DataFrame cujas linhas serão comparadas umas às outras.
#     n_jobs : int, opcional
#         Número de processos a usar para paralelização. -1 usa todos os núcleos disponíveis.
#
#     Retorna
#     -------
#     pandas.DataFrame
#         Matriz simétrica de distâncias, indexada e colunada como `df`.
#     """
#     # Concatena cada linha em uma string única
#     strings = df.astype(str).agg(" ".join, axis=1).to_list()
#     n = len(strings)
#     matrix = np.zeros((n, n), dtype=float)
#
#     # Gera lista de pares (i, j) com i <= j
#     pairs = list(combinations_with_replacement(range(n), 2))
#
#     # Função auxiliar que calcula distância para um par
#     def calc_dist(pair):
#         i, j = pair
#         dist = normalized_levenshtein_distance(strings[i], strings[j])
#         return i, j, dist
#
#     # Executa todos os pares em paralelo com barra de progresso
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(calc_dist)(pair) for pair in tqdm(pairs, total=len(pairs))
#     )
#
#     # Preenche a matriz simétrica com os resultados
#     for i, j, dist in results:
#         matrix[i, j] = dist
#         matrix[j, i] = dist
#
#     # Retorna como DataFrame preservando índices e colunas originais
#     return pd.DataFrame(matrix, index=df.index, columns=df.index)


def nld_distance_matrix_parallel(df, n_jobs=-1):  # Mantemos a assinatura para compatibilidade
    """
    Calcula a matriz de distâncias NLD (Normalized Levenshtein Distance).
    Versão sequencial robusta (substituindo a paralela que estava falhando).
    """
    # Se df for DataFrame, pega os valores da primeira coluna (assumindo que é a string)
    # Se já for uma série ou lista, usa direto.
    if isinstance(df, pd.DataFrame):
        # Assume que os dados estão na primeira coluna se for um DF de strings
        data = df.iloc[:, 0].astype(str).tolist()
    else:
        data = df.astype(str).tolist()

    n = len(data)
    dist_matrix = np.zeros((n, n))

    print(f"Calculando matriz de distância ({n}x{n}) de forma sequencial...")

    # Loop duplo otimizado (calcula apenas o triângulo superior)
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            # Chama a sua função de distância de string
            # Certifique-se de que normalized_levenshtein_distance está acessível aqui
            dist = normalized_levenshtein_distance(data[i], data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix

# Função principal: gera matriz de distância entre linhas de um DataFrame
def nld_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Converte todo o DataFrame em strings de uma vez
    strings = df.astype(str).agg(' '.join, axis=1).to_list()
    n = len(strings)
    matrix = np.zeros((n, n), dtype=float)

    # Calcula apenas para pares únicos (i <= j) e preenche a matriz simétrica
    for i, j in tqdm(combinations_with_replacement(range(n), 2), total=n*(n+1)//2):
        dist = normalized_levenshtein_distance(strings[i], strings[j])
        matrix[i, j] = dist
        matrix[j, i] = dist

    # The following line is the fix. It creates a DataFrame from the `matrix`.
    return pd.DataFrame(matrix, index=df.index, columns=df.index)

def matrizDistancia(amostras, tipo, num_vizinhos = 6):
    # num_vizinhos É USADO APENAS QUANDO A DISTANCIA FOR GEODESICA
    if tipo == 'gaussiana':
        return matrizDistanciaGaussiana(amostras)
    elif tipo == 'euclidiana':
        # return simetrizacao(euclidean_distances(amostras, amostras))
        return euclidean_distances(amostras, amostras)
    elif tipo == 'geodesica':
        return geodesic_knn_graph(amostras, num_vizinhos)
        # return matrizDistanciaGeodesica(amostras, num_vizinhos)
    elif tipo == 'cofeneticaTopologica':
        return matrizDistanciaCofenetica(amostras, num_vizinhos)
    elif tipo == 'geodesica_multi_thread':
        return matrizDistanciaGeodesicaMultiThread(amostras, num_vizinhos)
    



def outliersSupInf(df, quartil1, quartil3, whis = 1.5):
    q1, q3 = df.quantile([quartil1, quartil3]).values
#     print(q1,q3)
    iqr = q3 - q1
#     print(iqr)

    lower_bound = q1 - (whis * iqr) 
    upper_bound = q3 + (whis * iqr) 
#     print(lower_bound, upper_bound)
    
    repeticoesAcimaUpperBound = df > upper_bound
    repeticoesAcimaUpperBound = repeticoesAcimaUpperBound[repeticoesAcimaUpperBound == True]
    repeticoesAbaixoLowerBound = df < lower_bound
    repeticoesAbaixoLowerBound = repeticoesAbaixoLowerBound[repeticoesAbaixoLowerBound == True]
    repeticoesAcimaUpperBound, repeticoesAbaixoLowerBound

    outliersSupIdx = repeticoesAcimaUpperBound.index.to_list()
    outliersSupRepValues = abs(df[outliersSupIdx].values)
    outliersInfIdx = repeticoesAbaixoLowerBound.index.to_list()
    outliersInfRepValues = abs(df[outliersInfIdx].values)
    
    return outliersSupIdx, outliersSupRepValues, outliersInfIdx, outliersInfRepValues


if __name__ == "__main__":
    grupos = [[64,53,0,120,31,107,48,77,148,91,58,89,138,30,37,28,65,45,51,
  92,84,79,17,88,67,55,47,78,135,29,2,80,108,22,95,104,139,93],
              [86,121,103,102,40,97,82,117,35,99,98,101,137,105,133,60,12,
  18,7,24,111,56,94,41,76,39,112,1,23,134,52,140,20,87,13,144,25],
              [63,66,81,90,109,14,100,27,62,132,16,125,44,46,122,83,73,34,147,
  68,113,8,118,149,50,85,32,124,21,146,70,115,69,143,142,141,38,15,3],
              [49,114,106,136,43,26,126,110,96,57,9,33,6,71,36,11,72,116,5,
  75,42,61,119,74,131,129,19,4,128,54,127,10,130,123,145,59]]
    pontos = np.array([[0.96208036,-1.31922251],[-0.21197622,1.42283498],[0.86526697,-1.43012002],[0.23994981,-1.51030465],[-1.06588637,1.36507658],[-1.45082214,0.90690586],[-1.69440162,0.09448841],[0.39763865,0.75435296],[-0.35201053,-0.52721918],[-1.64053164,0.3429067],[-0.55192222,1.56513473],[-1.40324697,0.86045709],[0.33948851,0.649851],[0.16630829,1.15665678],[0.18389686,-1.45722123],[0.21015232,-1.57890894],[-0.11658085,-1.24472836],[1.73313303,0.36224599],[0.37185883,0.66432777],[-1.08149547,1.29672252],[0.00297357,1.28862097],[-0.64623424,0.29554546],[0.76168745,-1.49426103],[-0.17827986,1.47272605],[0.28766612,1.02425649],[0.32616853,0.7324244],[-1.75433467,-0.38164122],[0.10789298,-1.39702715],[1.60239204,-0.29393279],[0.98927631,-1.46209603],[1.52043838,-0.69265247],[1.29539035,-1.4382841],[-0.58783269,0.30576416],[-1.67758104,0.23130616],[-0.31849195,-0.89624062],[0.54150971,0.08656783],[-1.4589059,0.72328355],[1.53964022,-0.4951539],[-0.4339606,-0.6855141],[-0.00862254,1.57605878],[0.63751464,-0.22958141],[-0.0429026,1.22450109],[-1.28028274,1.00277511],[-1.6910955,-0.38202008],[-0.09855439,-1.0590276],[1.72666167,-0.15299015],[-0.21564381,-1.03227273],[1.3620808,-1.17574995],[1.25120692,-1.08421587],[-1.67443436,-0.09846626],[-0.44771556,-0.09756735],[1.75260601,0.05104156],[-0.33789402,1.61289945],[0.9178225,-1.30348331],[-0.85929778,1.35214836],[1.25897329,-1.15767218],[0.154599,1.04485378],[-1.60324746,0.34667102],[1.38302874,-0.89119949],[-0.73136618,1.49965734],[0.46095189,0.61420106],[-1.26962322,1.10901007],[0.02944343,-1.4258614],[0.51719024,-1.46472504],[0.88507675,-1.30108989],[1.66299976,-0.19798412],[0.48165007,-1.36232436],[1.66596951,-0.0622081],[-0.3829936,-0.81532465],[-0.55264199,-0.04374472],[-0.53702336,0.16255005],[-1.35593047,0.83366288],[-1.48391923,0.78449156],[-0.31456787,-0.96294552],[-1.28074158,1.11100897],[-1.17531151,1.17843437],[-0.13580438,1.33217254],[1.25173598,-0.94883194],[0.95886386,-1.3964019],[1.68474727,0.54035836],[0.79008331,-1.4024266],[0.42585584,-1.46806167],[0.49252013,-0.16490271],[-0.20459221,-0.90409362],[1.76479492,0.23064612],[-0.56073027,0.24813435],[0.64121865,-1.32020231],[0.16873783,1.14953659],[1.69087645,-0.08355061],[1.36145118,-0.80120945],[0.39175072,-1.67769564],[1.43876995,-0.97360421],[1.65616017,0.12567659],[0.62857521,-1.65176736],[0.12448187,1.23478362],[0.73580061,-1.49772788],[-1.60679253,0.32293282],[0.57697162,-0.21897598],[0.41916671,0.20278232],[0.51122968,0.09747016],[0.13091318,-1.34012757],[0.54895275,0.28867061],[0.58102688,-0.29396367],[0.57252401,-0.35658355],[0.57424976,-1.45802906],[0.41172822,0.31180814],[-1.75993015,-0.21407012],[1.07311413,-1.12198441],[0.81416418,-1.56076115],[0.27432676,-1.5078449],[-1.56311814,0.23875465],[0.1621013,1.01923505],[-0.3180295,1.35576821],[-0.37277704,-0.5739901],[-1.69332172,-0.23721361],[-0.52934968,-0.07159111],[-1.50898948,0.87269708],[0.55746237,-0.07555056],[-0.45961281,-0.47676762],[-1.28781969,1.10730396],[1.05625606,-1.25520026],[0.58531652,-0.58076583],[-0.31121531,-1.09613155],[-0.63525029,1.69816704],[-0.6446758,0.47925234],[-0.09485214,-1.08596526],[-1.65174294,0.53754421],[-0.7195998,1.40058181],[-0.89183313,1.34918813],[-0.9745603,1.38724406],[-0.58856097,1.57592552],[-0.99602834,1.28597569],[0.05943982,-1.28062689],[0.46145716,0.46993223],[-0.38315932,1.47169002],[0.9722082,-1.60306257],[-1.6845889,-0.50883151],[0.46409023,0.28127321],[1.57397837,-0.85419096],[0.67606891,-1.60751952],[-0.0047905,1.31472246],[-0.40942246,-0.62319757],[-0.51667188,-0.24699665],[-0.50948994,-0.21146884],[0.3270177,0.75470552],[-0.66365658,1.47824994],[-0.53394975,0.15681168],[-0.37816858,-0.90213413],[1.31047195,-0.9570717],[-0.45302164,-0.37769381]])
    
    # centros, var, std = getCentroDosGrupos(grupos, pontos)
#     centroDosGrupos = []
#     varianciaDosGrupos = []
#     desvioPadraoDosGrupos = []
#     for grupo in grupos:
#         print('grupo')
#         print(grupo)
#         pontosDoGrupo = pontos[grupo]
#         print('pontodosgrupos')
#         print(pontosDoGrupo)
#         centroDoGrupo = [sum(pontosDoGrupo[:,0])/len(pontosDoGrupo), sum(pontosDoGrupo[:,1])/len(pontosDoGrupo)]
#         centroDosGrupos.append(centroDoGrupo)
#         varianciaDosGrupos.append(np.var(pontosDoGrupo))
#         desvioPadraoDosGrupos.append(np.std(pontosDoGrupo))
    
#     for grupo in grupos:
#             x_aux = []
#             y_aux = []
#             for i in grupo:
#                 x_aux.append(pontos[i,0])
#                 y_aux.append(pontos[i,1])
# #     print(grupo)
#             plt.scatter(x_aux,y_aux, alpha = 0.5)
    
    
    
# In[teste matriz cofenetica]

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
