import os
import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

from tqdm import tqdm

from joblib import Parallel, delayed

from typing import Dict, Any, List, Tuple, Union

import logging

import utils

import pickle

import time

def phyil(
    results: Dict[str, Dict[str, Any]],
    n_neighbor: int,
    detection_algo: str,
    resolution: float,
    sample_label_type: str,
    utils_module: Any
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, List[int]]
]:
    """
    Executa detecção de comunidades via DAMICore e seleciona amostras para rotulagem.

    Para cada tipo em ['FIRST', 'LAST'], aplica:
      1) utils_module.damicore(dist_df, matriz_dist, n_neighbor, detection_algo, resolution)
         → retorna (grupos, grupos_all_nodes, tree)
      2) utils_module.chooseSamplesToLabel(grupos, None, tree, "nj_tree.newick", sample_label_type)
         → retorna lista de índices de amostras próximas aos centros de comunidade

    Parâmetros:
    -----------
    results : dict
        Dicionário contendo, para cada tipo ('FIRST' e 'LAST'), uma chave 'dist_df' com
        o DataFrame de distâncias e outras informações.
    n_neighbor : int
        Número de vizinhos a considerar na construção do grafo de comunidades.
    detection_algo : str
        Nome do algoritmo de detecção de comunidades (passado a utils_module.damicore).
    resolution : float
        Parâmetro de resolução para ajuste de granularidade na detecção de comunidades.
    sample_label_type : str
        Identificador de estratégia de amostragem para chooseSamplesToLabel.
    utils_module : module
        Módulo com as funções damicore() e chooseSamplesToLabel().

    Retorna:
    --------
    grupos : Dict[str, Any]
        Dicionário com o objeto 'grupos' retornado para cada tipo.
    grupos_all_nodes : Dict[str, Any]
        Dicionário com o objeto 'grupos_all_nodes' retornado para cada tipo.
    trees : Dict[str, Any]
        Dicionário com o objeto 'tree' retornado para cada tipo.
    elementos_prox_centros : Dict[str, List[int]]
        Índices de amostras selecionadas para rotulagem em cada tipo.

    Levanta:
    --------
    KeyError
        Se 'FIRST' ou 'LAST' não estiverem em results, ou se faltar 'dist_df'.
    Exception
        Qualquer erro inesperado propagado pela utils_module.
    """
    tipos = ['FIRST', 'LAST']
    grupos: Dict[str, Any] = {}
    grupos_all_nodes: Dict[str, Any] = {}
    trees: Dict[str, Any] = {}
    elementos_prox_centros: Dict[str, List[int]] = {}

    for tipo in tipos:
        if tipo not in results or 'dist_df' not in results[tipo]:
            raise KeyError(f"Chave 'dist_df' não encontrada em results['{tipo}']")

        dist_df = results[tipo]['dist_df']
        dist_matrix = dist_df.to_numpy()

        # 1) Detecção de comunidades
        logging.debug(f"Executando damicore para tipo={tipo}, "
                      f"n_neighbor={n_neighbor}, algo={detection_algo}, resolution={resolution}")

        grupos[tipo], grupos_all_nodes[tipo], trees[tipo] = utils_module.damicore(
            dist_df,
            y=None,
            distancias = dist_matrix,
            n_neighbors = n_neighbor,
            algo_deteccao = detection_algo,
            resolution=resolution
        )

        # 2) Seleção de amostras para rotulagem
        logging.debug(f"Selecionando amostras para tipo={tipo} com sample_label_type={sample_label_type}")
        elementos = utils_module.chooseSamplesToLabel(
            grupos[tipo],
            None,
            trees[tipo],
            "nj_tree.newick",
            sample_label_type
        )
        if not isinstance(elementos, list):
            raise TypeError(f"Esperado list de índices em chooseSamplesToLabel, obteve {type(elementos)}")
        elementos_prox_centros[tipo] = elementos

    return grupos, grupos_all_nodes, trees, elementos_prox_centros


def difusao_rotulos(
        df: pd.DataFrame,
        matriz_dist_path: str,
        resultados: dict,
        elementos_proximos_centros: dict,
        algoritmo: str = 'label_spreading',
        sigma: float = 0.02,
        alpha: float = 0.1,
        qq: int = 16,
        salvar_resultados: bool = True,
        limiar_margem_forte: float = 0.6,
        limiar_margem_fraca: float = 0.2,
        limiar_entropia_forte: float = 0.8,
        limiar_entropia_fraca: float = 0.9,
        matriz_sim: bool = False,
        matriz_sim_path: str = None,
        grupos=None
) -> Tuple[pd.DataFrame, List[int], List[int], np.ndarray, Dict, Dict, Dict]:  # ALTERADO: Tipo de retorno
    """
    Executa propagação de rótulos e retorna informações detalhadas dos grupos.

    Retorna:
    --------
    Tuple[pd.DataFrame, List[int], List[int], np.ndarray, Dict, Dict, Dict]
        - DataFrame com rótulos previstos e métricas.
        - Lista de índices rotulados como classe 0 (FIRST).
        - Lista de índices rotulados como classe 1 (LAST).
        - Matriz de similaridade utilizada.
        - Dicionário com os IDs das amostras em cada grupo ('FIRST' e 'LAST').
        - Dicionário com os IDs das amostras selecionadas para rotulagem.
        - NOVO: Dicionário de DataFrames ('FIRST', 'LAST') mapeando cada ID original à sua comunidade.
    """
    # ... (O corpo da função do passo 1 ao 8 permanece exatamente o mesmo) ...
    # Validação de parâmetros
    if algoritmo not in ['label_propagation', 'label_spreading']:
        raise ValueError("Algoritmo deve ser 'label_propagation' ou 'label_spreading'")
    if not os.path.exists(matriz_dist_path):
        raise FileNotFoundError(f"Arquivo {matriz_dist_path} não encontrado")
    # 1. Carregamento da matriz de distância
    dist_df = pd.read_csv(matriz_dist_path, index_col=0, sep=';', decimal=',')
    dist_df = dist_df.map(lambda x: float(str(x).replace(',', '.'))).astype(float)

    # 2. Preparação dos rótulos iniciais
    def extrair_indices_rotulados(chave: str) -> List[int]:
        df_foco = resultados[chave]['df_foco']
        indices_selecionados = df_foco.index[df_foco.index.isin(elementos_proximos_centros[chave])]
        return df_foco.loc[indices_selecionados]['index'].tolist()

    labeled_indices = {'first': extrair_indices_rotulados('FIRST'), 'last': extrair_indices_rotulados('LAST')}
    # 3. Vetor de rótulos
    labels = np.full(len(df), -1, dtype=int)
    np.put(labels, labeled_indices['first'], 0)
    np.put(labels, labeled_indices['last'], 1)
    # 4. Construção do kernel
    D = dist_df.to_numpy()
    if matriz_sim:
        similarity_matrix = pd.read_csv(matriz_sim_path, sep=';', decimal=',', index_col=0).to_numpy()
    else:
        similarity_matrix = np.exp(-np.square(D) / (2 * sigma ** 2))

    def custom_kernel(X, Y=None):
        return similarity_matrix

    # 5. Configuração do modelo
    if algoritmo == 'label_spreading':
        model = LabelSpreading(kernel=custom_kernel, alpha=alpha, n_jobs=-1, max_iter=100)
    else:
        model = LabelPropagation(kernel=custom_kernel, n_jobs=-1)
    # 6. Treinamento
    X_dummy = np.zeros((len(df), 1))
    model.fit(X_dummy, labels)
    # 7. Pós-processamento
    df = df.assign(**{f'predicted_label_{qq}_quantis': model.transduction_, f'margem_{qq}_quantis': np.abs(
        model.label_distributions_[:, 0] - model.label_distributions_[:, 1]), f'entropia_{qq}_quantis': -np.sum(
        model.label_distributions_ * np.log2(model.label_distributions_ + 1e-12), axis=1)})

    df[f'prob_classe0_{qq}_quantis'] = model.label_distributions_[:, 0]
    df[f'prob_classe1_{qq}_quantis'] = model.label_distributions_[:, 1]

    # 8. Classificação de confiança
    margens = df[f'margem_{qq}_quantis']
    entropias = df[f'entropia_{qq}_quantis']
    df[f'status_margem_{qq}_quantis'] = np.select([margens > limiar_margem_forte, margens < limiar_margem_fraca],
                                                  ['Confiança Forte', 'Confiança Fraca'], default='Confiança Média')
    df[f'status_entropia_{qq}_quantis'] = np.select(
        [entropias < limiar_entropia_forte, entropias > limiar_entropia_fraca], ['Confiança Forte', 'Confiança Fraca'],
        default='Confiança Média')

    # --- NOVO: Bloco para processar os 'grupos' e mapear para os IDs originais ---
    indices_reais_dos_grupos = {}
    if grupos:
        for tipo in ['FIRST', 'LAST']:
            if tipo in grupos:
                comunidades = grupos[tipo]
                df_foco = resultados[tipo]['df_foco']

                mapeamento_comunidades = []
                for id_comunidade, membros_da_comunidade in enumerate(comunidades):
                    # Para cada membro na comunidade, encontre seu ID original na coluna 'index' do df_foco
                    indices_originais = df_foco.loc[df_foco.index.isin(membros_da_comunidade)]['index'].tolist()

                    for original_id in indices_originais:
                        mapeamento_comunidades.append({
                            'original_ID': original_id,
                            'comunidade_id': id_comunidade
                        })

                # Converte a lista de mapeamentos em um DataFrame para fácil utilização
                indices_reais_dos_grupos[tipo] = pd.DataFrame(mapeamento_comunidades)

    # ALTERADO: A instrução de retorno agora inclui o novo dicionário de mapeamento
    return df.copy(), labeled_indices['first'], labeled_indices[
        'last'], similarity_matrix, grupos, elementos_proximos_centros, indices_reais_dos_grupos


# def difusao_rotulos(
#     df: pd.DataFrame,
#     matriz_dist_path: str,
#     resultados: dict,
#     elementos_proximos_centros: dict,
#     algoritmo: str = 'label_spreading',
#     sigma: float = 0.02,
#     alpha: float = 0.1,
#     qq: int = 16,
#     salvar_resultados: bool = True,
#     limiar_margem_forte: float = 0.6,
#     limiar_margem_fraca: float = 0.2,
#     limiar_entropia_forte: float = 0.8,
#     limiar_entropia_fraca: float = 0.9,
#     matriz_sim: bool = False,
#     matriz_sim_path: str = None,
#     grupos = None
# ) -> Tuple[pd.DataFrame, List[int], List[int], np.ndarray]:
#     """
#     Executa propagação de rótulos usando matriz de similaridade pré-calculada.
#
#     Parâmetros:
#     -----------
#     df : pd.DataFrame
#         DataFrame original com dados a serem rotulados.
#     matriz_dist_path : str
#         Caminho para arquivo CSV com matriz de distâncias.
#     resultados : dict
#         Dicionário com resultados de pré-processamento contendo índices de foco.
#     elementos_proximos_centros : dict
#         Dicionário com índices de elementos próximos aos centros.
#     algoritmo : {'label_propagation', 'label_spreading'}, padrão='label_propagation'
#         Algoritmo de difusão a ser utilizado.
#     sigma : float, padrão=0.02
#         Parâmetro do kernel Gaussiano.
#     alpha : float, padrão=0.5
#         Parâmetro de suavização (apenas para Label Spreading).
#     qq : int, padrão=16
#         Número de quantis usado no pré-processamento (para nomenclatura de arquivos).
#     salvar_resultados : bool, padrão=True
#         Se True, salva resultados em arquivo CSV.
#     limiar_margem_* : float
#         Limiares para classificação de confiança baseada em margens.
#     limiar_entropia_* : float
#         Limiares para classificação de confiança baseada em entropia.
#
#     Retorna:
#     --------
#     Tuple[pd.DataFrame, List[int], List[int]]
#         - DataFrame com rótulos previstos e métricas
#         - Lista de índices rotulados como classe 0 (FIRST)
#         - Lista de índices rotulados como classe 1 (LAST)
#
#     Levanta:
#     --------
#     ValueError: Se parâmetros inválidos forem fornecidos.
#     FileNotFoundError: Se arquivo de matriz não existir.
#     """
#     # print(elementos_proximos_centros)
#     # Validação de parâmetros
#     if algoritmo not in ['label_propagation', 'label_spreading']:
#         raise ValueError("Algoritmo deve ser 'label_propagation' ou 'label_spreading'")
#
#     if not os.path.exists(matriz_dist_path):
#         raise FileNotFoundError(f"Arquivo {matriz_dist_path} não encontrado")
#
#     # 1. Carregamento e pré-processamento da matriz de distância
#     dist_df = pd.read_csv(matriz_dist_path, index_col=0, sep=';', decimal=',')
#     dist_df = dist_df.map(lambda x: float(str(x).replace(',', '.'))).astype(float)
#
#     # 2. Preparação dos rótulos iniciais
#     def extrair_indices_rotulados(chave: str) -> List[int]:
#         return resultados[chave]['df_foco'].loc[
#             resultados[chave]['df_foco'].index.isin(elementos_proximos_centros[chave])
#         ]['index'].tolist()
#
#     labeled_indices = {
#         'first': extrair_indices_rotulados('FIRST'),
#         'last': extrair_indices_rotulados('LAST')
#     }
#
#     # print(f'FIRST - {len(labeled_indices['first'])} amostras: {labeled_indices['first']}')
#     # print(f'LAST - {len(labeled_indices['last'])} amostras: {labeled_indices['last']}')
#
#     # 3. Vetor de rótulos (-1 = não rotulado)
#     labels = np.full(len(df), -1, dtype=int)
#     np.put(labels, labeled_indices['first'], 0)
#     np.put(labels, labeled_indices['last'], 1)
#
#     # 4. Construção do kernel personalizado
#     D = dist_df.to_numpy()
#     if matriz_sim:
#         similarity_matrix = pd.read_csv(matriz_sim_path, sep=';', decimal=',', index_col=0).to_numpy()
#     else:
#         similarity_matrix = np.exp(-np.square(D)/(2*sigma**2))
#     # kernel = lambda X, Y=None: similarity_matrix  # Kernel pré-computado
#
#     # # 5. Configuração do modelo
#     # model_params = {
#     #     'kernel': kernel,
#     #     # 'max_iter': 1000,
#     #     'n_jobs': -1
#     # }
#
#     def custom_kernel(X, Y=None):
#       return similarity_matrix
#
#
#     if algoritmo == 'label_spreading':
#         model = LabelSpreading(kernel=custom_kernel, alpha=alpha, n_jobs=-1, max_iter=100)
#         pasta_saida = 'label_spreading'
#     else:
#         model = LabelPropagation(kernel=custom_kernel, n_jobs=-1)
#         pasta_saida = 'label_propagation'
#
#     # 6. Treinamento com matriz dummy (formato exigido pelo scikit-learn)
#     X_dummy = np.zeros((len(df), 1))  # Matriz dummy
#     model.fit(X_dummy, labels)
#
#     # import pdb
#     # pdb.set_trace()  # Coloque aqui antes de chamadas suspeitas
#
#     # 7. Pós-processamento dos resultados
#     df = df.assign(
#         **{
#             f'predicted_label_{qq}_quantis': model.transduction_,
#             f'margem_{qq}_quantis': np.abs(model.label_distributions_[:,0] - model.label_distributions_[:,1]),
#             f'entropia_{qq}_quantis': -np.sum(model.label_distributions_ * np.log2(model.label_distributions_ + 1e-12), axis=1)
#         }
#     )
#
#     # 8. Classificação de confiança
#     margens = df[f'margem_{qq}_quantis']
#     entropias = df[f'entropia_{qq}_quantis']
#
#     df[f'status_margem_{qq}_quantis'] = np.select(
#         [
#             margens > limiar_margem_forte,
#             margens < limiar_margem_fraca
#         ],
#         ['Confiança Forte', 'Confiança Fraca'],
#         default='Confiança Média'
#     )
#
#     df[f'status_entropia_{qq}_quantis'] = np.select(
#         [
#             entropias < limiar_entropia_forte,
#             entropias > limiar_entropia_fraca
#         ],
#         ['Confiança Forte', 'Confiança Fraca'],
#         default='Confiança Média'
#     )
#
#     # 9. Salvamento opcional
#     # if salvar_resultados:
#     #     os.makedirs(pasta_saida, exist_ok=True)
#     #     nome_arquivo = f'rotulos_{algoritmo}_sigma_{sigma}_alpha_{alpha}_quantis_{qq}.csv'
#     #     df.to_csv(
#     #         os.path.join(pasta_saida, nome_arquivo),
#     #         sep=';',
#     #         decimal=',',
#     #         encoding='utf-8-sig'
#     #     )
#
#     return df.copy(), labeled_indices['first'], labeled_indices['last'], similarity_matrix, grupos, elementos_proximos_centros


def processar_rotulos_qq(args):
    """
    Processa todos os cálculos para um determinado valor de 'qq'.
    Esta função é projetada para ser executada em um processo separado.
    Ao final, será gerado um arquivo com os rotulos para cada pertubação, valor de sigma e quantil analisados.
    LEMBRANDO QUE O INDICE 0 NA COLUNA PERTUBACAO REPRESENTA OS DADOS ORIGINAIS, OU SEJA, A ROTULACAO USANDO
    A MATRIZ DE SIMILARIDADE ORIGINAL
    """
    # Desempacota os argumentos recebidos
    qq,  df_original, matriz_distance_name, algoritmo, sigmaNoise, qq_inicial, sigma_inicial, position = args

    # Adiciona um pequeno atraso com base na posição para evitar que as barras de progresso
    # se sobreponham no início.
    time.sleep(position * 0.1)

    print(f'Processo para qq={qq} iniciado...')

    # Cria a pasta de saída para este 'qq' se ela não existir
    output_folder = os.path.join(os.getcwd(), f'{qq}_quantis')
    os.makedirs(output_folder, exist_ok=True)

    # Define o nome do arquivo de saída dentro da pasta específica
    output_csv_filename = os.path.join(output_folder, f'resultados_rotulos_{qq}_quantis.csv')

    # Verifica se o arquivo de saída JÁ EXISTE para continuar de onde parou.
    header_escrito = os.path.exists(output_csv_filename)

    # Determina o valor inicial de sigma para este 'qq' específico.
    # Isso permite retomar um processamento que falhou.
    sigma_start_value = sigma_inicial if qq == qq_inicial else 0.01

    # Loop principal para os valores de sigma
    sigma_range = np.arange(sigma_start_value, 0.31, 0.01)

    # A barra de progresso (tqdm) recebe uma 'position' para que cada processo
    # tenha sua própria linha no console, evitando sobreposição.
    for sigma in tqdm(sigma_range, desc=f'Progresso qq={qq}', position=position, leave=True):
        try:
            sigma = np.round(sigma, decimals=2)

            # --- PASSO 1: Rotular os dados originais (perturbação = 0) ---
            sim_folder_path = os.path.join(os.getcwd(), 'matrizes_similaridades', f'{qq}_quantis')
            # filename_sim = f'matriz_similaridade_{sigma:.2f}_sigma_{qq}_quantis.csv'
            filename_sim = f'matriz_similaridade_original_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}.csv'

            # NOTA: A chamada da função 'phyil' foi mantida como no seu original.
            # Certifique-se de que ela seja 'thread-safe' se modificar dados globais.
            # Passar 'df_original.copy()' já é uma boa prática.
            df_rotulado, _, _, _, _, _, _ = rotular_por_quantil(
                qq, df_original.copy(), load_results_from_csv, utils,
                sigma=sigma, matriz_dist_path=os.path.join(os.getcwd(), matriz_distance_name),
                algoritmo=algoritmo, matriz_sim=True, matriz_sim_path=os.path.join(sim_folder_path, filename_sim)
            )
            rotulos_mat_original = df_rotulado[f'predicted_label_{qq}_quantis']

            # Monta e salva a linha para a matriz ORIGINAL
            linha_original = {'sigma': sigma, 'qq': qq, 'perturbacao': 0}
            for i, rotulo in enumerate(rotulos_mat_original.values):
                linha_original[f'rotulo_{i}'] = rotulo
            df_linha_original = pd.DataFrame([linha_original])

            if not header_escrito:
                df_linha_original.to_csv(output_csv_filename, mode='w', header=True, index=False, sep=';', decimal=',')
                header_escrito = True
            else:
                df_linha_original.to_csv(output_csv_filename, mode='a', header=False, index=False, sep=';', decimal=',')

            # Loop 3: Itera sobre as perturbações.
            for pert in range(1, 101):
                # Rotula os dados com a matriz PERTURBADA
                pert_folder_path = os.path.join(os.getcwd(), 'matrizes_margens_perturbadas', f'{qq}_quantis')
                filename_pert = f'matriz_pertubada_{qq}_quantis_{algoritmo}_sigma_{sigma}_sigmaNoise_{sigmaNoise}_pert_{pert}.csv'

                df_rotulado_pert, _, _, _, _, _, _ = rotular_por_quantil(
                    qq, df_original.copy(), load_results_from_csv, utils,
                    sigma=sigma, matriz_dist_path=os.path.join(os.getcwd(), matriz_distance_name),
                    algoritmo=algoritmo, matriz_sim=True, matriz_sim_path=os.path.join(pert_folder_path, filename_pert)
                )
                rotulos_mat_pert = df_rotulado_pert[f'predicted_label_{qq}_quantis']

                # Monta e salva a linha para a matriz PERTURBADA
                linha_pert = {'sigma': sigma, 'qq': qq, 'perturbacao': pert}
                for i, rotulo in enumerate(rotulos_mat_pert.values):
                    linha_pert[f'rotulo_{i}'] = rotulo
                df_linha_pert = pd.DataFrame([linha_pert])
                df_linha_pert.to_csv(output_csv_filename, mode='a', header=False, index=False, sep=';', decimal=',')

        except Exception as e:
            error_message = f"\n!!!!!! ERRO no processo qq={qq} com sigma={sigma}: {e} !!!!!!"
            print(error_message)
            with open("log_de_erros.txt", "a") as f:
                f.write(error_message + "\n")
            continue

    return f"Processo para qq={qq} concluído com sucesso!"


def salvar_matriz_similaridade(
    matriz_dist_path: str,
    sigma: float,
    matriz_sim_path: str
) -> None:
    """
    Calcula e salva a matriz de similaridade baseada na matriz de distância para um dado sigma.

    Parâmetros:
    -----------
    matriz_dist_path : str
        Caminho para arquivo CSV contendo a matriz de distâncias.
    sigma : float
        Parâmetro do kernel Gaussiano.
    matriz_sim_path : str
        Caminho para salvar a matriz de similaridade calculada.

    Levanta:
    --------
    FileNotFoundError: Se o arquivo da matriz de distância não existir.
    """

    # Verifica se a matriz de distância existe
    if not os.path.exists(matriz_dist_path):
        raise FileNotFoundError(f"Arquivo {matriz_dist_path} não encontrado")

    # Carrega a matriz de distância
    dist_df = pd.read_csv(matriz_dist_path, index_col=0, sep=';', decimal=',')
    dist_df = dist_df.map(lambda x: float(str(x).replace(',', '.'))).astype(float)

    # Calcula a matriz de similaridade usando kernel Gaussiano
    similarity_matrix = np.exp(-np.square(dist_df.to_numpy()) / (2 * sigma**2))

    # Converte para DataFrame para salvamento
    sim_df = pd.DataFrame(similarity_matrix, index=dist_df.index, columns=dist_df.columns)

    # Salva a matriz de similaridade
    sim_df.to_csv(matriz_sim_path, sep=';', decimal=',', encoding='utf-8-sig')

    # print(f"Matriz de similaridade salva em {matriz_sim_path}")

# def rotular_por_quantil(
#     qq,
#     df,
#     load_results_from_csv,
#     utils,
#     sigma=1.0,
#     matriz_dist_path='matrizDistanciaLevenshteinBaseInteira.csv',
#     algoritmo='label_spreading',    # 'label_spreading' ou 'label_propagation'
#     alpha=0.1,                       # Parâmetro para LabelSpreading
#     matriz_sim = False,
#     matriz_sim_path = None
# ):
#     """
#     Rotula o DataFrame df para o quantil qq usando label spreading ou label propagation,
#     salva margens, entropias e status. Retorna o DataFrame rotulado.
#
#     Parâmetros:
#         qq: identificador do quantil
#         df: DataFrame completo a ser rotulado
#         load_results_from_csv: função que carrega resultados pré-computados
#         utils: módulo com funções damicore() e chooseSamplesToLabel()
#         sigma: largura da gaussiana para calcular similaridade
#         matriz_dist_path: caminho para a matriz de distância CSV
#         algoritmo: 'label_spreading' ou 'label_propagation'
#         alpha: parâmetro para LabelSpreading (ignoradono LabelPropagation)
#     """
#     # Carrega matriz de distancia, df_foco, etc
#     results = load_results_from_csv(qq)
#
#     # Escolha de amostras iniciais
#     tipoLabelSample = 'NodeInFilogramCenter'
#     # elementos_prox_centros = {}
#     # grupos = {}
#
#
#     n_neighbors = [0]
#     ADCs = ['fastgreedy']
#     for algo_deteccao in ADCs:
#         resolutions = [1.0] if algo_deteccao == 'fastgreedy' else [0.25, 0.5, 0.75, 1.0]
#         for n_neighbor in n_neighbors:
#             for resolution in resolutions:
#                 grupos, grupos_all_nodes, trees, elementos_prox_centros = phyil(results,
#                                                                               n_neighbor,
#                                                                               algo_deteccao,
#                                                                               resolution,
#                                                                               tipoLabelSample,
#                                                                               utils)
#
#     return difusao_rotulos(
#         df.copy(),
#         matriz_dist_path = matriz_dist_path,
#         resultados = results,
#         elementos_proximos_centros = elementos_prox_centros,
#         algoritmo = algoritmo,
#         qq = qq,
#         sigma=sigma,
#         alpha=alpha,
#         matriz_sim = matriz_sim,
#         matriz_sim_path = matriz_sim_path,
#         grupos = grupos)

def rotular_por_quantil(
        qq,
        df,
        load_results_from_csv,
        utils,
        sigma=1.0,
        matriz_dist_path='matrizDistanciaLevenshteinBaseInteira.csv',
        algoritmo='label_spreading',  # 'label_spreading' ou 'label_propagation'
        alpha=0.1,  # Parâmetro para LabelSpreading
        matriz_sim=False,
        matriz_sim_path=None
):
    """
    Rotula o DataFrame df para o quantil qq.
    Implementa cache GRANULAR: cada chamada à função phyil com uma combinação
    única de parâmetros é salva/carregada de um arquivo de cache específico.
    """
    # Carrega os resultados iniciais que são input para o phyil
    results = load_results_from_csv(qq)

    # Inicializa as variáveis que serão preenchidas dentro do loop.
    # Elas serão usadas APÓS o loop pela função difusao_rotulos.
    grupos, grupos_all_nodes, trees, elementos_prox_centros = {}, {}, {}, {}

    tipoLabelSample = 'NodeInFilogramCenter'
    cache_dir = "phyil_cache"
    # print(os.getcwd())
    os.makedirs(cache_dir, exist_ok=True)

    # Definição dos parâmetros dos loops
    n_neighbors_list = [0]
    ADCs_list = ['fastgreedy']

    # Início dos loops
    for algo_deteccao in ADCs_list:
        resolutions_list = [1.0] if algo_deteccao == 'fastgreedy' else [0.25, 0.5, 0.75, 1.0]
        for n_neighbor in n_neighbors_list:
            for resolution in resolutions_list:

                # --- INÍCIO DO BLOCO DE CACHE GRANULAR (DENTRO DO LOOP) ---

                # 1. Construir nome do arquivo para a iteração ATUAL
                sigma_str = str(sigma)
                alpha_str = str(alpha)
                resolution_str = str(resolution)

                # por definicao minha, os resultados serão gerados usando fastgreedy
                # entao nao tem o parametro resolucao nem vizinhos.
                # vizinhos nao tem por que o damicore esta sendo executado sobre uma matriz de distancia ja calculada.
                # entao a definicao de vizinhos devo ocorrer no calculo da distancia e nao na detecção de comunidades
                # feita pelo damicore nem no NJ.
                cache_filename = (
                    f"phyil_results_qq_{qq}.pkl"
                )
                cache_filepath = os.path.join(cache_dir, cache_filename)

                # print(f"Caching {cache_filepath}")

                # 2. Verificar, carregar ou calcular e salvar para ESTA iteração
                if os.path.exists(cache_filepath):
                    # Se o arquivo para esta combinação de parâmetros existe, carrega-o
                    # print(f"Carregando do cache: {os.path.basename(cache_filepath)}")
                    with open(cache_filepath, 'rb') as f:
                        grupos, grupos_all_nodes, trees, elementos_prox_centros = pickle.load(f)
                else:
                    # Se não existe, executa o cálculo
                    # print(f"Calculando e salvando em cache: {os.path.basename(cache_filepath)}")

                    grupos, grupos_all_nodes, trees, elementos_prox_centros = phyil(
                        results,
                        n_neighbor,
                        algo_deteccao,
                        resolution,
                        tipoLabelSample,
                        utils
                    )

                    # Salva o resultado desta iteração no seu arquivo de cache específico
                    data_to_save = (grupos, grupos_all_nodes, trees, elementos_prox_centros)
                    with open(cache_filepath, 'wb') as f:
                        pickle.dump(data_to_save, f)

                # --- FIM DO BLOCO DE CACHE GRANULAR ---

    # --- Ponto de Atenção ---
    # A estrutura atual dos seus laços faz com que as variáveis (grupos, trees, etc.)
    # sejam sobrescritas a cada iteração. Portanto, mesmo com o cache granular,
    # apenas os resultados da ÚLTIMA combinação de parâmetros serão passados
    # para a função 'difusao_rotulos'.
    # O cache funcionará para acelerar as execuções, mas o comportamento de
    # sobrescrita é mantido.
    # O codigo foi mantido assim pois nao esta havendo variação destes parametros, atualmente.

    return difusao_rotulos(
        df.copy(),
        matriz_dist_path=matriz_dist_path,
        resultados=results,
        elementos_proximos_centros=elementos_prox_centros,
        algoritmo=algoritmo,
        qq=qq,
        sigma=sigma,
        alpha=alpha,
        matriz_sim=matriz_sim,
        matriz_sim_path=matriz_sim_path,
        grupos=grupos
    )

def perturbar_grafo(
    W: np.ndarray,
    n_perturbacoes: int = 50,
    sigma_noise: float = 0.1
) -> np.ndarray:
    """
    Perturba a matriz de similaridade com ruído gaussiano,
    tomando o desvio-padrão apenas dos valores off-diagonal.
    
    Parâmetros:
        W             – matriz quadrada de similaridade.
        n_perturbacoes– número de elementos a perturbar.
        sigma_noise   – fração do desvio-padrão off-diagonal a usar.
    
    Retorna:
        Matriz perturbada (mesma forma de W).
    """
    # 1. Verifica quadratura e obtém dimensão
    n_rows, n_cols = W.shape
    if n_rows != n_cols:
        raise ValueError("Matriz W deve ser quadrada")
    
    # 2. Calcula o desvio-padrão off-diagonal
    mask_off = ~np.eye(n_rows, dtype=bool)    # True para i≠j
    std_off = np.std(W[mask_off])             # std apenas dos off-diagonais
    
    # 3. Seleciona índices únicos para perturbar
    total = n_rows * n_cols
    pert_idx = np.random.choice(
        total,
        size=min(n_perturbacoes, total),
        replace=False
    )
    rows, cols = np.unravel_index(pert_idx, W.shape)
    
    # 4. Gera ruído gaussiano usando std_off
    noise = np.random.normal(0, sigma_noise * std_off, size=len(pert_idx))
    
    # 5. Aplica ruído e recorta valores no intervalo [0,1]
    W_pert = W.copy()
    W_pert[rows, cols] = np.clip(W[rows, cols] + noise, 0, 1)
    
    return W_pert


def perturbar_grafo_antiga(W, n_perturbacoes=50, sigma_noise=0.1):
    """Perturba a matriz de similaridade com ruído gaussiano."""
    
    n_rows, n_cols = W.shape  # Correção crítica: shape retorna (linhas, colunas)
    total_elements = n_rows * n_cols
    
    # Gera índices únicos sem reposição
    perturb_indices = np.random.choice(
        total_elements, 
        size=min(n_perturbacoes, total_elements),  # Evita oversampling
        replace=False
    )
    
    rows, cols = np.unravel_index(perturb_indices, W.shape)
    noise = np.random.normal(0, sigma_noise * np.std(W), len(perturb_indices))
    
    W_pert = W.copy()
    W_pert[rows, cols] = np.clip(W[rows, cols] + noise, 0, 1)
    
    
    return W_pert

def carregar_matriz(caminho):
    """Carrega e valida a matriz de distância com tratamento robusto de erros."""
    try:
        df = pd.read_csv(
            caminho,
            index_col=0,
            header=0,
            decimal=',',
            sep=';',
            thousands='.',  # Adiciona tratamento para separador de milhares
            na_values=['-', 'NaN', ' ', ''],
            dtype=np.float64  # Força tipo numérico
        )
        
        if df.empty:
            raise ValueError("Arquivo vazio ou sem dados válidos")
            
        if df.shape[0] != df.shape[1]:
            raise ValueError("Matriz não é quadrada")
            
        return df.fillna(0).astype(np.float64)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {caminho} não encontrado") from None
    except pd.errors.ParserError:
        raise ValueError("Formato de arquivo inválido") from None


def gerar_matrizes_perturbadas(
    matriz_dist_path: str,
    sigma: float = 0.02,
    num_matriz_Pert: int = 100,
    num_pert: int = 100,
    sigma_noise: float = 0.5
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Gera matrizes de similaridade perturbadas a partir de uma matriz de distâncias.
    
    Parâmetros:
        matriz_dist_path (str): Caminho para o arquivo CSV com a matriz de distâncias
        sigma (float): Parâmetro de largura para o kernel RBF (default: 0.02)
        num_matriz_Pert (int): Número de matrizes perturbadas a gerar (default: 100)
        num_pert (int): Número de perturbações por matriz (default: 100)
    
    Retorna:
        Lista de matrizes NumPy perturbadas
    
    Levanta:
        ValueError: Se sigma não for positivo ou ocorrer erro no carregamento
    """
    try:
        
        # 1. Carregamento da matriz de distâncias
        dist_df = carregar_matriz(matriz_dist_path)
        
        # 2. Validação dos parâmetros
        if sigma <= 0:
            raise ValueError("Sigma deve ser positivo")
            
        # 3. Cálculo da matriz de similaridade
        D = dist_df.to_numpy()
        similarity_matrix = np.exp(-np.square(D) / (2 * sigma**2))
        
        # 4. Geração das matrizes perturbadas
        print(similarity_matrix.shape)
        return [
            perturbar_grafo(similarity_matrix, num_pert, sigma_noise=sigma_noise)
            for _ in tqdm(range(num_matriz_Pert), desc="Gerando matrizes perturbadas")
        ], similarity_matrix
        
    except Exception as e:
        raise ValueError(f"Falha na geração de matrizes: {str(e)}") from e

def delete_files_in_folder(folder_path):
  """Deletes all files within a specified folder.

  Args:
    folder_path: The path to the folder.
  """
  print(f"Deleting files in: {folder_path}")
  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
      if os.path.isfile(file_path):
        os.remove(file_path)
    except OSError as e:
      print(f"Error deleting {file_path}: {e}")


def processar_perturbacao(idx, W_pert, sigma, qq, df_original, load_results_from_csv, utils, algoritmo, output_dir, save):
    # 2.1 Salva temporariamente a matriz perturbada
    temp_path = os.path.join(output_dir, f'temp_W_pert_sigma_{sigma}_idx_{idx}.csv')
    pd.DataFrame(W_pert).to_csv(temp_path, sep=';', decimal=',')

    # 2.2 Calcula as margens
    df_rotulado, _, _, sim_matrix  = rotular_por_quantil(
        qq,
        df_original.copy(),
        load_results_from_csv,
        utils,
        sigma=sigma,
        matriz_dist_path=temp_path,
        algoritmo=algoritmo,
        matriz_sim = True,
        matriz_sim_path = temp_path
    )

    # 2.3 Extrai margens (96 valores)
    margens = df_rotulado[f'margem_{qq}_quantis'].values

    # 2.4 Cria DataFrame com linha única
    df_margens = pd.DataFrame([margens], columns=[f'margem_{i}' for i in range(1, 97)])
    df_margens['sigma'] = sigma  # Adiciona coluna sigma

    # 2.5 Salva arquivo final
    output_file = os.path.join(
        output_dir,
        f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_pert_{idx}.csv'
    )
    if save:
        df_margens.to_csv(output_file, index=False, sep=';', decimal=',')

    # 2.6 Remove arquivo temporário
    os.remove(temp_path)
    return df_rotulado, sim_matrix

def gerar_matrizes_margens_por_sigma_paralela(
    qq: int,
    algoritmo: str,
    sigmas: list,
    df_original: pd.DataFrame,
    load_results_from_csv=None,
    utils=None,
    num_matriz_Pert: int = 100,
    num_pert: int = 100,
    output_dir: str = 'matrizes_margens',
    matriz_dist_path ='matrizDistanciaLevenshteinBaseInteira.csv',
    sigma_noise: float = 0.5,
    save: bool = True,
    n_jobs: int = -1  # Número de processos paralelos, -1 usa todos os núcleos
):
    """
    NAO FUNCIONA AINDA. TEM QUE GERAR UM OBJETO PHYIL PARA CADA PROCESSO DE DIFUSAO. USAR POR ENQUANTO A VERSAO SE SER PARALELA
    Para cada sigma e matriz perturbada, gera um CSV com as margens correspondentes.
    """
    if save:
        os.makedirs(output_dir, exist_ok=True)
        delete_files_in_folder(output_dir)

    for sigma in tqdm(sigmas):
        # 1. Gera matrizes perturbadas para este sigma
        matrizes_pert, sim_matrix_ori = gerar_matrizes_perturbadas(
            matriz_dist_path=matriz_dist_path,
            sigma=sigma,
            num_matriz_Pert=num_matriz_Pert,
            num_pert=num_pert
        )
        output_file = os.path.join(
            output_dir,
            f'matriz_similaridade_original_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}.csv'
        )
        pd.DataFrame(sim_matrix_ori).to_csv(output_file, index=True, sep=';', decimal=',')

        # 2. Processa cada matriz perturbada em paralelo
        results = Parallel(n_jobs=n_jobs)(
            delayed(processar_perturbacao)(
                idx, W_pert, sigma, qq, df_original, load_results_from_csv, utils, algoritmo, output_dir, save
            )
            for idx, W_pert in enumerate(matrizes_pert, start=1)
        )

        # Se quiser retornar o último resultado para manter compatibilidade:
        df_rotulado, sim_matrix = results[-1]

    return df_rotulado, sim_matrix_ori


# def gerar_matrizes_margens_por_sigma(
#     qq: int,
#     algoritmo: str,
#     sigmas: List[float],
#     df_original: pd.DataFrame,
#     load_results_from_csv,
#     utils,
#     num_matriz_Pert: int = 100,
#     num_pert: int = 100,
#     output_dir: str = 'matrizes_margens',
#     matriz_dist_path ='matrizDistanciaLevenshteinBaseInteira.csv',
#     sigma_noise: float = 0.5,
#     alpha = 0.1,
#     save: bool = True
# ):
#     """
#     Para cada sigma e matriz perturbada, gera um CSV com as margens correspondentes.
    
#     Parâmetros:
#         qq (int): Número de quantis
#         algoritmo (str): 'label_propagation' ou 'label_spreading'
#         sigmas (List[float]): Lista de valores de sigma a serem testados
#         df_original (pd.DataFrame): DataFrame com dados originais
#         load_results_from_csv: Função de carregamento de resultados
#         utils: Módulo com funções auxiliares
#         num_matriz_Pert (int): Número de matrizes perturbadas por sigma
#         num_pert (int): Número de perturbações por matriz
#         output_dir (str): Diretório de saída
#     """
#     if save == True:
#       os.makedirs(output_dir, exist_ok=True)
#       delete_files_in_folder(output_dir)

#     print('\nIniciando processamento...\n')
#     for sigma in tqdm(sigmas):
        
#         # 1. Gera matrizes de similaridades perturbadas para este sigma
#         matrizes_pert, sim_matrix_ori = gerar_matrizes_perturbadas(
#             matriz_dist_path,
#             sigma=sigma,
#             num_matriz_Pert=num_matriz_Pert,
#             num_pert=num_pert,
#             sigma_noise = sigma_noise
#         )
#         output_file = os.path.join(
#                 output_dir,  f'matriz_similaridade_original_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}.csv'
#             )
#         pd.DataFrame(sim_matrix_ori).to_csv(output_file, index=True, sep=';', decimal=',')
        
#         # 2. Processa cada matriz perturbada
#         for idx, W_pert in enumerate(matrizes_pert, start=1):
            
#             # 2.1 Salva temporariamente a matriz perturbada
#             temp_path = os.path.join(output_dir, f'matriz_pertubada_{qq}_quantis_{algoritmo}_sigma_{sigma}_sigmaNoise_{sigma_noise}_ pert_{idx}.csv')
#             pd.DataFrame(W_pert).to_csv(temp_path, sep=';', decimal=',')

#             # 2.2 Calcula as margens
#             # aqui esta usando matriz_sim = True, pois W_pert sao pertubacoes da
#             # matriz de similaridade e nao da matriz de distancia.
#             df_rotulado, _, _, sim_matrix  = rotular_por_quantil(
#                 qq,
#                 df_original.copy(),
#                 load_results_from_csv,
#                 utils,
#                 sigma=sigma,
#                 # aqui é irrelevante o caminho da matriz_diz.
#                 matriz_dist_path=temp_path,
#                 algoritmo=algoritmo,
#                 alpha = alpha,
#                 matriz_sim = True,
#                 matriz_sim_path = temp_path
#             )


#             # 2.3 Extrai margens (96 valores)
#             margens = df_rotulado[f'margem_{qq}_quantis'].values

#             # 2.4 Cria DataFrame com linha única
#             df_margens = pd.DataFrame([margens], columns=[f'margem_{i}' for i in range(1, 97)])
#             df_margens['sigma'] = sigma  # Adiciona coluna sigma

#             # 2.5 Salva arquivo final
#             output_file = os.path.join(
#                 output_dir,
#                 f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_pert_{idx}.csv'
#             )
#             if save == True:
#               df_margens.to_csv(output_file, index=False, sep=';', decimal=',')

#             # 2.6 Remove arquivo temporário
#             # os.remove(temp_path)
#     return df_rotulado, sim_matrix_ori

# Dentro do seu arquivo phyil.py, substitua a função existente por esta:

# Dentro do seu arquivo phyil.py, substitua a função existente por esta:

# VERSAO ANTIGA, USADA ATE 21.10.25
# def gerar_matrizes_margens_por_sigma(
#         qq: int,
#         algoritmo: str,
#         sigmas: list,
#         df_original: 'pd.DataFrame',
#         load_results_from_csv,
#         utils,
#         num_matriz_Pert: int = 100,
#         num_pert: int = 100,
#         output_dir: str = 'matrizes_margens',
#         matriz_dist_path: str = 'matrizDistanciaLevenshteinBaseInteira.csv',
#         # Este é o caminho que precisamos passar adiante
#         sigma_noise: float = 0.5,
#         alpha=0.1,
#         save: bool = True
# ):
#     """
#     Para cada sigma, gera matrizes de similaridade (original e perturbadas),
#     executa a rotulagem e salva tanto as MARGENS quanto os RÓTULOS em arquivos CSV.
#     """
#     if save:
#         os.makedirs(output_dir, exist_ok=True)
#
#     rotulos_output_path = os.path.join(output_dir, f'resultados_rotulos_{qq}_quantis.csv')
#     header_escrito = os.path.exists(rotulos_output_path)
#
#     print(f'\nIniciando processamento para qq={qq}...\n')
#     for sigma in tqdm(sigmas, desc=f"Processando Sigmas para qq={qq}"):
#         sigma = round(sigma, 2)
#
#         matrizes_pert, sim_matrix_ori = gerar_matrizes_perturbadas(
#             matriz_dist_path,  # Usa o caminho completo aqui
#             sigma=sigma,
#             num_matriz_Pert=num_matriz_Pert,
#             num_pert=num_pert,
#             sigma_noise=sigma_noise
#         )
#
#         output_file_ori = os.path.join(output_dir,
#                                        f'matriz_similaridade_original_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}.csv')
#         pd.DataFrame(sim_matrix_ori).to_csv(output_file_ori, index=True, sep=';', decimal=',')
#
#         if save:
#             # print(f"  - Rotulando matriz original para sigma={sigma:.2f}")
#             df_rotulado_original, _, _, _, _, _ = rotular_por_quantil(
#                 qq, df_original.copy(), load_results_from_csv, utils,
#                 sigma=sigma, algoritmo=algoritmo, alpha=alpha,
#                 matriz_sim=True, matriz_sim_path=output_file_ori,
#                 matriz_dist_path=matriz_dist_path  # ALTERADO: Adicionado o caminho da matriz de distância
#             )
#             rotulos_originais = df_rotulado_original[f'predicted_label_{qq}_quantis']
#
#             linha_original_dict = {'sigma': sigma, 'qq': qq, 'perturbacao': 0}
#             for i, rotulo in enumerate(rotulos_originais):
#                 linha_original_dict[f'rotulo_{i}'] = rotulo
#
#             df_linha_original = pd.DataFrame([linha_original_dict])
#
#             if not header_escrito:
#                 df_linha_original.to_csv(rotulos_output_path, mode='w', header=True, index=False, sep=';', decimal=',')
#                 header_escrito = True
#             else:
#                 df_linha_original.to_csv(rotulos_output_path, mode='a', header=False, index=False, sep=';', decimal=',')
#
#         for idx, W_pert in enumerate(matrizes_pert, start=1):
#             temp_path = os.path.join(output_dir,
#                                      f'matriz_pertubada_{qq}_quantis_{algoritmo}_sigma_{sigma}_sigmaNoise_{sigma_noise}_pert_{idx}.csv')
#             pd.DataFrame(W_pert).to_csv(temp_path, sep=';', decimal=',')
#
#             df_rotulado, _, _, _, _, _ = rotular_por_quantil(
#                 qq, df_original.copy(), load_results_from_csv, utils,
#                 sigma=sigma, algoritmo=algoritmo, alpha=alpha,
#                 matriz_sim=True, matriz_sim_path=temp_path,
#                 matriz_dist_path=matriz_dist_path  # ALTERADO: Adicionado também aqui por segurança
#             )
#
#             margens = df_rotulado[f'margem_{qq}_quantis'].values
#             df_margens = pd.DataFrame([margens], columns=[f'margem_{i}' for i in range(df_original.shape[0])])
#             df_margens['sigma'] = sigma
#
#             if save:
#                 if algoritmo == "label_spreading":
#                     output_file_margem = os.path.join(output_dir,
#                                                       f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_alpha_{alpha:.2f}_pert_{idx}.csv')
#                 else:
#                     output_file_margem = os.path.join(output_dir,
#                                                       f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_pert_{idx}.csv')
#                 df_margens.to_csv(output_file_margem, index=False, sep=';', decimal=',')
#
#             if save:
#                 rotulos_perturbados = df_rotulado[f'predicted_label_{qq}_quantis']
#                 linha_pert_dict = {'sigma': sigma, 'qq': qq, 'perturbacao': idx}
#                 for i, rotulo in enumerate(rotulos_perturbados):
#                     linha_pert_dict[f'rotulo_{i}'] = rotulo
#                 df_linha_pert = pd.DataFrame([linha_pert_dict])
#                 df_linha_pert.to_csv(rotulos_output_path, mode='a', header=False, index=False, sep=';', decimal=',')
#
#     return df_rotulado, sim_matrix_ori

# Dentro da função gerar_matrizes_margens_por_sigma:

def gerar_matrizes_margens_por_sigma(
        qq: int,
        algoritmo: str,
        sigmas: list,
        df_original: 'pd.DataFrame',
        load_results_from_csv,
        utils,
        num_matriz_Pert: int = 100,
        num_pert: int = 100,
        output_dir: str = 'matrizes_margens',
        matriz_dist_path: str = 'matrizDistanciaLevenshteinBaseInteira.csv',
        sigma_noise: float = 0.5,
        alpha=0.1,
        save: bool = True
):
    """
    Para cada sigma, gera matrizes de similaridade (original e perturbadas),
    executa a rotulagem e salva tanto as MARGENS quanto os RÓTULOS e PROBABILIDADES em arquivos CSV.
    """
    if save:
        os.makedirs(output_dir, exist_ok=True)

    rotulos_output_path = os.path.join(output_dir, f'resultados_rotulos_{qq}_quantis.csv')
    header_escrito = os.path.exists(rotulos_output_path)

    print(f'\nIniciando processamento para qq={qq}...\n')
    for sigma in tqdm(sigmas, desc=f"Processando Sigmas para qq={qq}"):
        sigma = round(sigma, 2)

        # Gera matriz original e perturbadas
        matrizes_pert, sim_matrix_ori = gerar_matrizes_perturbadas(
            matriz_dist_path, sigma=sigma, num_matriz_Pert=num_matriz_Pert,
            num_pert=num_pert, sigma_noise=sigma_noise
        )

        # Salva matriz de similaridade original
        output_file_ori = os.path.join(output_dir,
                                       f'matriz_similaridade_original_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}.csv')
        pd.DataFrame(sim_matrix_ori).to_csv(output_file_ori, index=True, sep=';', decimal=',')

        # Processa e salva resultados da matriz ORIGINAL (perturbação 0)
        if save:
            # Roda a rotulagem (que agora retorna df com probs)
            df_rotulado_original, _, _, _, _, _, _ = rotular_por_quantil(
                qq, df_original.copy(), load_results_from_csv, utils,
                sigma=sigma, algoritmo=algoritmo, alpha=alpha,
                matriz_sim=True, matriz_sim_path=output_file_ori,
                matriz_dist_path=matriz_dist_path
            )
            # Extrai rótulos e probabilidades
            rotulos_originais = df_rotulado_original[f'predicted_label_{qq}_quantis']
            probs_classe0_orig = df_rotulado_original[f'prob_classe0_{qq}_quantis'] # NOVO
            probs_classe1_orig = df_rotulado_original[f'prob_classe1_{qq}_quantis'] # NOVO

            # Monta o dicionário da linha
            linha_original_dict = {'sigma': sigma, 'qq': qq, 'perturbacao': 0}
            for i, rotulo in enumerate(rotulos_originais):
                linha_original_dict[f'rotulo_{i}'] = rotulo
                # --- ADICIONADO AQUI ---
                linha_original_dict[f'prob0_{i}'] = probs_classe0_orig.iloc[i] # Adiciona prob classe 0
                linha_original_dict[f'prob1_{i}'] = probs_classe1_orig.iloc[i] # Adiciona prob classe 1
                # -----------------------

            # Salva a linha no CSV
            df_linha_original = pd.DataFrame([linha_original_dict])
            if not header_escrito:
                df_linha_original.to_csv(rotulos_output_path, mode='w', header=True, index=False, sep=';', decimal=',')
                header_escrito = True
            else:
                df_linha_original.to_csv(rotulos_output_path, mode='a', header=False, index=False, sep=';', decimal=',')

        # Processa e salva resultados das matrizes PERTURBADAS
        for idx, W_pert in enumerate(matrizes_pert, start=1):
            # Salva matriz perturbada temporária
            temp_path = os.path.join(output_dir,
                                     f'matriz_pertubada_{qq}_quantis_{algoritmo}_sigma_{sigma}_sigmaNoise_{sigma_noise}_pert_{idx}.csv')
            pd.DataFrame(W_pert).to_csv(temp_path, sep=';', decimal=',')

            # Roda a rotulagem
            df_rotulado, _, _, _, _, _, _ = rotular_por_quantil(
                qq, df_original.copy(), load_results_from_csv, utils,
                sigma=sigma, algoritmo=algoritmo, alpha=alpha,
                matriz_sim=True, matriz_sim_path=temp_path,
                matriz_dist_path=matriz_dist_path
            )

            # Salva as MARGENS (código inalterado)
            margens = df_rotulado[f'margem_{qq}_quantis'].values
            df_margens = pd.DataFrame([margens], columns=[f'margem_{i}' for i in range(df_original.shape[0])])
            df_margens['sigma'] = sigma
            if save:
                if algoritmo == "label_spreading":
                    output_file_margem = os.path.join(output_dir, f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_alpha_{alpha:.2f}_pert_{idx}.csv')
                else:
                    output_file_margem = os.path.join(output_dir, f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_pert_{idx}.csv')
                df_margens.to_csv(output_file_margem, index=False, sep=';', decimal=',')

            # Salva os RÓTULOS e PROBABILIDADES das perturbações
            if save:
                # Extrai rótulos e probabilidades
                rotulos_perturbados = df_rotulado[f'predicted_label_{qq}_quantis']
                probs_classe0_pert = df_rotulado[f'prob_classe0_{qq}_quantis'] # NOVO
                probs_classe1_pert = df_rotulado[f'prob_classe1_{qq}_quantis'] # NOVO

                # Monta o dicionário da linha
                linha_pert_dict = {'sigma': sigma, 'qq': qq, 'perturbacao': idx}
                for i, rotulo in enumerate(rotulos_perturbados):
                    linha_pert_dict[f'rotulo_{i}'] = rotulo
                    # --- ADICIONADO AQUI ---
                    linha_pert_dict[f'prob0_{i}'] = probs_classe0_pert.iloc[i] # Adiciona prob classe 0
                    linha_pert_dict[f'prob1_{i}'] = probs_classe1_pert.iloc[i] # Adiciona prob classe 1
                    # -----------------------

                # Salva a linha no CSV
                df_linha_pert = pd.DataFrame([linha_pert_dict])
                # Note que o header já foi escrito pela linha da matriz original
                df_linha_pert.to_csv(rotulos_output_path, mode='a', header=False, index=False, sep=';', decimal=',')

            # Remove o arquivo temporário (opcional)
            # os.remove(temp_path)

    # Retorna o último df rotulado e a matriz original (comportamento inalterado)
    return df_rotulado, sim_matrix_ori

# def gerar_matrizes_margens_por_sigma(
#     qq: int,
#     algoritmo: str,
#     sigmas: list,
#     df_original: 'pd.DataFrame',
#     load_results_from_csv,
#     utils,
#     num_matriz_Pert: int = 100,
#     num_pert: int = 100,
#     output_dir: str = 'matrizes_margens',
#     matriz_dist_path: str = 'matrizDistanciaLevenshteinBaseInteira.csv',
#     sigma_noise: float = 0.5,
#     alpha = 0.1,
#     save: bool = True
# ):
#     """
#     Para cada sigma e matriz perturbada, gera um CSV com as margens correspondentes.
#     O nome do arquivo de margens inclui o valor de alpha se o algoritmo for label_spreading.
#     """
#     if save:
#         os.makedirs(output_dir, exist_ok=True)
#         delete_files_in_folder(output_dir)
#
#     print('\nIniciando processamento...\n')
#     for sigma in tqdm(sigmas):
#         # 1. Gera matrizes de similaridades perturbadas para este sigma
#
#         sigma = round(sigma,2)
#         matrizes_pert, sim_matrix_ori = gerar_matrizes_perturbadas(
#             matriz_dist_path,
#             sigma=sigma,
#             num_matriz_Pert=num_matriz_Pert,
#             num_pert=num_pert,
#             sigma_noise=sigma_noise
#         )
#         output_file_ori = os.path.join(
#             output_dir, f'matriz_similaridade_original_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}.csv'
#         )
#         pd.DataFrame(sim_matrix_ori).to_csv(output_file_ori, index=True, sep=';', decimal=',')
#
#         # 2. Processa cada matriz perturbada
#         for idx, W_pert in enumerate(matrizes_pert, start=1):
#             temp_path = os.path.join(
#                 output_dir,
#                 f'matriz_pertubada_{qq}_quantis_{algoritmo}_sigma_{sigma}_sigmaNoise_{sigma_noise}_pert_{idx}.csv'
#             )
#             # Só salva a matriz perturbada se não existir
#             if not os.path.exists(temp_path):
#                 pd.DataFrame(W_pert).to_csv(temp_path, sep=';', decimal=',')
#
#             # Define o nome do arquivo de margens conforme o algoritmo
#             if algoritmo == "label_spreading":
#                 output_file_margem = os.path.join(
#                     output_dir,
#                     f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_alpha_{alpha:.2f}_pert_{idx}.csv'
#                 )
#             else:
#                 output_file_margem = os.path.join(
#                     output_dir,
#                     f'matriz_margens_completa_{qq}_quantis_{algoritmo}_sigma_{sigma:.2f}_pert_{idx}.csv'
#                 )
#
#             # Só calcula e salva as margens se o arquivo ainda não existir
#             if not os.path.exists(output_file_margem):
#
#                 df_rotulado, _, _, sim_matrix, _, _ = rotular_por_quantil(
#                     qq,
#                     df_original.copy(),
#                     load_results_from_csv,
#                     utils,
#                     sigma=sigma,
#                     matriz_dist_path=temp_path,
#                     algoritmo=algoritmo,
#                     alpha=alpha,
#                     matriz_sim=True,
#                     matriz_sim_path=temp_path
#                 )
#
#                 margens = df_rotulado[f'margem_{qq}_quantis'].values
#                 df_margens = pd.DataFrame([margens], columns=[f'margem_{i}' for i in range(1, df_original.shape[0]+1)])
#                 df_margens['sigma'] = sigma
#
#                 if save:
#                     df_margens.to_csv(output_file_margem, index=False, sep=';', decimal=',')
#
#                 # os.remove(temp_path)  # Descomente se quiser remover o temporário
#
#             else:
#                 print(f"Arquivo de margem já existe: {output_file_margem}, pulando cálculo.")
#     return df_rotulado, sim_matrix_ori


def load_results_from_csv(qq):
    results = {}
    for tipo in ['FIRST', 'LAST']:
        dist_df = pd.read_csv(os.path.join(os.getcwd(),f'dist_df_{tipo.lower()}_{qq}_quantis.csv'), sep=';', decimal=',', index_col=0)
        # descomentar se quiser a geodesica
        # dist_df_geodesica = pd.read_csv(os.path.join(os.getcwd(),f'dist_df_geodesica_{tipo.lower()}_{qq}_quantis.csv'), sep=';', decimal=',', index_col=0)
        df_foco = pd.read_csv(os.path.join(os.getcwd(),f'df_foco_{tipo.lower()}_{qq}_quantis.csv'), sep=';', decimal=',', index_col=0)
        # Adicione outros campos se necessário, mas para damicore só precisa dist_df
        results[tipo] = {
            'dist_df': dist_df,
            # descomendar se quiser a geodesica
            # 'dist_df_geodesica': dist_df_geodesica,
            'df_foco': df_foco
        }
    return results


def process_distance_matrices(df, first_quantis, last_quantis, dict_quantiles,
                            metric='levenshtein', num_vizim=5,qnt_quantis=8,
                             distance_matrix=False):
    """
    Usada apenas se as matrizes de distancia ainda nao estiverem salvas em arquivos.
    
    Processa matrizes de distância para subconjuntos 'FIRST' e 'LAST' de um DataFrame.

    Para cada tipo ('FIRST' e 'LAST'), seleciona as amostras correspondentes ao quantil
    indicado, calcula uma matriz de distância (usando Levenshtein ou NCD), normaliza-a,
    e em seguida constrói um grafo geodésico baseado nesta matriz.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame original contendo todas as instâncias indexadas por inteiro.
    first_quantis : dict
        Dicionário que associa cada número de quantis a um DataFrame contendo
        as linhas iniciais ('FIRST') de cada quantil. Sua função é fornecer os ID
        no dataframe para fazer o filtro.
    last_quantis : dict
        Dicionário que associa cada número de quantis a um DataFrame contendo
        as linhas finais ('LAST') de cada quantil.
    dict_quantiles : dict
        Dicionário que mapeia o número de quantis (`qnt_quantis`) à chave usada
        para indexar `first_quantis` e `last_quantis`.
    metric : str, opcional
        Métrica de distância a ser usada:
          - 'levenshtein' : distância de edição.
          - 'NCD'         : Normalized Compression Distance (requer compressão lzma).
        Padrão: 'levenshtein'.
    num_vizim : int, opcional
        Número de vizinhos a considerar ao construir o grafo geodésico (k na KNN).
        Padrão: 5.
    qnt_quantis : int, opcional
        Número de quantis a selecionar nos dicionários `first_quantis`/`last_quantis`.
        Padrão: 8.
    distance_matrix: bool
        Indica se haverá ou não o calculo da matriz de distancia pela função e seu respectivo retorno

    Retorna (se distance_matrix == True)
    -------
    dict
        Dicionário com duas chaves: 'FIRST' e 'LAST'. Cada valor é um sub-dicionário contendo:
            - df_foco               : pandas.DataFrame com as linhas selecionadas de `df`.
            - dist_df               : pandas.DataFrame da matriz de distância normalizada.
            - dist_df_geodesica     : pandas.DataFrame da matriz de distância geodésica.
            - geodesic_knn_sparse   : matriz esparsa de adjacência do grafo geodésico.
            - geodesic_knn          : lista ou estrutura completa do grafo geodésico.
            - geodesic_dist_matrix  : numpy.ndarray da matriz de distâncias geodésicas.

    Exceções
    ---------
    ValueError
        Se `metric` não for 'levenshtein' nem 'NCD'.
    """
    results = {}

    for tipo in ['FIRST', 'LAST']:
        # Seleciona o dataframe com base no tipo
        if tipo == 'FIRST':
            df_quantis = first_quantis[dict_quantiles[qnt_quantis]]
        else:  # tipo == 'LAST'
            df_quantis = last_quantis[dict_quantiles[qnt_quantis]]

        # Extrai os IDs e seleciona as linhas do df
        ids_to_select = df_quantis['ID'].tolist()
        df_foco = df.iloc[ids_to_select].reset_index().copy()

        # Define compressão caso a métrica seja NCD
        compress = 'lzma' if metric == 'NCD' else ''

        if distance_matrix == True:
            # Calcula a matriz de distância
            if metric == 'NCD':
                dist_df = utils.calcular_matriz_ncd_parallel(df_foco, compress)
            elif metric == 'levenshtein':
                dist_df = utils.nld_distance_matrix_parallel(df_foco)
            elif metric == 'euclidiana':
                matriz_dist_all_dataset = utils.matrizDistancia(df_foco.to_numpy(), 'euclidiana', 2)
                dist_df = pd.DataFrame(
                                matriz_dist_all_dataset,
                                index=df_foco.index,
                                columns=df_foco.index
                               )
            else:
                raise ValueError(f"Métrica não suportada: {metric}")
    
            # Normaliza a matriz
            normalized_dist_df = normalize_df(dist_df)
            dist_df_normalized = normalized_dist_df.copy()
            np.fill_diagonal(dist_df_normalized.values, 0)
    
            # Constrói grafo geodésico
            geodesic_knn_sparse, geodesic_knn, geodesic_dist_matrix = utils.geodesic_knn_from_distance_matrix(
                dist_df_normalized.values, n_neighbors=num_vizim
            )
    
            # Cria DataFrame para matriz geodésica
            dist_df_geodesica = pd.DataFrame(geodesic_dist_matrix, columns=dist_df.columns)

            # Salva os resultados no dicionário
            results[tipo] = {
                'df_foco': df_foco,
                'dist_df': dist_df_normalized,
                'dist_df_geodesica': dist_df_geodesica,
                'geodesic_knn_sparse': geodesic_knn_sparse,
                'geodesic_knn': geodesic_knn,
                'geodesic_dist_matrix': geodesic_dist_matrix
            }
    
            print(f"Processamento concluído para tipo: {tipo} com matriz de distância\n")
        else:
            # Salva os resultados no dicionário
            results[tipo] = {
                'df_foco': df_foco,
            }
    
            print(f"Processamento concluído para tipo: {tipo} com df_foco apenas\n")
    return results




def normalize_df(df, ignore_diagonal=True):
    """
    Usanda se o df for uma matriz quadrada.
    
    Normaliza os valores de um DataFrame de distâncias para o intervalo [0.001, 1.0].

    Esta função mapeia linearmente os valores de `df` do seu intervalo original
    para um novo intervalo definido por [min_value, max_value] = [0.001, 1.0].
    Pode opcionalmente ignorar a diagonal principal ao determinar os valores
    mínimo e máximo reais.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo a matriz de distâncias ou quaisquer valores numéricos
        que se deseja normalizar. Deve ser quadrado se `ignore_diagonal=True`.
    ignore_diagonal : bool, opcional
        Se True (padrão), exclui a diagonal principal do cálculo dos valores
        mínimo e máximo, para que zeros na diagonal não distorçam a escala.
        Se False, considera todos os valores do DataFrame.

    Retorna
    -------
    pandas.DataFrame
        Cópia de `df` com seus valores normalizados no intervalo [0.001, 1.0].
        Mantém os mesmos índices e colunas do DataFrame de entrada.

    Exceções
    ---------
    ValueError
        Se todos os valores considerados (fora da diagonal, caso `ignore_diagonal=True`)
        forem idênticos, impedindo o cálculo de uma transformação linear válida.

    Exemplos
    --------
    >>> import pandas as pd
    >>> mat = pd.DataFrame([[0.0, 2.0], [2.0, 0.0]])
    >>> normalize_df(mat)
         0        1
    0  0.001  1.000
    1  1.000  0.001

    >>> normalize_df(mat, ignore_diagonal=False)
         0        1
    0  0.001  1.000
    1  1.000  0.001
    """
    df = df.copy()
    min_value = 0.001
    max_value = 1.0

    # Seleciona valores para cálculo de mínimo e máximo
    if ignore_diagonal:
        mask = ~np.eye(df.shape[0], dtype=bool)
        values = df.values[mask]
    else:
        values = df.values.flatten()

    # Determina os extremos reais
    min_real = values.min()
    max_real = values.max()

    if max_real == min_real:
        raise ValueError("Todos os valores são iguais — normalização não é possível.")

    # Normalização linear para [min_value, max_value]
    normalized = (df - min_real) / (max_real - min_real)
    normalized = normalized * (max_value - min_value) + min_value

    # Definir a diagonal principal como 0.0
    np.fill_diagonal(normalized.values, 0.0)
    return normalized

import numpy as np
import pandas as pd

def normalize_df_0_1(df: pd.DataFrame, ignore_diagonal: bool = True) -> pd.DataFrame:
    """
    Normaliza os valores de um DataFrame de distâncias para o intervalo [0.0, 1.0].

    Esta função mapeia linearmente os valores de `df` do seu intervalo original
    para o novo intervalo [0.0, 1.0] usando a escala Min-Max. Pode opcionalmente
    ignorar a diagonal principal ao determinar os valores mínimo e máximo reais
    dos dados para definir a escala.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame contendo a matriz de distâncias ou quaisquer valores numéricos
        que se deseja normalizar. Deve ser quadrado se `ignore_diagonal=True`.
    ignore_diagonal : bool, opcional
        Se True (padrão), exclui a diagonal principal do cálculo dos valores
        mínimo e máximo que definem a escala de normalização.
        Se False, considera todos os valores do DataFrame para definir a escala.

    Retorna
    -------
    pandas.DataFrame
        Cópia de `df` com seus valores normalizados no intervalo [0.0, 1.0].
        Mantém os mesmos índices e colunas do DataFrame de entrada.
        A diagonal principal é garantida como 0.0.

    Exceções
    ---------
    ValueError
        Se o DataFrame não for quadrado (quando ignore_diagonal=True), ou
        se todos os valores considerados (fora da diagonal, caso
        ignore_diagonal=True) forem idênticos, impedindo a normalização.
    TypeError
        Se a entrada 'df' não for um pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
         raise TypeError("Entrada 'df' deve ser um pandas DataFrame.")
    if ignore_diagonal and df.shape[0] != df.shape[1]:
         raise ValueError("DataFrame deve ser quadrado quando ignore_diagonal=True.")

    df_normalized = df.copy() # Trabalha com uma cópia

    # Seleciona valores para cálculo de mínimo e máximo que definirão a escala
    if ignore_diagonal:
        mask_off_diagonal = ~np.eye(df.shape[0], dtype=bool)
        values_to_scale = df.values[mask_off_diagonal]
        if values_to_scale.size == 0 and df.shape[0] > 0: # Caso de matriz 1x1 ou só com diagonal
             values_to_scale = df.values.flatten() # Recai para usar todos os valores
    else:
        values_to_scale = df.values.flatten()

    # Verifica se há valores para escalar
    if values_to_scale.size == 0:
         # Se o DataFrame original era vazio, retorna a cópia vazia
         return df_normalized

    # Determina os extremos reais dos valores a serem escalonados
    min_real = values_to_scale.min()
    max_real = values_to_scale.max()

    # Verifica se a normalização é possível (evita divisão por zero)
    if max_real == min_real:
        # Se todos os valores considerados são iguais, não normaliza ou levanta erro.
        # Vamos levantar erro para consistência com a função original.
        raise ValueError("Todos os valores considerados para escala são idênticos — normalização não é possível.")

    # --- FÓRMULA DE NORMALIZAÇÃO MIN-MAX PARA [0, 1] ---
    # Aplica a fórmula a todos os elementos do DataFrame copiado.
    # x_normalized = (x - min_real) / (max_real - min_real)
    range_real = max_real - min_real
    df_normalized = (df_normalized - min_real) / range_real

    # --- PÓS-PROCESSAMENTO ---
    # Garante que a diagonal principal seja exatamente 0.0
    # (A normalização pode ter alterado os zeros originais se min_real > 0)
    np.fill_diagonal(df_normalized.values, 0.0)

    # Garante que os valores estejam estritamente dentro de [0, 1]
    # (corrige pequenas imprecisões de ponto flutuante > 1 ou < 0)
    df_normalized.clip(lower=0.0, upper=1.0, inplace=True)
    # Re-zera a diagonal caso o clip tenha afetado (improvável, mas seguro)
    np.fill_diagonal(df_normalized.values, 0.0)

    return df_normalized


def normalize_df_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza uma matriz de distância (DataFrame) para o intervalo [0, 1]
    com tratamento especial pós-normalização para zeros, calculando epsilon
    a partir dos dados já normalizados.

    1. Encontra o mínimo e máximo dos valores fora da diagonal na matriz original.
    2. Aplica Min-Max Scaling aos valores fora da diagonal para mapeá-los para [0, 1].
    3. Define a diagonal principal como 0.0.
    4. Encontra o menor valor positivo FORA da diagonal na matriz JÁ NORMALIZADA.
    5. Calcula epsilon = 1e-3 * menor_valor_positivo_normalizado.
    6. Substitui quaisquer valores zero restantes FORA da diagonal principal
       pelo valor epsilon calculado.

    Parâmetros
    ----------
    df : pandas.DataFrame
        DataFrame quadrado contendo a matriz de distâncias original.

    Retorna
    -------
    pandas.DataFrame
        Cópia de `df` com seus valores normalizados conforme as regras.

    Exceções
    ---------
    ValueError
        Se o DataFrame não for quadrado ou se todos os valores fora da
        diagonal na matriz original forem idênticos.
    """
    if df.shape[0] != df.shape[1]:
        raise ValueError("O DataFrame de entrada deve ser uma matriz quadrada.")

    df_norm = df.copy()
    n = df_norm.shape[0]
    mask_off_diagonal = ~np.eye(n, dtype=bool)

    # --- Passos baseados nos dados ORIGINAIS para a escala ---
    off_diagonal_values_original = df.values[mask_off_diagonal]
    min_off_diagonal_original = off_diagonal_values_original.min()
    max_off_diagonal_original = off_diagonal_values_original.max()

    if max_off_diagonal_original == min_off_diagonal_original:
        # Se todos os valores originais fora da diagonal forem iguais.
        raise ValueError("Todos os valores fora da diagonal originais são idênticos — normalização padrão não é possível.")

    # --- Aplicação da Normalização [0, 1] ---
    range_original = max_off_diagonal_original - min_off_diagonal_original
    # Aplica a normalização apenas aos elementos fora da diagonal
    df_norm.values[mask_off_diagonal] = (df_norm.values[mask_off_diagonal] - min_off_diagonal_original) / range_original

    # --- Tratamento de Zeros PÓS-NORMALIZAÇÃO ---
    # 1. Zera a diagonal principal explicitamente
    np.fill_diagonal(df_norm.values, 0.0)

    # 2. Encontra o menor valor POSITIVO fora da diagonal NA MATRIZ NORMALIZADA
    off_diagonal_values_normalized = df_norm.values[mask_off_diagonal]
    positive_off_diagonal_normalized = off_diagonal_values_normalized[off_diagonal_values_normalized > 0]

    if len(positive_off_diagonal_normalized) == 0:
        # Caso especial: todos fora da diagonal se tornaram 0 após normalização,
        # ou já eram 0 originalmente.
        print("Aviso: Nenhum valor positivo encontrado fora da diagonal após a normalização.")
        epsilon = 1e-10 # Usa um epsilon muito pequeno default
    else:
        min_pos_off_diagonal_normalized = positive_off_diagonal_normalized.min()
        # 3. Calcula epsilon baseado no mínimo normalizado
        epsilon = 1e-3 * min_pos_off_diagonal_normalized
        # Garante que epsilon não seja zero devido à precisão de ponto flutuante
        if epsilon == 0:
             epsilon = np.finfo(float).eps # Menor número positivo representável


    # 4. Identifica e substitui os zeros FORA da diagonal pelo epsilon
    zero_off_diagonal_mask = (df_norm.values == 0.0) & mask_off_diagonal
    num_zeros_off_diag = np.sum(zero_off_diagonal_mask)
    if num_zeros_off_diag > 0:
        print(f"Substituindo {num_zeros_off_diag} zeros fora da diagonal por epsilon ({epsilon:.2e})...")
        df_norm.values[zero_off_diagonal_mask] = epsilon

    # Garante clipagem final para [0, 1]
    df_norm.clip(lower=0.0, upper=1.0, inplace=True)
    # Re-zera a diagonal caso o clip tenha afetado (improvável, mas seguro)
    np.fill_diagonal(df_norm.values, 0.0)

    return df_norm

# --- EXEMPLO DE USO ---
# Suponha que df_dist_original é sua matriz de distância calculada
# filename = 'matrizDistanciaLinha3DBaseInteira_NORMALIZADAv4.csv' # Novo nome
# if not os.path.exists(filename):
#     try:
#         df_dist_normalizada_v4 = normalize_df_v4(df_dist_original)
#         print("Normalização v4 concluída.")
#         df_dist_normalizada_v4.to_csv(filename, sep=';', decimal=',')
#         print(f'Arquivo normalizado "{filename}" gerado e salvo.')
#     except ValueError as e:
#         print(f"ERRO durante a normalização: {e}")
# else:
#     print(f'Arquivo "{filename}" já existe.')
#
# print(df_dist_normalizada_v4.head()) # Verificar resultado


def calcular_flip_rate_de_arquivo(arquivo_de_entrada: str,
                                  arquivo_de_saida: str = None,
                                  salvar_csv: bool = True,
                                  sep: str = ';',
                                  decimal: str = ',') -> pd.DataFrame:
    """
    Calcula o flip rate a partir de um arquivo CSV e retorna um DataFrame em formato largo.

    Lê um arquivo de rótulos, calcula o flip rate para cada perturbação e, em seguida,
    formata a saída para que cada linha represente um valor de 'sigma' e as colunas
    contenham os flip rates para cada perturbação individual.

    Args:
        arquivo_de_entrada (str): O caminho para o arquivo CSV com os rótulos.
        arquivo_de_saida (str, optional): O caminho para salvar o CSV com os resultados.
            Se None, um nome padrão será gerado. Defaults to None.
        salvar_csv (bool, optional): Se True, salva os resultados em um arquivo CSV.
            Defaults to True.
        sep (str, optional): O separador de colunas do arquivo CSV. Defaults to ';'.
        decimal (str, optional): O caractere de decimal do arquivo CSV. Defaults to ','.

    Returns:
        pd.DataFrame: Um DataFrame em formato largo com as colunas
                      ['sigma', 'qq', 'flip_rate_pert_1', 'flip_rate_pert_2', ...].
    """

    print(f"Carregando o arquivo de entrada: {arquivo_de_entrada}...")
    try:
        df = pd.read_csv(arquivo_de_entrada, sep=sep, decimal=decimal)
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{arquivo_de_entrada}' não foi encontrado.")
        return pd.DataFrame()

    colunas_rotulos = [col for col in df.columns if col.startswith('rotulo_')]
    total_de_rotulos = len(colunas_rotulos)

    if total_de_rotulos == 0:
        print("ERRO: Nenhuma coluna de rótulo (ex: 'rotulo_0') foi encontrada.")
        return pd.DataFrame()

    print(f"Arquivo carregado com sucesso. Encontradas {total_de_rotulos} colunas de rótulos.")

    resultados_flip_rate_longo = []
    sigmas_unicos = df['sigma'].unique()

    print("Calculando o flip rate para cada combinação de sigma e perturbação...")
    for sigma in tqdm(sigmas_unicos, desc="Processando Sigmas"):
        df_sigma_atual = df[df['sigma'] == sigma]
        linha_original = df_sigma_atual[df_sigma_atual['perturbacao'] == 0]

        if linha_original.empty:
            print(f"\nAviso: Nenhuma linha original (perturbação == 0) encontrada para sigma = {sigma}. Pulando.")
            continue

        rotulos_originais = linha_original[colunas_rotulos].iloc[0]
        df_perturbado = df_sigma_atual[df_sigma_atual['perturbacao'] > 0]

        for index, linha_pert in df_perturbado.iterrows():
            rotulos_perturbados = linha_pert[colunas_rotulos]
            num_diferencas = (rotulos_originais != rotulos_perturbados).sum()
            flip_rate = num_diferencas / total_de_rotulos

            resultados_flip_rate_longo.append({
                'sigma': sigma,
                'qq': linha_pert['qq'],
                'perturbacao': int(linha_pert['perturbacao']),
                'flip_rate': flip_rate
            })

    # Se nenhum resultado foi calculado, retorna um DataFrame vazio.
    if not resultados_flip_rate_longo:
        print("Nenhum resultado de flip rate foi calculado.")
        return pd.DataFrame()

    # Cria o DataFrame no formato longo primeiro
    df_longo = pd.DataFrame(resultados_flip_rate_longo)

    # --- NOVO: PIVOTANDO O DATAFRAME PARA O FORMATO LARGO ---
    print("\nFormatando a saída para o formato largo...")
    df_largo = df_longo.pivot_table(index=['sigma', 'qq'],
                                    columns='perturbacao',
                                    values='flip_rate').reset_index()

    # Renomeia as colunas para um formato mais descritivo (ex: 1 -> flip_rate_pert_1)
    df_largo.columns = ['sigma', 'qq'] + [f'flip_rate_pert_{col}' for col in df_largo.columns[2:]]
    # --------------------------------------------------------

    if salvar_csv:
        if arquivo_de_saida is None:
            base_name = os.path.basename(arquivo_de_entrada)
            arquivo_de_saida = f'resultados_fliprate_formato_largo_de_{base_name}'

        df_largo.to_csv(arquivo_de_saida, index=False, sep=sep, decimal=decimal)
        print("\n--- CÁLCULO CONCLUÍDO ---")
        print(f"Resultados salvos com sucesso em: '{arquivo_de_saida}'")

    return df_largo


# # --- EXEMPLO DE COMO USAR A FUNÇÃO ---
# if __name__ == "__main__":
#
#     arquivo_com_rotulos = 'resultados_rotulos_formato_largo_64_quantis.csv'
#
#     # Chama a função para obter o DataFrame no novo formato largo
#     df_flip_rates_largo = calcular_flip_rate_de_arquivo(arquivo_com_rotulos)
#
#     if not df_flip_rates_largo.empty:
#         print("\nAmostra dos resultados calculados no formato largo:")
#         # Mostra as 5 primeiras colunas para facilitar a visualização
#         print(df_flip_rates_largo.iloc[:, :5].head())
#         print(f"\nDimensões do DataFrame final: {df_flip_rates_largo.shape}")


def calcular_flip_rate(rotulos_originais, rotulos_perturb):
    return 1 - np.mean(rotulos_originais == rotulos_perturb)



def processar_qq(qq, df_original, algoritmo, alpha, sigmas, num_matriz_Pert, num_pert, output_dir_base, matriz_dist_path, sigma_noise, save):
    """
    Função que executa o processo de geração de matrizes pertubadas, suas margens e rotulacoes para um único valor de qq.
    """
    try:
        print(f"Iniciando processo para qq = {qq}...")
        print(f'Matriz de distancia {matriz_dist_path}...')
        # Cria uma subpasta específica para este valor de qq
        qq_dir = os.path.join(output_dir_base, f'{qq}_quantis')
        os.makedirs(qq_dir, exist_ok=True)

        # Chamada da função principal de processamento
        gerar_matrizes_margens_por_sigma(
            qq=qq,
            algoritmo=algoritmo,
            sigmas=sigmas,
            df_original=df_original.copy(),
            load_results_from_csv=load_results_from_csv,
            utils=utils,
            num_matriz_Pert=num_matriz_Pert,
            num_pert=num_pert,
            output_dir=qq_dir,
            matriz_dist_path=matriz_dist_path,
            sigma_noise=sigma_noise,
            alpha=alpha,
            save=save
        )

        print(f"Processo para qq = {qq} finalizado com sucesso.")
        return f"Sucesso para qq={qq}"
    except Exception as e:
        print(f"Erro no processo para qq = {qq}: {e}")
        return f"Erro para qq={qq}: {e}"
