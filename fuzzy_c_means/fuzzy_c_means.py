# Artur Mello
# Fuzzy C Means
# TP 1 - Sistemas Nebulosos
import numpy as np

ITER_MAX = 30

'''
    cluster_distance(X, c)
    # calcula a distancia euclidiana 
    # entre o dado observado e cada cluster

    :param X = matriz contendo dados observados
    :param c = centroide do cluster 
    :return d = matriz de distâncias euclidianas      
'''


def cluster_distance(X, c):
    n_obs, n_dim = np.shape(X)
    n_clust = np.shape(c)[0]
    d = np.zeros([n_obs, n_clust])
    for j in range(n_clust):
        for i in range(0, n_obs):
            d[i][j] = np.sqrt(np.sum((X[i] - c[j]) ** 2))
    return d


'''
    fuzzy_c_means_centroid(X, U)
    # calcula e atualiza os centroids dos clusters 
    # baseando-se na matriz de pertinencia
    
    :param X = matriz contendo dados observados
    :param U = matriz de pertinencia 
    :return c = vetor contendo centroides atualizados de cada cluster       
'''


def fuzzy_c_means_centroid(X, U):
    n_obs, n_dim = np.shape(X)
    n_clust = np.shape(U)[1]
    c = np.zeros([n_clust, n_dim])
    for j in range(n_clust):
        sum_num = sum_den = 0
        for i in range(0, n_obs):
            sum_num += ((U[i][j]**2)*X[i])
            sum_den += (U[i][j]**2)
        c[j] = sum_num/sum_den
    return c


'''
    fcm_membership(X, c)
    # calcula a matriz de pertinência de cada amostra para cada cluster existente
    # Este calculo é feito a partir da distancia euclidiana do ponto amostral para o 
    # centroide do cluster, e, ao realizar o particionamento fuzzy, cada ponto terá
    # um grau de pertencimento para cada cluster. Este grau é atualizado a cada iteração
    # até que a função objetivo convirja, ou seja, quando a matriz de pertinência não se 
    # altera entre duas iterações consecutivas.
'''


def fcm_membership(X, c):
    n_obs, n_dim = np.shape(X)
    n_clust = np.shape(c)[0]
    U = np.zeros([n_obs, n_clust])
    d = cluster_distance(X, c)
    for j in range(n_clust):
        for i in range(n_obs):
            sumup = 0
            for k in range(n_clust):
                sumup += (d[i][j]/d[i][k])**2
            U[i][j] = 1/sumup
    return U


'''
    calculate_data_clusters(cluster_membership)
    Função auxiliar que calcula o cluster ao qual o determinado dado pertence,
    selecionando o maior valor do grau de pertencimento para cada ponto amostral 
    
    :param cluster_membership = matriz de pertinencia das amostras
    :return data_cluster = array com cluster ao qual cada amostra pertence  
'''


def calculate_data_clusters(cluster_membership):
    rows, cols = np.shape(cluster_membership)
    data_cluster = np.zeros([rows, 1])
    for i in range(rows):
        data_cluster[i, 0] = int(np.where(cluster_membership[i] == np.amax(cluster_membership[i]))[0][0])

    return data_cluster


def fuzzy_c_means(data, k):

    old_U = np.random.uniform(0, 1/k, [len(data), k])

    eta = 1000
    _iter = 0

    while eta > 0.0005 and _iter < ITER_MAX:
        _iter += 1

        centroids = fuzzy_c_means_centroid(data, old_U)
        new_U = fcm_membership(data, centroids)

        eta = np.sqrt(np.sum((new_U[0] - old_U[0]) ** 2))

        old_U = new_U

    data_clusters = calculate_data_clusters(new_U)

    return [data, centroids, data_clusters, _iter]
