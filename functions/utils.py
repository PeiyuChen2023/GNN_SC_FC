import glob
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf
import pickle
import random
import bct
from bct.utils import binarize
#from spektral.utils import normalized_laplacian
from sklearn.decomposition import PCA
from mapalign import embed
from sklearn.metrics import pairwise_distances
import multiprocessing
import scipy.stats as stats
import scipy
# import torch
import seaborn as sns
from scipy.sparse import coo_matrix

# def correlation_coefficient_loss(y_true, y_pred):
#     x = y_true
#     y = y_pred
#     mx = K.mean(x)
#     my = K.mean(y)
#     xm, ym = x-mx, y-my
#     r_num = K.sum(xm * ym)
#     r_den = K.sum(K.sum(K.square(xm)) * K.sum(K.square(ym)))
#     r = r_num / r_den
#     return 1 - r**2
#
# def correlation(x, y):
#     mx = tf.math.reduce_mean(x)
#     my = tf.math.reduce_mean(y)
#     xm, ym = x-mx, y-my
#     r_num = tf.math.reduce_mean(tf.multiply(xm,ym))
#     r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
#     return r_num / r_den
# laplacian矩阵


def spearmanr(x,y):
    '''
    :param x,y are all 1-D numpy array
    :return: spearmanr corr, similar to scipy.stats.spearmanr
    '''
    x = pd.Series(x)
    y = pd.Series(y)

    # 处理数据删除Nan
    x = x.dropna()
    y = y.dropna()
    n = x.count()
    x.index = np.arange(n)
    y.index = np.arange(n)

    rho = x.corr(y, method='spearman')
    return rho


def get_communicability(adj_matrix):
    # negative square root of nodal degrees
    row_sum = adj_matrix.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adj_matrix @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = scipy.sparse.linalg.expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0
    return cmc

#
# def get_communicability(adj_matrix):
#     R = np.sum(adj_matrix, axis=1)
#     R_sqrt = 1 / np.sqrt(R)
#     D_sqrt = np.diag(R_sqrt)
#     A = np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
#     return scipy.sparse.linalg.expm(A)  # Compute the matrix exponential of A.


def get_search_information(adj_matrix, transform="log"):
    return bct.search_information(adj_matrix, transform=transform)


def get_r2(target, pred):
    return 1 - np.sum((target - pred) ** 2) / np.sum((target - np.mean(target)) ** 2)




def rewired_graph(x, rewire_parameter=3):
    if len(x.shape) == 2:
        temp_graph = x.copy()
        rewire_graph, _ = bct.randmio_und_connected(temp_graph, rewire_parameter)

        return rewire_graph
    result = np.zeros_like(x)
    for i in range(x.shape[0]):
        print("rewired:", i)
        temp_graph = x[i, :, :].copy()
        rewire_graph, _ = bct.randmio_und_connected(temp_graph, 3)
        result[i, :, :] = rewire_graph
    return result


def rewired_graph_single(i, x, par=5):
    print("rewired:", i)
    rewire_graph, _ = bct.randmio_und_connected(x[i,:,:], par)
    return rewire_graph

def rewired_graph_parallel(x, par, ncpu=10):
    num_cpus = multiprocessing.cpu_count()
    print("cpu num: ", num_cpus)
    pool = multiprocessing.Pool(processes=ncpu)
    results = pool.starmap(rewired_graph_single, [(i, x, par) for i in range(x.shape[0])])
    pool.close()
    pool.join()
    result = np.stack(results)
    for i in range(x.shape[0]):
        temp_corr = np.corrcoef(x[i, :, :].reshape(-1), result[i, :, :].reshape(-1))[0,1]
        if temp_corr > 0.9:
            print("rewire fail")
            result[i, :, :] = rewired_graph(edge_filter(x[i, :, :], 0.5))
            print("new rewire r:", np.corrcoef(x[i, :, :].reshape(-1), result[i, :, :].reshape(-1))[0,1])
    return result



def get_symmetric_matrix(a, half="upper"):
    x = a.copy()
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[1]):
            if half=="upper":
                x[j, i] = x[i, j]
            else:
                x[i, j] = x[j, i]

    return x


def get_PCA_result(x, component_num=1):
    pca = PCA()
    lower_x = pca.fit_transform(x)
    # print("the pca is")
    #print(pca.explained_variance_ratio_)
    return lower_x[:,:component_num]


def get_diffusion_map(x, component_num=1):
    aff = 1 - pairwise_distances(x, metric='cosine')
    emb = embed.compute_diffusion_map(aff, alpha=0.5)
    return emb[:,:component_num]


def get_submatrix(x, index1, index2):
    '''
    :param x: ndarray 2-d
    :param index1: ndarray row index
    :param index2: numpy column index
    :return: mask the matrix outside the row and column index
    '''
    adj = np.zeros((x.shape[0], x.shape[1]))
    for i in index1:
        for j in index2:
            adj[i,j] = x[i,j]
    return adj


def get_sub_region(x, r1, r2, if_between_inc_within, if_rewired_within=False, rewire_parameter=3):
    region2index_dict, index2region_dict, name = get_intrinsic_index(if_left_right=False)
    if r1!=r2:
        index1 = region2index_dict[r1]
        index2 = region2index_dict[r2]
        if not if_between_inc_within:
            region1 = np.zeros((len(index1), len(index1)))
            region2 = np.zeros((len(index2), len(index2)))

        else:
            region1 = x[index1, :]
            region1 = region1[:, index1]
            region2 = x[index2, :]
            region2 = region2[:, index2]
        region3 = x[index1, :]
        region3 = region3[:, index2]
        region4 = x[index2, :]
        region4 = region4[:, index1]
        if if_rewired_within and if_between_inc_within:
            region1, _ = bct.randmio_und_connected(region1, rewire_parameter)
            region2, _ = bct.randmio_und_connected(region2, rewire_parameter)

        result1 = np.concatenate((region1, region3), axis=1)
        result2 = np.concatenate((region2, region4), axis=1)
        result = np.concatenate((result1, result2), axis=0)
        return result
    else:
        index1 = region2index_dict[r1]
        region1 = x[index1, :]
        result = region1[:, index1]
        if if_rewired_within:
            result, _ = bct.randmio_und_connected(result, rewire_parameter)
        return result


def remove_one_region(x, r, if_rewired=False, rewire_parameter=3):
    region2index_dict, index2region_dict, name = get_intrinsic_index(if_left_right=False)
    remv_index = region2index_dict[r]
    result = x.copy()
    result = np.delete(result, remv_index, axis=0)
    result = np.delete(result, remv_index, axis=1)
    return result




def select_region(x, r1, r2, if_left_right=False, if_cross_include_within=True):
    adj = x
    region2index_dict, index2region_dict, name = get_intrinsic_index(if_left_right=if_left_right)
    index1 = region2index_dict[r1]
    index2 = region2index_dict[r2]
    result = get_submatrix(adj, index1, index2)
    if r1!=r2:
        result += get_submatrix(adj, index2, index1)
        if if_cross_include_within:
            result += get_submatrix(adj, index1, index1)
            result += get_submatrix(adj, index2, index2)
    return result
    # if r1==r2:
    #     index = region2index_dict[r1]
        # index = np.concatenate((np.array(range(index[0])), np.array(range(index[-1]+1, 400)))).astype(int)
        # adj[index, :] = 0
        # adj[:, index] = 0
    #     return adj
    # else:
    #     r_min = min(r1, r2)
    #     r_max = max(r1, r2)
    #     index_min = region2index_dict[r_min]
    #     index_max = region2index_dict[r_max]
    #     index = np.concatenate((np.array(range(index_min[0])), np.array(range(index_min[-1]+1, index_max[0])), np.array(range(index_max[-1]+1,  400)))).astype(int)
    #     adj[index, :] = 0
    #     adj[:, index] = 0
        # adj[index_min, :] = 0
        # adj[:, index_max] = 0
        # adj = adj + adj.T
        # for i in index_min:
        #     for j in index_min:
        #         adj[i,j] = 0
        # for i in index_max:
        #     for j in index_max:
        #         adj[i,j] = 0
        # return adj



def getDimensionality_SVD(weight):
    u, sigmas, vT = np.linalg.svd(weight)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for sig in sigmas:
        dimensionality_nom += np.real(sig)
        dimensionality_denom += np.real(sig)**2
    dimensionality = dimensionality_nom**2/dimensionality_denom
    dimensionality = dimensionality/min(weight.shape)
    return dimensionality

def getDimensionality(data):
    """
    data needs to be a square, symmetric matrix
    """
    corrmat = data
    eigenvalues, eigenvectors = np.linalg.eig(corrmat)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2

    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality



def get_intrinsic_index(if_left_right=False):
    w = [0, 31, 68, 91, 113, 126, 148, 200, 230, 270, 293, 318, 331, 361, 400]
    region2index_dict = {}
    index2region_dict = {}
    region = 0
    for i in range(400):
        if i >= w[region + 1]:
            region += 1
        index2region_dict[i] = region
    if if_left_right:
        region_num = 14
        name = ["LH_Vis", "LH_SomMot", "LH_DorsAttn", "LH_SalVentAttn", "LH_Limbic", "LH_Cont", "LH_Default",
                "RH_Vis", "RH_SomMot", "RH_DorsAttn", "RH_SalVentAttn", "RH_Limbic", "RH_Cont", "RH_Default"]
        for i in range(region_num):
            region2index_dict[i] = np.array(np.arange(w[i], w[i + 1]), dtype=int).reshape(-1)
        return region2index_dict, index2region_dict, name
    else:
        region_num = 7
        name = ["VIS", "SMN", "DAN", "VAN", "LIM", "FPN", "DMN"]
        for i in range(region_num):
            region2index_dict[i] = np.concatenate((np.arange(w[i], w[i + 1]),np.arange(w[i + 7], w[i + 8])),dtype=int). reshape(-1)
        for i in range(400):
            index2region_dict[i] = (index2region_dict[i]) % 7
        return region2index_dict, index2region_dict, name



def get_intrinsic_index_17(if_left_right=False):
    w = [0, 12, 24, 43, 59, 72, 85, 100, 108, 113, 120, 133, 143, 148, 166, 187, 194, 200, 212, 223, 243, 258, 272, 284, 303, 312, 318, 324, 335, 350, 357, 373, 384, 390,  400]
    region2index_dict = {}
    index2region_dict = {}
    region = 0
    for i in range(400):
        if i >= w[region + 1]:
            region += 1
        index2region_dict[i] = region
    region_num = 17
    name = ["VisCent","VisPeri", "SomMotA", "SomMotB", "DorsAttnA","DorsAttnB",  "SalVentAttnA", "SalVentAttnB", "LimbicB", "LimbicA",  "ContA", "ContB", "ContC", "DefaultA", "DefaultB",  "DefaultC", "TempPar"]
    for i in range(region_num):
        region2index_dict[i] = np.concatenate((np.arange(w[i], w[i + 1]),np.arange(w[i + 7], w[i + 8])),dtype=int). reshape(-1)
    for i in range(400):
        index2region_dict[i] = (index2region_dict[i]) % 17
    return region2index_dict, index2region_dict, name



def is_symmetric_matrix(x):
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[1]):
            if(x[i,j]!=x[j,i]):
                # print(x[i])
                # print(x[j])
                # print(x[i,j])
                # print(x[j,i])
                return False
    return True


#
def edge_filter(adj_matrix, ratio=0.5):
    flatten_matrix = adj_matrix.reshape(-1)
    flatten_matrix_abs = np.abs(flatten_matrix)
    length = len(adj_matrix)
    remv_num = int((1-ratio)*len(flatten_matrix))
    rank_index = np.argsort(flatten_matrix_abs)
    rm_index = rank_index[:remv_num]
    flatten_matrix[rm_index]=0
    filtered_matrix = flatten_matrix.reshape(length, -1)
    return filtered_matrix


def edge_filter_zero(adj_matrix):
    x = adj_matrix.copy()
    x[x<0] = 0
    return x



def unnormalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adj_matrix


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)


def add_pos(adj):
    length = adj.shape[0]
    pos = np.ones(length)
    pos = np.diag(pos)
    fea = np.concatenate((adj,pos),axis=-1)
    return fea

def get_train_val_test_split(data, val_ratio, test_ratio):
    if test_ratio + val_ratio > 0.5:
        print("split mistake!")
        return 0,0
    test_len = int(test_ratio*len(data))
    val_len = int(val_ratio*len(data))
    val_index = len(data) - test_len - val_len
    test_index = len(data) - test_len
    print(test_index, val_index)
    return val_index, test_index


def flatten_symmetric_matrix(x):
    '''
    :param x: 352*352 symmetric matrix
    :return: a flatten matrix with shape 352*(352-1)/2, ignore the diagonal elements
    '''
    length = x.shape[0]
    size = length * (length - 1) / 2
    flat_array = np.zeros(int(size))
    num = 0
    for i in range(length - 1):
        for j in range(i + 1, length):
            flat_array[num] = x[i][j]
            num += 1
    return flat_array



def flatten_symmetric_matrix_to_ADJ(flat_array, length=400):
    size = len(flat_array)
    adj = np.zeros((length, length))
    num = 0
    for i in range(length-1):
        for j in range(i + 1, length):
            adj[i][j] = flat_array[num]
            num += 1
    if num!=size:
        print("false matrix")
    adj = adj + adj.T
    return adj


def get_node_strength(x):
    '''
    :param x: a symmetric matrix
    :return: an 1-D numpy array
    '''
    ns_array = np.sum(x, axis=0)
    for i in range(len(x)):
        # minus the diagonal element
        ns_array[i] -= x[i, i]
    return ns_array


def decoding_probability(x):
    mean_p = 0
    for i in range(x.shape[0]):
        temp  = x[i, :]
        rank = np.argsort(np.argsort(temp))
        mean_p += rank[i]/x.shape[0]
    return mean_p/x.shape[0]

def decoding_accuracy(x, mode="all"):
    '''
    :param x: input
    :param type: row or column, here row is SC, column is FC
    :return:
    '''
    success = 0
    for i in range(x.shape[0]):
        target = x[i, i]
        if mode == "SC":
            temp = x[:, i]
        elif mode == "FC":
            temp = x[i, :]
        else:
            temp = np.concatenate((x[:, i], x[i, :]), axis=0)
        if(target == np.max(temp)):
            success+=1
    return success/x.shape[0]

def diagonal_diff(x):
    diagonal = np.diagonal(x)
    others = (np.sum(x) - np.sum(diagonal))/(x.shape[0]*(x.shape[0]-1))
    diagonal = np.mean(diagonal)
    return diagonal - others


def get_ind_effect(x, mode='all'):
    diagonal = np.diagonal(x)
    if mode =='SC':
        others = (np.sum(x, axis=1) - diagonal)/(x.shape[0]-1)
    elif mode =='FC':
        others = (np.sum(x, axis=0) - diagonal)/(x.shape[0]-1)
    else:
        others = (np.sum(x, axis=0) + np.sum(x, axis=1) - 2*diagonal)/((x.shape[0]-1)*2)

    t, p = stats.ttest_rel(diagonal, others)
    return diagonal - others, t, p


def get_group_ind_effect(x, mode='all'):
    diagonal = np.diagonal(x)
    if mode =='SC':
        others = (np.sum(x, axis=1) - diagonal)/(x.shape[0]-1)
    elif mode =='FC':
        others = (np.sum(x, axis=0) - diagonal)/(x.shape[0]-1)
    else:
        others = (np.sum(x, axis=0) + np.sum(x, axis=1) - 2*diagonal)/((x.shape[0]-1)*2)

    t, p = stats.ttest_rel(diagonal, others, alternative='greater')
    return np.mean(others), np.mean(diagonal - others), t, p


# def get_match_mismatch_t_p(x, mode='all',alternative='greater'):
#     diagonal = np.diagonal(x)
#     if mode =='SC':
#         others = (np.sum(x, axis=1) - diagonal)/(x.shape[0]-1)
#     elif mode =='FC':
#         others = (np.sum(x, axis=0) - diagonal)/(x.shape[0]-1)
#     else:
#         others = (np.sum(x, axis=0) + np.sum(x, axis=1) - 2*diagonal)/((x.shape[0]-1)*2)
#
#     t, p = stats.ttest_rel(diagonal, others, alternative=alternative)
#     return diagonal, others, t, p

def get_match_mismatch_t_p(x):
    diagonal = np.diagonal(x)
    others = (np.sum(x, axis=0) + np.sum(x, axis=1) - 2*diagonal)/((x.shape[0]-1)*2)
    t, p = stats.ttest_rel(diagonal, others)
    return diagonal, others, t, p


def diagonal_value(x):
    diagonal = np.diagonal(x)
    diagonal = np.mean(diagonal)
    return diagonal


def diagonal_t_value(x):
    diagonal = np.diagonal(x).reshape(-1)
    others = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if i!=j:
                others.append(x[i, j])
    others = np.array(others)
    t, p = stats.ttest_ind(diagonal, others)
    sign = np.sign(np.mean(diagonal)-np.mean(others))
    t = abs(t) * sign
    p = abs(p) * sign
    return t, p


# with open('/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/ABCD_SC_FC_reSC_info_1301_final.pickle', 'rb') as out_data:
#     (SC, FC, re_SC, info) = pickle.load(out_data)
#
# re_SC = rewired_graph_parallel(SC, 10, 18)
#
# with open('/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/ABCD_SC_FC_reSC_info_1301_final.pickle', 'wb') as in_data:
#     pickle.dump((SC, FC, re_SC, info), in_data, pickle.HIGHEST_PROTOCOL)
#

# with open('/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/SC_FC_341.pickle', 'rb') as out_data:
#     (SC, FC) = pickle.load(out_data)
# rewired_SC = SC
# avg_SC_FC = 0
# avg_reSC_SC = 0
# avg_reSC_FC = 0
#
# for i in range(SC.shape[0]):
#     random.seed(10)
#     # for j in range(400):
#     #     SC[i, j, j] = 0
#     #     rewired_SC[i, j, j] = 0
#     #     FC[i, j, j] = 0
#
#     temp_SC = SC[i, :, :]
#     temp_FC = FC[i, :, :]
#
#     temp_rewired_SC,_ = bct.randmio_und_connected(temp_SC, 3)
#
#
#     # print(i, is_symmetric_matrix(temp_FC))
#     # temp_rewired_FC, _ = bct.randmio_und_connected(temp_FC, 5)
#     # rewired_FC[i,:,:] = temp_rewired_FC
#
#     # temp_rewired_SC = temp_rewired_SC.reshape(-1)
#     # temp_SC = temp_SC.reshape(-1)
#
#     corr_1 = np.corrcoef(temp_SC.reshape(-1), temp_FC.reshape(-1))[0,1]
#     avg_SC_FC += corr_1
#     corr_2 = np.corrcoef(temp_rewired_SC.reshape(-1), temp_SC.reshape(-1))[0,1]
#     avg_reSC_SC += corr_2
#     corr_3 = np.corrcoef(temp_rewired_SC.reshape(-1), temp_FC.reshape(-1))[0,1]
#     avg_reSC_FC += corr_3
#
#     # for k in range(len(temp_rewired_SC)):
#     #     if temp_SC[k] != temp_rewired_SC[k]:
#     #         print("yes")
#
#
#     print(corr_1, corr_2, corr_3)
#     break

# print(avg_SC_FC/241)
# print(avg_reSC_SC/241)
# print(avg_reSC_FC/241)


# with open('/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/SC_FC_reSC_241.pickle', 'wb') as in_data:
#     pickle.dump((SC, FC, rewired_SC), in_data, pickle.HIGHEST_PROTOCOL)
# a = np.array(range(300))
# x, y, z,= get_intrinsic_index()
# print(x)
# print(y)
# print(z)
# print(y[100])
# print(y[399])
# x, y, z,= get_intrinsic_index(if_left_right=True)
# print(x)
# print(y)
# print(z)
# print(y[100])
# print(y[399])

# a = np.ones((400,400), dtype=int)
# np.set_printoptions(threshold=np.inf)
# a1 = select_region(a, 0, 0)
# print(a1)
# a2 = select_region(a,0,1)
# print(a2)
#
# plt.figure(dpi=60)
# sns.heatmap(data=pd.DataFrame(a1))
# plt.savefig("/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_plot/temp1.png")
# plt.figure(dpi=60)
# sns.heatmap(data=pd.DataFrame(a2))
# plt.savefig("/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_plot/temp2.png")
#


# temp_graph = np.array([[0, 1, 2], [0, 0, 3], [1, 3, 0]])
# temp_graph += 100
# for i in range(3):
#     temp_graph[i,i] = 0
# coo_temp_x = coo_matrix(temp_graph)
# edge_index = torch.tensor(np.vstack((coo_temp_x.row, coo_temp_x.col)), dtype=torch.long)
# values = torch.tensor(coo_temp_x.data, dtype=torch.float)
# values -= 100
# temp_graph_tensor = Data(edge_index=edge_index, edge_attr=values)
# print(temp_graph_tensor.edge_index)
# print(temp_graph_tensor.edge_attr)
# #