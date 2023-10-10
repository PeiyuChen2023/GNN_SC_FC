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



def rewired_graph(x, rewire_parameter=5):
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


def get_symmetric_matrix(a, half="upper"):
    x = a.copy()
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[1]):
            if half=="upper":
                x[j, i] = x[i, j]
            else:
                x[i, j] = x[j, i]
    return x



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



def get_intrinsic_index_17():
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
                return False
    return True


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
    :param x: N*N  symmetric matrix
    :return: flatten matrix with shape N*(N-1)/2, ignore the diagonal elements
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
    '''
    :param x: flatten matrix with shape N*(N-1)/2
    :return: N*N  symmetric matrix
    '''
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


def get_match_mismatch_t_p(x):
    diagonal = np.diagonal(x)
    others = (np.sum(x, axis=0) + np.sum(x, axis=1) - 2*diagonal)/((x.shape[0]-1)*2)
    t, p = stats.ttest_rel(diagonal, others)
    return diagonal, others, t, p


def diagonal_value(x):
    diagonal = np.diagonal(x)
    diagonal = np.mean(diagonal)
    return diagonal
