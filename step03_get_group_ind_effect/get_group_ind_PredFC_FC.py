import sys
sys.path.append("../functions")
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.stats as stats
from utils import *
import argparse



def get_intermatrix_i_j(i, j, PredFC, FC):
    PredFC_i_flatten = flatten_symmetric_matrix(PredFC[i, :, :])
    FC_j_flatten = flatten_symmetric_matrix(FC[j, :, :])
    return i, j, np.nan_to_num(np.corrcoef(PredFC_i_flatten, FC_j_flatten)[0, 1])



def get_intermatrix(PredFC, FC):

    sub_num = PredFC.shape[0]
    PredFC_FC_inter_r = np.zeros((sub_num, sub_num))

    results = Parallel(n_jobs=-1)(
        delayed(get_intermatrix_i_j)(i, j, PredFC, FC) for i in range(sub_num) for j in range(sub_num))

    for i, j, result in results:
        PredFC_FC_inter_r[i, j] = result

    return PredFC_FC_inter_r



if __name__ == '__main__':

    # parser settings
    parser = argparse.ArgumentParser(description='GNN SC FC')
    parser.add_argument('--dataset', default='HCPA', help='choose a dataset')


    args = parser.parse_args()
    dataset_type = args.dataset

    if dataset_type == 'HCPA':
        try:
            with open('../data/preprocessed_data/HCPA_SC_FC_reSC_info_245.pickle',
                      'rb') as out_data:
                (SC, FC, SC_re, info) = pickle.load(out_data)
        except Exception:
            # pandas package may report error for >2.0 version
            (SC, FC, SC_re, info) = pd.read_pickle('../data/preprocessed_data/HCPA_SC_FC_reSC_info_245.pickle')
        with open('../data/result_out/HCPA/GCN_dim128lr0.001b2ep400reg0.0001prelu.pickle', 'rb') as out_data:
            predFC_train, predFC_test = pickle.load(out_data)

    elif dataset_type == 'HCPD':
        try:
            with open('../data/preprocessed_data/HCPD_SC_FC_reSC_info_410.pickle',
                      'rb') as out_data:
                (SC, FC, SC_re, info) = pickle.load(out_data)
        except:
            (SC, FC, SC_re, info) = pd.read_pickle('../data/preprocessed_data/HCPD_SC_FC_reSC_info_410.pickle')

        with open('../data/result_out/HCPD/GCN_dim128lr0.001b2ep400reg0.0001prelu.pickle', 'rb') as out_data:
            predFC_train, predFC_test = pickle.load(out_data)

    else:
        with open('../data/preprocessed_data/ABCD_SC_FC_reSC_info_1572.pickle', 'rb') as out_data:
            (SC, FC, SC_re, info) = pickle.load(out_data)
        with open('../data/result_out/ABCD/GCN_dim128lr0.001b2ep400reg0.0001prelu.pickle', 'rb') as out_data:
            predFC_train, predFC_test = pickle.load(out_data)

    SC_train, SC_test, FC_train, FC_test, SC_train_re, SC_test_re, = train_test_split(SC, FC, SC_re, test_size=0.5,
                                                                                      random_state=0)

    predFC_FC_inter_r = get_intermatrix(predFC_test, FC_test)
    # np.save('../data/result_out_2/' + dataset_type + '/Pred/'+dataset_type + '_inter_matrix_Pred_FC.npy', predFC_FC_inter_r)

    print("finish!")