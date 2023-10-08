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


def get_roi_inter_matrix(args):
    PredFC, FC, roi = args
    sub_num = PredFC.shape[0]
    PredFC_FC_inter_roi = np.zeros((sub_num, sub_num))

    for i in range(sub_num):
        PredFC_i = PredFC[i, :, :]
        PredFC_i_roi = PredFC_i[roi, :]
        PredFC_i_roi = np.delete(PredFC_i_roi, roi)

        for j in range(sub_num):
            FC_j = FC[j, :, :]
            FC_j_roi = FC_j[roi, :]
            FC_j_roi = np.delete(FC_j_roi, roi)

            PredFC_FC_inter_roi[i, j] = np.nan_to_num(np.corrcoef(PredFC_i_roi, FC_j_roi)[0, 1])

    match, mismatch, t, p = get_match_mismatch_t_p(PredFC_FC_inter_roi)

    gp_factor_roi = np.mean(mismatch)
    ind_factor_roi = np.mean(match) - np.mean(mismatch)

    return match, mismatch, gp_factor_roi, ind_factor_roi, t, p



def get_roi_group_ind_result(PredFC, FC):
    roi_num = PredFC.shape[1]

    with multiprocessing.Pool() as p:
        results = p.map(get_roi_inter_matrix,  [(PredFC, FC, i) for i in range(roi_num)])

    ind_match_roi = np.array([res[0] for res in results])
    ind_mismatch_roi = np.array([res[1] for res in results])
    gp_factor_roi = np.array([res[2] for res in results])
    ind_factor_roi = np.array([res[3] for res in results])
    ind_factor_t = np.array([res[4] for res in results])
    ind_factor_p = np.array([res[5] for res in results])

    return ind_match_roi, ind_mismatch_roi, gp_factor_roi, ind_factor_roi, ind_factor_t, ind_factor_p



if __name__ == '__main__':

    # parser settings
    parser = argparse.ArgumentParser(description='GNN SC FC')
    parser.add_argument('--dataset', default='HCPD', help='choose a dataset')


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

    predFC_FC_match_roi, predFC_FC_mismatch_roi, predFC_FC_gp, predFC_FC_ind, predFC_FC_ind_t, predFC_FC_ind_p = get_roi_group_ind_result(predFC_test, FC_test)

    with open(
            '../data/result_out_2/' + dataset_type + '/Pred/'+dataset_type + '_roi_gp_ind_t_p.pickle',
            'wb') as in_data:
        pickle.dump((predFC_FC_match_roi, predFC_FC_mismatch_roi, predFC_FC_gp, predFC_FC_ind, predFC_FC_ind_t, predFC_FC_ind_p), in_data, pickle.HIGHEST_PROTOCOL)