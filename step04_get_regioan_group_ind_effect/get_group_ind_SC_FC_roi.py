import sys
sys.path.append("../functions")
from utils import *
import pickle
import bct
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import argparse


def get_roi_inter_matrix_SC_FC(args):
    SC, FC, roi = args
    sub_num = SC.shape[0]
    SC_FC_inter_roi = np.zeros((sub_num, sub_num))

    for i in range(SC.shape[0]):
        SC_i = SC[i, :, :]

        for j in range(SC.shape[0]):
            FC_j = FC[j, :, :]

            SC_i_roi = SC_i[roi, :]
            SC_i_roi = np.delete(SC_i_roi, roi)
            nozero_index_i = np.where(SC_i_roi > 0)[0]
            SC_i_roi = SC_i_roi[nozero_index_i]
            FC_j_roi = FC_j[roi, :]
            FC_j_roi = np.delete(FC_j_roi, roi)
            FC_j_roi = FC_j_roi[nozero_index_i]
            SC_FC_inter_roi[i, j] = np.nan_to_num(np.corrcoef(np.log(SC_i_roi), FC_j_roi)[0, 1])

    match, mismatch, t, p = get_match_mismatch_t_p(SC_FC_inter_roi)
    gp_factor_roi = np.mean(mismatch)
    ind_factor_roi = np.mean(match) - np.mean(mismatch)

    return match, mismatch, gp_factor_roi, ind_factor_roi, t, p



def get_roi_group_ind_result_SC_FC(SC, FC, SC_mask):
    for i in range(SC.shape[0]):
        SC[i, :, :] = SC[i, :, :] * SC_mask

    roi_num = SC.shape[1]
    with multiprocessing.Pool() as p:
        results = p.map(get_roi_inter_matrix_SC_FC,  [(SC, FC, i) for i in range(roi_num)])

    match_roi = np.array([res[0] for res in results])
    mismatch_roi = np.array([res[1] for res in results])
    gp_factor_roi = np.array([res[2] for res in results])
    ind_factor_roi = np.array([res[3] for res in results])
    ind_factor_t = np.array([res[4] for res in results])
    ind_factor_p = np.array([res[5] for res in results])

    return match_roi, mismatch_roi, gp_factor_roi, ind_factor_roi, ind_factor_t, ind_factor_p


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GNN SC FC')
    parser.add_argument('--dataset', default='ABCD', help='choose a dataset')
    parser.add_argument('--mask_type', default=75, help='choose a dataset')

    args = parser.parse_args()
    dataset_type = args.dataset

    with open(
            '../data/info/SC_mask_' + str(args.mask_type) + '.pickle',
            'rb') as out_data:
        (HCPA_sc_mask, HCPD_sc_mask, ABCD_sc_mask) = pickle.load(out_data)

    if dataset_type == 'HCPA':
        with open('../data/preprocessed_data/final/HCPA_SC_FC_reSC_info_245.pickle',
                  'rb') as out_data:
            (SC, FC, SC_re, info) = pickle.load(out_data)
            SC_mask = HCPA_sc_mask

    elif dataset_type == 'HCPD':
        with open(
                    '../data/preprocessed_data/HCPD_SC_FC_reSC_info_410.pickle',
                    'rb') as out_data:
            (SC, FC, re_SC, info) = pickle.load(out_data)
            SC_mask = HCPD_sc_mask
    else:

        with open('../data/preprocessed_data/ABCD_SC_FC_reSC_info_1572.pickle',
                    'rb') as out_data:
            (SC, FC, SC_re, info) = pickle.load(out_data)
            SC_mask = ABCD_sc_mask



    SC_FC_match_roi, SC_FC_mismatch_roi, SC_FC_gp, SC_FC_ind, SC_FC_ind_t, SC_FC_ind_p = get_roi_group_ind_result_SC_FC(SC, FC, SC_mask)

    # with open(
    #         '../data/result_out_2/' + dataset_type + '/SC/'+dataset_type + '_roi_gp_ind_t_p_' +  str(args.mask_type) + '.pickle',
    #         'wb') as in_data:
    #     pickle.dump((SC_FC_match_roi, SC_FC_mismatch_roi, SC_FC_gp, SC_FC_ind, SC_FC_ind_t, SC_FC_ind_p), in_data, pickle.HIGHEST_PROTOCOL)
