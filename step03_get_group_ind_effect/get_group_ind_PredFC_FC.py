import pickle
import bct
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.stats as stats
from utils import *
import scipy.io as sio
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import scipy
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


def get_IntrinNetwork_group_ind_r_result(PredFC, FC):
    region2index_dict, index2region_dict, name = get_intrinsic_index()

    gp_factors_sys = []
    ind_factors_sys = []
    match_sys = []
    mismatch_sys = []


    pro_region = region2index_dict[0].tolist() + region2index_dict[1].tolist()
    con_region = region2index_dict[2].tolist() + region2index_dict[3].tolist() + region2index_dict[5].tolist()

    PredFC_region = PredFC[:, pro_region, :]
    PredFC_region = PredFC_region[:, :, pro_region]
    FC_region = FC[:, pro_region, :]
    FC_region = FC_region[:, :, pro_region]
    inter_r = get_intermatrix(PredFC_region, FC_region)
    match, mismatch, t, p = get_match_mismatch_t_p(inter_r)
    gp_factor = np.mean(mismatch)
    ind_factor = np.mean(match) - np.mean(mismatch)
    gp_factors_sys.append(gp_factor)
    ind_factors_sys.append(ind_factor)
    match_sys.append(match)
    mismatch_sys.append(mismatch)

    PredFC_region = PredFC[:, con_region, :]
    PredFC_region = PredFC_region[:, :, con_region]
    FC_region = FC[:, con_region, :]
    FC_region = FC_region[:, :, con_region]
    inter_r = get_intermatrix(PredFC_region, FC_region)
    match, mismatch, t, p = get_match_mismatch_t_p(inter_r)
    gp_factor = np.mean(mismatch)
    ind_factor = np.mean(match) - np.mean(mismatch)
    gp_factors_sys.append(gp_factor)
    ind_factors_sys.append(ind_factor)
    match_sys.append(match)
    mismatch_sys.append(mismatch)

    return match_sys, mismatch_sys, gp_factors_sys, ind_factors_sys




if __name__ == '__main__':

    # parser settings
    parser = argparse.ArgumentParser(description='GNN SC FC')
    parser.add_argument('--dataset', default='HCPD', help='choose a dataset')


    args = parser.parse_args()

    dataset_type = args.dataset


    if dataset_type == 'HCPA':
        with open('/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/HCPA_SC_FC_reSC_info_245_final.pickle',
                  'rb') as out_data:
            (SC, FC, SC_re, info) = pickle.load(out_data)

        SC_train, SC_test, FC_train, FC_test = train_test_split(SC, FC, test_size=0.5, random_state=0)
        with open('../data/result_out/HCPA/GCN_dim128lr0.001b2ep200reg0.0prelu.pickle', 'rb') as out_data:
            predFC_train, predFC_test = pickle.load(out_data)

    elif dataset_type == 'HCPD':
        with open(
                    '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/HCPD_SC_FC_reSC_info_410_final.pickle',
                    'rb') as out_data:
            (SC, FC, re_SC, info) = pickle.load(out_data)
        SC_train, SC_test, FC_train, FC_test = train_test_split(SC, FC, test_size=0.5,
                                                                                          random_state=0)
        with open('result_out/HCPD/GCN_dim128lr0.001b2ep400reg0.0prelu.pickle', 'rb') as out_data:
            predFC_train, predFC_test = pickle.load(out_data)
    else:
        # with open(
        #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/ABCD_SC_FC_reSC_info_1301_final.pickle',
        #         'rb') as out_data:
        with open( '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/ABCD_SC_FC_reSC_info_1572_4run.pickle',
                    'rb') as out_data:
            (SC, FC, SC_re, info) = pickle.load(out_data)
        SC_train, SC_test, FC_train, FC_test = train_test_split(SC, FC, test_size=0.5,
                                                                     random_state=0)

        with open('result_out/ABCD/GCN_dim128lr0.001b2ep600reg0.0prelu.pickle', 'rb') as out_data:
            predFC_train, predFC_test = pickle.load(out_data)

    # match, mismatch, gp_factors_sys, ind_factors_sys = get_IntrinNetwork_group_ind_r_result(predFC_test, FC_test)
    # print(dataset_type)
    # print(gp_factors_sys, ind_factors_sys)
    # print(stats.ttest_rel(match[0]-mismatch[0], match[1]-mismatch[1]))


    predFC_FC_inter_r = get_intermatrix(predFC_test, FC_test)
    np.save('/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out_2/' + dataset_type + '/Pred/'+dataset_type + '_inter_matrix_Pred_FC.npy', predFC_FC_inter_r)


    # predFC_FC_match_roi, predFC_FC_mismatch_roi, predFC_FC_gp, predFC_FC_ind, predFC_FC_ind_t, predFC_FC_ind_p = get_roi_group_ind_result(predFC_test, FC_test)
    #
    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out_2/' + dataset_type + '/Pred/'+dataset_type + '_roi_gp_ind_t_p_0926_600.pickle',
    #         'wb') as in_data:
    #     pickle.dump((predFC_FC_match_roi, predFC_FC_mismatch_roi, predFC_FC_gp, predFC_FC_ind, predFC_FC_ind_t, predFC_FC_ind_p), in_data, pickle.HIGHEST_PROTOCOL)


    # PredFC_FC_ind_match_roi, PredFC_FC_ind_mismatch_roi, PredFC_FC_gp, PredFC_FC_ind, PredFC_FC_ind_t, PredFC_FC_ind_p = get_roi_group_ind_r_result(predFC_test, FC_test, dataset_type, mode=mode, corrtype=corrtype)

    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out/roi/PredFC/'+dataset_type + '_' + mode + '_roi_' + corrtype + '_gp_ind_t_p_0814_Pred.pickle',
    #         'wb') as in_data:
    #     pickle.dump((PredFC_FC_ind_match_roi, PredFC_FC_ind_mismatch_roi, PredFC_FC_gp, PredFC_FC_ind, PredFC_FC_ind_t, PredFC_FC_ind_p), in_data, pickle.HIGHEST_PROTOCOL)
    #


    # with open(
    #             '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/HCPD_SC_FC_reSC_info_410_final.pickle',
    #             'rb') as out_data:
    #     (HCPD_SC, HCPD_FC, re_SC, info) = pickle.load(out_data)
    # SC_train, SC_test, FC_train, HCPD_FC_test = train_test_split(HCPD_SC, HCPD_FC, test_size=0.3,
    #                                                                                   random_state=0)
    # with open('result_out/HCPD/GCN_dim128_reg.pickle', 'rb') as out_data:
    #     HCPD_predFC_train, HCPD_predFC_test = pickle.load(out_data)
    #
    #
    # HCPD_SC_FC_gp, HCPD_SC_FC_ind, HCPD_SC_FC_ind_t, HCPD_SC_FC_ind_p = get_group_ind_rho_SC_result(HCPD_SC, HCPD_FC, 'HCPD')
    #
    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out/HCPD_roi_gp_ind_t_p_0629.pickle',
    #         'wb') as in_data:
    #     pickle.dump((HCPD_SC_FC_gp, HCPD_SC_FC_ind, HCPD_SC_FC_ind_t, HCPD_SC_FC_ind_p), in_data, pickle.HIGHEST_PROTOCOL)
    #

    # HCPD_SC_FC_inter_r, HCPD_SC_FC_inter_r_ind = get_inter_r_SC_result(HCPD_SC, HCPD_FC, 'HCPD')
    # HCPD_PredFC_FC_test_inter_r, HCPD_PredFC_FC_test_inter_r_ind = get_inter_r_PredFC_result(HCPD_predFC_test, HCPD_FC_test, 'HCPD')
    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out/inter_r_HCPD_0623.pickle',
    #         'wb') as in_data:
    #     pickle.dump((HCPD_SC_FC_inter_r, HCPD_SC_FC_inter_r_ind, HCPD_PredFC_FC_test_inter_r, HCPD_PredFC_FC_test_inter_r_ind), in_data, pickle.HIGHEST_PROTOCOL)
    #


    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/ABCD_SC_FC_reSC_info_1301_final.pickle',
    #         'rb') as out_data:
    #     (ABCD_SC, ABCD_FC, SC_re, info) = pickle.load(out_data)
    # SC_train, SC_test, FC_train, ABCD_FC_test = train_test_split(ABCD_SC, ABCD_FC, test_size=0.3,
    #                                                                                   random_state=0)
    #
    # ABCD_SC_FC_gp, ABCD_SC_FC_ind, ABCD_SC_FC_ind_t, ABCD_SC_FC_ind_p = get_group_ind_rho_SC_result(ABCD_SC, ABCD_FC, 'ABCD')
    #
    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out/ABCD_roi_gp_ind_t_p_0629.pickle',
    #         'wb') as in_data:
    #     pickle.dump((ABCD_SC_FC_gp, ABCD_SC_FC_ind, ABCD_SC_FC_ind_t, ABCD_SC_FC_ind_p), in_data, pickle.HIGHEST_PROTOCOL)

    # #

    # ABCD_SC_FC_inter_r, ABCD_SC_FC_inter_r_ind = get_inter_r_SC_result(ABCD_SC, ABCD_FC, 'ABCD')
    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out/inter_r_ABCD_0620.pickle',
    #         'wb') as in_data:
    #     pickle.dump((ABCD_SC_FC_inter_r, ABCD_SC_FC_inter_r_ind), in_data, pickle.HIGHEST_PROTOCOL)
    #

    # HCPD_SC_FC_roi_inter_rho = get_region_inter_rho_SC_result(HCPD_SC, HCPD_FC, 'HCPD')
    # ABCD_SC_FC_roi_inter_rho = get_region_inter_rho_SC_result(ABCD_SC, ABCD_FC, 'ABCD')
    #
    # with open(
    #         '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out/roi_inter_r_all_0627.pickle',
    #         'wb') as in_data:
    #     pickle.dump((HCPA_SC_FC_roi_inter_rho, HCPD_SC_FC_roi_inter_rho, ABCD_SC_FC_roi_inter_rho), in_data, pickle.HIGHEST_PROTOCOL)
    #
