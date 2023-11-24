clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/';

%% S-A rank correlation
[sa_rank,~,raw] = xlsread('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/schaefer400_sa_rank.xlsx');
sa_rank = sa_rank(:,2);

%% HCP-YA
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPA_PredFC_FC_test.mat')
eFC = HCPA_FC_test;
pFC = double(HCPA_PredFC_test);

[n_hcp,~,~] = size(HCPA_PredFC_test);
intra_mask = eye(n_hcp,n_hcp);

for roi_i = 1:400
    for idx = 1:n_hcp
    eFC_vec = squeeze(eFC(idx,roi_i, :));
    pFC_vec = squeeze(pFC(idx,roi_i, :));
    eFC_pFC_corr_hcp(roi_i,idx) = corr(eFC_vec,pFC_vec, "type","Pearson" );
    end 
end

%% HCP-D
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_PredFC_FC_test.mat')
eFC = HCPD_FC_test;
pFC = double(HCPD_PredFC_test);

[n_hcpd,~,~] = size(HCPD_PredFC_test);
intra_mask = eye(n_hcp,n_hcp);

for roi_i = 1:400
    for idx = 1:n_hcpd
    eFC_vec = squeeze(eFC(idx,roi_i, :));
    pFC_vec = squeeze(pFC(idx,roi_i, :));
    eFC_pFC_corr_hcpd(roi_i,idx) = corr(eFC_vec,pFC_vec, "type","Pearson" );
    end 
end

hcp.GNN_cp = mean(eFC_pFC_corr_hcp, 2);
hcp = struct2table(hcp);
writetable(hcp,[working_dir '/region_hcp_pFC_eFC_cp.csv'])


hcpd.GNN_cp = mean(eFC_pFC_corr_hcpd, 2);
hcpd = struct2table(hcpd);
writetable(hcpd,[working_dir '/region_hcpd_pFC_eFC_cp.csv'])