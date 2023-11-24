clear
clc


%% S-A rank correlation
[sa_rank,~,sa_rank_raw] = xlsread('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/schaefer400_sa_rank.xlsx');

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/';

%% HCP-YA

data_hcp = load("/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCP_SC_FC.mat");
SC_hcp = data_hcp.HCP_SC;
[~,~,n_hcp] = size(SC_hcp);

W_thr = threshold_consistency(SC_hcp, 0.75);
SC_mask_hcp = double(W_thr > 0);

eFC_hcp = data_hcp.HCP_FC;

for sub_i = 1:n_hcp
    SC_hcp(:,:,sub_i) = SC_hcp(:,:,sub_i) .* SC_mask_hcp;
end

eFC_hcp = permute(eFC_hcp,[3,1,2]);
SC_hcp = permute(SC_hcp,[3,1,2]);


for roi_i = 1:400
    for idx = 1:n_hcp
    eFC_vec = squeeze(eFC_hcp(idx,roi_i, :));
    SC_vec = squeeze(SC_hcp(idx,roi_i, :));
     SC_eFC_corr_hcp(roi_i,idx) = corr(log(SC_vec(SC_vec>0)),eFC_vec(SC_vec>0), "type","Pearson");
    end
end

%% HCP-D

data_hcpd = load("/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_SC_FC.mat");
SC_hcpd = data_hcpd.HCPD_SC;
[~,~,n_hcpd] = size(SC_hcpd);

W_thr = threshold_consistency(SC_hcpd, 0.75);
SC_mask_hcpd = double(W_thr > 0);

eFC_hcpd = data_hcpd.HCPD_FC;

for sub_i = 1:n_hcpd
    SC_hcpd(:,:,sub_i) = SC_hcpd(:,:,sub_i) .* SC_mask_hcpd;
end

eFC_hcpd = permute(eFC_hcpd,[3,1,2]);
SC_hcpd = permute(SC_hcpd,[3,1,2]);


for roi_i = 1:400
    for idx = 1:n_hcpd
    eFC_vec = squeeze(eFC_hcpd(idx,roi_i, :));
    SC_vec = squeeze(SC_hcpd(idx,roi_i, :));
     SC_eFC_corr_hcpd(roi_i,idx) = corr(log(SC_vec(SC_vec>0)),eFC_vec(SC_vec>0), "type","Pearson");
    end
end



sa_rank_raw(1,1:2) = {'label','lin_cp'};
sa_rank_raw(2:401,2) = num2cell(mean(SC_eFC_corr_hcp, 2));

for i = 2:201
    sa_rank_raw{i,1} = ['lh_' sa_rank_raw{i,1}];
end

for i = 202:401
    sa_rank_raw{i,1} = ['rh_' sa_rank_raw{i,1}];
end

writecell(sa_rank_raw(:, 1:2),[working_dir '/region_hcp_SC_eFC_cp.csv'])


sa_rank_raw(2:401,2) = num2cell(mean(SC_eFC_corr_hcpd, 2));
writecell(sa_rank_raw(:, 1:2),[working_dir '/region_hcpd_SC_eFC_cp.csv'])

