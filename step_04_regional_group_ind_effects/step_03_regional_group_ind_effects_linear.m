clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_04_regional_group_ind_effects/';

%% S-A rank correlation
[sa_rank,~,raw] = xlsread('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/schaefer400_sa_rank.xlsx');
sa_rank = sa_rank(:,2);

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

% Individual correlation
intra_mask = eye(n_hcp,n_hcp);
group_effect_hcp = zeros(400,1);
ind_effect_hcp = zeros(400,1);

for roi_i = 1:400
    roi_i
    idx = setdiff(1:400,roi_i);
    eFC_vec_hcp = eFC_hcp(:,idx,roi_i);
    SC_vec_hcp = SC_hcp(:,idx,roi_i);

    for i = 1:n_hcp
        SC_temp = SC_vec_hcp(i,:)';
        for j = 1:n_hcp
            eFC_temp = eFC_vec_hcp(j,:)';
            SC_eFC_corr_hcp(i,j) = corr(log(SC_temp(SC_temp>0)),eFC_temp(SC_temp>0));
        end
    end

    group_effect_hcp(roi_i,1) = mean(SC_eFC_corr_hcp(intra_mask == 0));
    ind_effect_hcp(roi_i,1) = mean(SC_eFC_corr_hcp(intra_mask == 1)) - group_effect_hcp(roi_i,1);

    group_effect_norm_hcp(roi_i,1) = group_effect_hcp(roi_i,1) ./ mean(SC_eFC_corr_hcp(intra_mask == 1));
    ind_effect_norm_hcp(roi_i,1) = ind_effect_hcp(roi_i,1) ./ mean(SC_eFC_corr_hcp(intra_mask == 1));
end

save([working_dir,'hcp_SC_eFC_group_ind_effect.mat'],'group_effect_hcp','ind_effect_hcp','group_effect_norm_hcp','ind_effect_norm_hcp')

%-----------
load([working_dir,'hcp_SC_eFC_group_ind_effect.mat'])
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/perm_id_schaefer400.mat')
rand_num = 10000;

% group effects sa-rank correlation spin test
non_outlier = abs(zscore(group_effect_hcp)) <= 3;
r_group_effect_hcp = corr(group_effect_hcp(non_outlier),sa_rank(non_outlier),"type","Spearman")

for i = 1:rand_num
    group_effect_hcp_temp = group_effect_hcp(perm_id(:,i));
    r_group_effect_hcp_spin(i,1) = corr(group_effect_hcp_temp(non_outlier),sa_rank(non_outlier),"type","Spearman");
end

if r_group_effect_hcp > 0
    p_group_effect_hcp_spin = (1+length(find(r_group_effect_hcp_spin >= r_group_effect_hcp)))/(rand_num+1);
else
    p_group_effect_hcp_spin = (1+length(find(r_group_effect_hcp_spin <= r_group_effect_hcp)))/(rand_num+1);
end

% individual effects sa-rank correlation spin test
non_outlier = abs(zscore(ind_effect_hcp)) <= 3;
r_ind_effect_hcp = corr(ind_effect_hcp(non_outlier),sa_rank(non_outlier),"type","Spearman")

for i = 1:rand_num
    ind_effect_hcp_temp = ind_effect_hcp(perm_id(:,i));
    r_ind_effect_hcp_spin(i,1) = corr(ind_effect_hcp_temp(non_outlier),sa_rank(non_outlier),"type","Spearman");
end

if r_ind_effect_hcp > 0
    p_ind_effect_hcp_spin = (1+length(find(r_ind_effect_hcp_spin >= r_ind_effect_hcp)))/(rand_num+1);
else
    p_ind_effect_hcp_spin = (1+length(find(r_ind_effect_hcp_spin <= r_ind_effect_hcp)))/(rand_num+1);
end

% save the results
hcp.group_effect = group_effect_hcp;
hcp.ind_effect = ind_effect_hcp;
hcp.sa_rank = sa_rank;
hcp = struct2table(hcp);
writetable(hcp,[working_dir '/hcp_SC_eFC_group_ind_effect.csv'])

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

% Individual correlation
intra_mask = eye(n_hcpd,n_hcpd);
group_effect_hcpd = zeros(400,1);
ind_effect_hcpd = zeros(400,1);

for roi_i = 1:400
    roi_i
    idx = setdiff(1:400,roi_i);
    eFC_vec_hcpd = eFC_hcpd(:,idx,roi_i);
    SC_vec_hcpd = SC_hcpd(:,idx,roi_i);

    for i = 1:n_hcpd
        SC_temp = SC_vec_hcpd(i,:)';
        for j = 1:n_hcpd
            eFC_temp = eFC_vec_hcpd(j,:)';
            SC_eFC_corr_hcpd(i,j) = corr(log(SC_temp(SC_temp>0)),eFC_temp(SC_temp>0));
        end
    end

    group_effect_hcpd(roi_i,1) = mean(SC_eFC_corr_hcpd(intra_mask == 0));
    ind_effect_hcpd(roi_i,1) = mean(SC_eFC_corr_hcpd(intra_mask == 1)) - group_effect_hcpd(roi_i,1);

    group_effect_norm_hcpd(roi_i,1) = group_effect_hcpd(roi_i,1) ./ mean(SC_eFC_corr_hcpd(intra_mask == 1));
    ind_effect_norm_hcpd(roi_i,1) = ind_effect_hcpd(roi_i,1) ./ mean(SC_eFC_corr_hcpd(intra_mask == 1));
end

save([working_dir,'hcpd_SC_eFC_group_ind_effect.mat'],'group_effect_hcpd','ind_effect_hcpd','group_effect_norm_hcpd','ind_effect_norm_hcpd')

%---------
load([working_dir,'hcpd_SC_eFC_group_ind_effect.mat'])
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/perm_id_schaefer400.mat')
rand_num = 10000;
% group effects sa-rank correlation spin test
non_outlier = abs(zscore(group_effect_hcpd)) <= 3;
r_group_effect_hcpd = corr(group_effect_hcpd(non_outlier),sa_rank(non_outlier),"type","Spearman")

for i = 1:rand_num
    group_effect_hcpd_temp = group_effect_hcpd(perm_id(:,i));
    r_group_effect_hcpd_spin(i,1) = corr(group_effect_hcpd_temp(non_outlier),sa_rank(non_outlier),"type","Spearman");
end

if r_group_effect_hcpd > 0
    p_group_effect_hcpd_spin = (1+length(find(r_group_effect_hcpd_spin >= r_group_effect_hcpd)))/(rand_num+1);
else
    p_group_effect_hcpd_spin = (1+length(find(r_group_effect_hcpd_spin <= r_group_effect_hcpd)))/(rand_num+1);
end

% individual effects sa-rank correlation spin test
non_outlier = abs(zscore(ind_effect_hcpd)) <= 3;
r_ind_effect_hcpd = corr(ind_effect_hcpd(non_outlier),sa_rank(non_outlier),"type","Spearman")

for i = 1:rand_num
    ind_effect_hcpd_temp = ind_effect_hcpd(perm_id(:,i));
    r_ind_effect_hcpd_spin(i,1) = corr(ind_effect_hcpd_temp(non_outlier),sa_rank(non_outlier),"type","Spearman");
end

if r_ind_effect_hcpd > 0
    p_ind_effect_hcpd_spin = (1+length(find(r_ind_effect_hcpd_spin >= r_ind_effect_hcpd)))/(rand_num+1);
else
    p_ind_effect_hcpd_spin = (1+length(find(r_ind_effect_hcpd_spin <= r_ind_effect_hcpd)))/(rand_num+1);
end

% save the results
hcpd.group_effect = group_effect_hcpd;
hcpd.ind_effect = ind_effect_hcpd;
hcpd.sa_rank = sa_rank;
hcpd = struct2table(hcpd);
writetable(hcpd,[working_dir '/hcpd_SC_eFC_group_ind_effect.csv'])

%% S-A rank correlation
[sa_rank,~,sa_rank_raw] = xlsread('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/schaefer400_sa_rank.xlsx');
sa_rank_raw(1,1:5) = {'label','group_effect_hcp','ind_effect_hcp','group_effect_hcpd','ind_effect_hcpd'};
sa_rank_raw(2:401,2:5) = num2cell([group_effect_hcp,ind_effect_hcp,group_effect_hcpd,ind_effect_hcpd]);

for i = 2:201
    sa_rank_raw{i,1} = ['lh_' sa_rank_raw{i,1}];
end

for i = 202:401
    sa_rank_raw{i,1} = ['rh_' sa_rank_raw{i,1}];
end

writecell(sa_rank_raw,[working_dir '/SC_eFC_group_ind_effect.csv'])

