clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_04_regional_group_ind_effects/';

%% S-A rank correlation
[sa_rank,~,raw] = xlsread('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/schaefer200_sa_rank.xlsx');

%% HCP-YA
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/val/HCPA_PredFC_FC_test_sch200.mat')
eFC = HCPA_FC_test;
pFC = double(HCPA_PredFC_test);

[n_hcp,~,~] = size(HCPA_PredFC_test);
intra_mask = eye(n_hcp,n_hcp);

for roi_i = 1:200
    idx = setdiff(1:200,roi_i);
    eFC_vec = eFC(:,idx,roi_i);
    pFC_vec = pFC(:,idx,roi_i);

    eFC_pFC_corr = corr(eFC_vec',pFC_vec', "type","Pearson" );
    group_effect_hcp(roi_i,1) = mean(eFC_pFC_corr(intra_mask == 0));
    ind_effect_hcp(roi_i,1) = mean(eFC_pFC_corr(intra_mask == 1)) - group_effect_hcp(roi_i,1);

    group_effect_norm_hcp(roi_i,1) = group_effect_hcp(roi_i,1) ./ mean(eFC_pFC_corr(intra_mask == 1));
    ind_effect_norm_hcp(roi_i,1) = ind_effect_hcp(roi_i,1) ./ mean(eFC_pFC_corr(intra_mask == 1));

end

save([working_dir,'hcp_pFC_eFC_group_ind_effect_schaefer200.mat'],'group_effect_hcp','ind_effect_hcp','group_effect_norm_hcp','ind_effect_norm_hcp')

%-----------
load([working_dir,'hcp_pFC_eFC_group_ind_effect_schaefer200.mat'])
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/perm_id_schaefer200.mat')
rand_num = 10000;

% group effects sa-rank correlation spin test
[r_group_effect_hcp,p_group_effect_hcp_spin] = corr_sa_spin(group_effect_hcp,sa_rank,perm_id,rand_num);

% individual effects sa-rank correlation spin test
[r_ind_effect_hcp,p_ind_effect_hcp_spin] = corr_sa_spin(ind_effect_hcp,sa_rank,perm_id,rand_num);
[r_ind_effect_norm_hcp,p_ind_effect_norm_hcp_spin] = corr_sa_spin(ind_effect_norm_hcp,sa_rank,perm_id,rand_num);

% save the results
hcp.group_effect = group_effect_hcp;
hcp.ind_effect = ind_effect_hcp;
hcp.group_effect_norm = group_effect_norm_hcp;
hcp.ind_effect_norm = ind_effect_norm_hcp;
hcp.sa_rank = sa_rank;
hcp = struct2table(hcp);
writetable(hcp,[working_dir '/hcp_pFC_eFC_group_ind_effect_schaefer200.csv'])

%% HCP-D
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/val/HCPD_PredFC_FC_test_sch200.mat')
eFC = HCPD_FC_test;
pFC = double(HCPD_PredFC_test);

[n_hcpd,~,~] = size(HCPD_PredFC_test);
intra_mask = eye(n_hcpd,n_hcpd);

for roi_i = 1:200
    idx = setdiff(1:200,roi_i);
    eFC_vec = eFC(:,idx,roi_i);
    pFC_vec = pFC(:,idx,roi_i);

    eFC_pFC_corr = corr(eFC_vec',pFC_vec', "type","Pearson" );

    group_effect_hcpd(roi_i,1) = mean(eFC_pFC_corr(intra_mask == 0));
    ind_effect_hcpd(roi_i,1) = mean(eFC_pFC_corr(intra_mask == 1)) - group_effect_hcpd(roi_i,1);

    group_effect_norm_hcpd(roi_i,1) = group_effect_hcpd(roi_i,1) ./ mean(eFC_pFC_corr(intra_mask == 1));
    ind_effect_norm_hcpd(roi_i,1) = ind_effect_hcpd(roi_i,1) ./ mean(eFC_pFC_corr(intra_mask == 1));
end

save([working_dir,'hcpd_pFC_eFC_group_ind_effect_schaefer200.mat'],'group_effect_hcpd','ind_effect_hcpd','group_effect_norm_hcpd','ind_effect_norm_hcpd')

load([working_dir,'hcpd_pFC_eFC_group_ind_effect_schaefer200.mat'])
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/perm_id_schaefer200.mat')
rand_num = 10000;
% group effects sa-rank correlation spin test
[r_group_effect_hcpd,p_group_effect_hcpd_spin] = corr_sa_spin(group_effect_hcpd,sa_rank,perm_id,rand_num);

% individual effects sa-rank correlation spin test
[r_ind_effect_hcpd,p_ind_effect_hcpd_spin] = corr_sa_spin(ind_effect_hcpd,sa_rank,perm_id,rand_num);
[r_ind_effect_norm_hcpd,p_ind_effect_norm_hcpd_spin] = corr_sa_spin(ind_effect_norm_hcpd,sa_rank,perm_id,rand_num);

% save the results
hcpd.group_effect = group_effect_hcpd;
hcpd.ind_effect = ind_effect_hcpd;
hcpd.group_effect_norm = group_effect_norm_hcpd;
hcpd.ind_effect_norm = ind_effect_norm_hcpd;
hcpd.sa_rank = sa_rank;
hcpd = struct2table(hcpd);
writetable(hcpd,[working_dir '/hcpd_pFC_eFC_group_ind_effect_schaefer200.csv'])

%% S-A rank correlation
[sa_rank,~,sa_rank_raw] = xlsread('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/schaefer200_sa_rank.xlsx');
sa_rank_raw(1,1:5) = {'label','group_effect_hcp','ind_effect_hcp','group_effect_hcpd','ind_effect_hcpd'};
sa_rank_raw(2:201,2:5) = num2cell([group_effect_hcp,ind_effect_hcp,group_effect_hcpd,ind_effect_hcpd]);

for i = 2:101
    sa_rank_raw{i,1} = ['lh_' sa_rank_raw{i,1}];
end

for i = 102:201
    sa_rank_raw{i,1} = ['rh_' sa_rank_raw{i,1}];
end

writecell(sa_rank_raw,[working_dir '/pFC_eFC_group_ind_effect_schaefer200.csv'])

%%
[sa_rank,~,sa_rank_norm] = xlsread('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/schaefer200_sa_rank.xlsx');
sa_rank_norm(1,1:5) = {'label','group_effect_norm_hcp','ind_effect_norm_hcp','group_effect_norm_hcpd','ind_effect_norm_hcpd'};
sa_rank_norm(2:201,2:5) = num2cell([group_effect_norm_hcp,ind_effect_norm_hcp,group_effect_norm_hcpd,ind_effect_norm_hcpd]);

% for i = 2:101
%     sa_rank_norm{i,1} = ['lh_' sa_rank_norm{i,1}];
% end
% 
% for i = 102:201
%     sa_rank_norm{i,1} = ['rh_' sa_rank_norm{i,1}];
% end
% 
% writecell(sa_rank_norm,[working_dir '/pFC_eFC_group_ind_effect_norm_schaefer200.csv'])
