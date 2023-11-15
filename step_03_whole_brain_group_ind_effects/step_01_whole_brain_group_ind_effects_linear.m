clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_03_whole_brain_group_ind_effects/';
cd(working_dir)
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

% Individual correlation
for i = 1:n_hcp
    i
    SC_temp = mat2vec(SC_hcp(:,:,i))';
    for j = 1:n_hcp
        eFC_temp = mat2vec(eFC_hcp(:,:,j))';
        corr_SC_eFC_hcp(i,j) = corr(log(SC_temp(SC_temp>0)),eFC_temp(SC_temp>0));
    end
end

[group_effect_hcp,ind_effect_hcp,match_corr_hcp,mismatch_corr_hcp] = get_group_ind_effects(corr_SC_eFC_hcp);
[h,p,~,stats] = ttest(match_corr_hcp,mismatch_corr_hcp)
save('hcp_group_ind_effects_SC_eFC.mat','corr_SC_eFC_hcp','group_effect_hcp','ind_effect_hcp','match_corr_hcp','mismatch_corr_hcp')

load('hcp_group_ind_effects_SC_eFC.mat')
ind_effect_hcp/(group_effect_hcp+ind_effect_hcp)
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

% Individual correlation
for i = 1:n_hcpd
    i
    SC_temp = mat2vec(SC_hcpd(:,:,i))';
    for j = 1:n_hcpd
        eFC_temp = mat2vec(eFC_hcpd(:,:,j))';
        corr_SC_eFC_hcpd(i,j) = corr(log(SC_temp(SC_temp>0)),eFC_temp(SC_temp>0));
    end
end

[group_effect_hcpd,ind_effect_hcpd,match_corr_hcpd,mismatch_corr_hcpd] = get_group_ind_effects(corr_SC_eFC_hcpd);
[h,p,~,stats] = ttest(match_corr_hcpd,mismatch_corr_hcpd)
ind_effect_hcpd/(group_effect_hcpd+ind_effect_hcpd)
save('hcpd_group_ind_effects_SC_eFC.mat','corr_SC_eFC_hcpd','group_effect_hcpd','ind_effect_hcpd','match_corr_hcpd','mismatch_corr_hcpd')

load('hcpd_group_ind_effects_SC_eFC.mat')
ind_effect_hcpd/(group_effect_hcpd+ind_effect_hcpd)
%%
SC_eFC_match.data = [match_corr_hcp;mismatch_corr_hcp;match_corr_hcpd;mismatch_corr_hcpd];
SC_eFC_match.type = [repmat({'Matched'},[n_hcp,1]);repmat({'Mismatched'},[n_hcp,1]);...
    repmat({'Matched'},[n_hcpd,1]);repmat({'Mismatched'},[n_hcpd,1]);];
SC_eFC_match.dataset = [repmat({'HCP-YA'},[n_hcp*2,1]);repmat({'HCP-D'},[n_hcpd*2,1])];

SC_eFC_match = struct2table(SC_eFC_match);
writetable(SC_eFC_match,[working_dir 'SC_eFC_match.csv'])

