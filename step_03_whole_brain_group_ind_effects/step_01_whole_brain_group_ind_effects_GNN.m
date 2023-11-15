clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_03_whole_brain_group_ind_effects/';

%% HCP-YA
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPA_PredFC_FC_test.mat')
eFC = permute(HCPA_FC_test,[2,3,1]);
pFC = permute(HCPA_PredFC_test,[2,3,1]);
pFC = double(pFC);

[n_hcp,~,~] = size(HCPA_FC_test);
intra_mask = eye(n_hcp,n_hcp);

for sub_i = 1:n_hcp
    eFC_vec(sub_i,:) = mat2vec(eFC(:,:,sub_i));
    pFC_vec(sub_i,:) = mat2vec(pFC(:,:,sub_i));
end
% corr_pFC_eFC_hcp = corr(eFC_vec',pFC_vec');
corr_pFC_eFC_hcp = corr(eFC_vec',pFC_vec', "type","Pearson" );
[group_effect_hcp,ind_effect_hcp,match_corr_hcp,mismatch_corr_hcp] = get_group_ind_effects(corr_pFC_eFC_hcp);
[h,p,~,stats] = ttest(match_corr_hcp,mismatch_corr_hcp)
save([working_dir 'hcp_group_ind_effects_pFC_eFC.mat'],'corr_pFC_eFC_hcp','group_effect_hcp','ind_effect_hcp','match_corr_hcp','mismatch_corr_hcp')

load('hcp_group_ind_effects_pFC_eFC.mat')
ind_effect_hcp/(group_effect_hcp+ind_effect_hcp)
%% HCP-D
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_PredFC_FC_test.mat')
eFC = permute(HCPD_FC_test,[2,3,1]);
pFC = permute(HCPD_PredFC_test,[2,3,1]);
pFC = double(pFC);

[n_hcpd,~,~] = size(HCPD_FC_test);
intra_mask = eye(n_hcpd,n_hcpd);

for sub_i = 1:n_hcpd
    eFC_vec(sub_i,:) = mat2vec(eFC(:,:,sub_i));
    pFC_vec(sub_i,:) = mat2vec(pFC(:,:,sub_i));
end

% corr_pFC_eFC_hcpd = corr(eFC_vec',pFC_vec' );
corr_pFC_eFC_hcpd = corr(eFC_vec',pFC_vec', "type","Pearson" );

[group_effect_hcpd,ind_effect_hcpd,match_corr_hcpd,mismatch_corr_hcpd] = get_group_ind_effects(corr_pFC_eFC_hcpd);
[h,p,~,stats] = ttest(match_corr_hcpd,mismatch_corr_hcpd)
save([working_dir,'hcpd_group_ind_effects_pFC_eFC.mat'],'corr_pFC_eFC_hcpd','group_effect_hcpd','ind_effect_hcpd','match_corr_hcpd','mismatch_corr_hcpd')

load('hcpd_group_ind_effects_pFC_eFC.mat')
ind_effect_hcpd/(group_effect_hcpd+ind_effect_hcpd)
%%
pFC_eFC_match.data = [match_corr_hcp;mismatch_corr_hcp;match_corr_hcpd;mismatch_corr_hcpd];
pFC_eFC_match.type = [repmat({'Matched'},[n_hcp,1]);repmat({'Mismatched'},[n_hcp,1]);...
    repmat({'Matched'},[n_hcpd,1]);repmat({'Mismatched'},[n_hcpd,1]);];
pFC_eFC_match.dataset = [repmat({'HCP-YA'},[n_hcp*2,1]);repmat({'HCP-D'},[n_hcpd*2,1])];

pFC_eFC_match = struct2table(pFC_eFC_match);
writetable(pFC_eFC_match,[working_dir 'pFC_eFC_match.csv'])
