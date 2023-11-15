clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_SC_FC_coupling';


%% SC-eFC
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPA_reSC.mat')
data_hcp = load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCP_SC_FC.mat');

SC = data_hcp.HCP_SC;
[~,~,n_hcp] = size(SC);
eFC = data_hcp.HCP_FC;
rSC = permute(HCPA_reSC,[2,3,1]);

W_thr = threshold_consistency(SC, 0.75);
SC_mask_hcp = double(W_thr > 0);
SC_mask_vec_hcp = mat2vec(SC_mask_hcp)'; 

eFC = data_hcp.HCP_FC;

for sub_i = 1:n_hcp
    SC(:,:,sub_i) = SC(:,:,sub_i) .* SC_mask_hcp;
end


for sub_i = 1:n_hcp
    eFC_temp = mat2vec(eFC(:,:,sub_i));
    eFC_temp = eFC_temp(SC_mask_vec_hcp>0);
    SC_temp = mat2vec(SC(:,:,sub_i));
    SC_temp = SC_temp(SC_mask_vec_hcp>0);
    corr_eFC_SC_hcp(sub_i,1) = corr(eFC_temp(SC_temp > 0)',log(SC_temp(SC_temp > 0))',"type","Pearson");

    eFC_temp = mat2vec(eFC(:,:,sub_i));
    rSC_temp = mat2vec(rSC(:,:,sub_i));
    corr_eFC_rSC_hcp(sub_i,1) = corr(eFC_temp(rSC_temp > 0)',log(rSC_temp(rSC_temp > 0))',"type","Pearson");
end



%% pFC-eFC
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPA_PredFC_FC_rePredFC_test.mat')
eFC = permute(HCPA_FC_test,[2,3,1]);
pFC = permute(HCPA_PredFC_test,[2,3,1]);
pFC = double(pFC);
rpFC = permute(HCPA_rePredFC_test,[2,3,1]);
rpFC = double(rpFC);

% Individual correlation
[n_pred,~,~] = size(HCPA_FC_test);
intra_mask = eye(n_pred,n_pred);

for sub_i = 1:n_pred
    eFC_vec_hcp(sub_i,:) = mat2vec(eFC(:,:,sub_i));
    pFC_vec_hcp(sub_i,:) = mat2vec(pFC(:,:,sub_i));
    rpFC_vec_hcp(sub_i,:) = mat2vec(rpFC(:,:,sub_i));

    corr_eFC_pFC_hcp(sub_i,1) = corr(eFC_vec_hcp(sub_i,:)',pFC_vec_hcp(sub_i,:)',"type","Pearson");
    corr_eFC_rpFC_hcp(sub_i,1) = corr(eFC_vec_hcp(sub_i,:)',rpFC_vec_hcp(sub_i,:)',"type","Pearson");
end


%%
hcp_SC_rSC_pFC_rpFC_eFC_coupling.data = [corr_eFC_SC_hcp;corr_eFC_rSC_hcp;corr_eFC_pFC_hcp;corr_eFC_rpFC_hcp];
hcp_SC_rSC_pFC_rpFC_eFC_coupling.type = [repmat({'Liner'},[n_hcp,1]);repmat({'Liner(rew)'},[n_hcp,1]);repmat({'GNN'},[n_pred,1]);repmat({'GNN(rew)'},[n_pred,1]);];

hcp_SC_rSC_pFC_rpFC_eFC_coupling = struct2table(hcp_SC_rSC_pFC_rpFC_eFC_coupling);
writetable(hcp_SC_rSC_pFC_rpFC_eFC_coupling,[working_dir '/hcp_SC_pFC_reSC_repFC_eFC_coupling.csv'])
