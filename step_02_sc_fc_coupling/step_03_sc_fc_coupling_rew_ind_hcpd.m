clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))
working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_SC_FC_coupling';


%% SC-eFC
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_reSC.mat')
data_hcpd = load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_SC_FC.mat');

SC = data_hcpd.HCPD_SC;
[~,~,n_hcpd] = size(SC);
eFC = data_hcpd.HCPD_FC;
rSC = permute(HCPD_reSC,[2,3,1]);

W_thr = threshold_consistency(SC, 0.75);
SC_mask_hcpd = double(W_thr > 0);
SC_mask_vec_hcpd = mat2vec(SC_mask_hcpd)'; 

eFC = data_hcpd.HCPD_FC;

for sub_i = 1:n_hcpd
    SC(:,:,sub_i) = SC(:,:,sub_i) .* SC_mask_hcpd;
end


for sub_i = 1:n_hcpd
    eFC_temp = mat2vec(eFC(:,:,sub_i));
    eFC_temp = eFC_temp(SC_mask_vec_hcpd>0);
    SC_temp = mat2vec(SC(:,:,sub_i));
    SC_temp = SC_temp(SC_mask_vec_hcpd>0);
    corr_eFC_SC_hcpd(sub_i,1) = corr(eFC_temp(SC_temp > 0)',log(SC_temp(SC_temp > 0))',"type","Pearson");

    eFC_temp = mat2vec(eFC(:,:,sub_i));
    rSC_temp = mat2vec(rSC(:,:,sub_i));
    corr_eFC_rSC_hcpd(sub_i,1) = corr(eFC_temp(rSC_temp > 0)',log(rSC_temp(rSC_temp > 0))',"type","Pearson");
end



%% pFC-eFC
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_PredFC_FC_rePredFC_test.mat')
eFC = permute(HCPD_FC_test,[2,3,1]);
pFC = permute(HCPD_PredFC_test,[2,3,1]);
pFC = double(pFC);
rpFC = permute(HCPD_rePredFC_test,[2,3,1]);
rpFC = double(rpFC);

% Individual correlation
[n_pred,~,~] = size(HCPD_FC_test);
intra_mask = eye(n_pred,n_pred);

clear eFC_vec_hcpd
for sub_i = 1:n_pred
    eFC_vec_hcpd(sub_i,:) = mat2vec(eFC(:,:,sub_i));
    pFC_vec_hcpd(sub_i,:) = mat2vec(pFC(:,:,sub_i));
    rpFC_vec_hcpd(sub_i,:) = mat2vec(rpFC(:,:,sub_i));

    corr_eFC_pFC_hcpd(sub_i,1) = corr(eFC_vec_hcpd(sub_i,:)',pFC_vec_hcpd(sub_i,:)',"type","Pearson");
    corr_eFC_rpFC_hcpd(sub_i,1) = corr(eFC_vec_hcpd(sub_i,:)',rpFC_vec_hcpd(sub_i,:)',"type","Pearson");
end


%%
hcpd_SC_rSC_pFC_rpFC_eFC_coupling.data = [corr_eFC_SC_hcpd;corr_eFC_rSC_hcpd;corr_eFC_pFC_hcpd;corr_eFC_rpFC_hcpd];
hcpd_SC_rSC_pFC_rpFC_eFC_coupling.type = [repmat({'Liner'},[n_hcpd,1]);repmat({'Liner(rew)'},[n_hcpd,1]);repmat({'GNN'},[n_pred,1]);repmat({'GNN(rew)'},[n_pred,1]);];

hcpd_SC_rSC_pFC_rpFC_eFC_coupling = struct2table(hcpd_SC_rSC_pFC_rpFC_eFC_coupling);
writetable(hcpd_SC_rSC_pFC_rpFC_eFC_coupling,[working_dir '/hcpd_SC_pFC_reSC_repFC_eFC_coupling.csv'])
