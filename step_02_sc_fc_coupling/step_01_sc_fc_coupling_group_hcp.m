clear
clc

% addpath(genpath('\Users\chenpeiyu\PycharmProjects\SC_FC_Pred\'))
addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))

working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_SC_FC_coupling/';

load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/group_pFC.mat')
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/info/7net_label_schaefer400.mat')
net_order = [1 2 3 4 5 6 7]; %1 VIS 2 SMN 3 DAN 4 VAN 5 LIM 6 FPN 7 DMN.
half_flag = 0; % if plot the lower triangle of the matrix, set it to 1.

%% SC-eFC
data_hcp = load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCP_SC_FC.mat');
SC_hcp = data_hcp.HCP_SC;
[~,~,n_hcp] = size(SC_hcp);

W_thr = threshold_consistency(SC_hcp, 0.75);
SC_mask_hcp = double(W_thr > 0);
SC_mask_vec_hcp = mat2vec(SC_mask_hcp)'; 

eFC_hcp = data_hcp.HCP_FC;

for sub_i = 1:n_hcp
    SC_hcp(:,:,sub_i) = SC_hcp(:,:,sub_i) .* SC_mask_hcp;
end

% plot the mean matrix
SC_hcp_mean = mean(SC_hcp,3) .* SC_mask_hcp;
% plot_matrix(log(SC_hcp_mean),net_label,net_order,half_flag)
% print(gcf,'-dpng','-r300',[working_dir 'matrix_plot_SC_hcp_mean.png'])
close all

eFC_hcp_mean = mean(eFC_hcp,3);
% plot_matrix(tanh(eFC_hcp_mean),net_label,net_order,half_flag)
caxis([-0.6,0.8]);
% print(gcf,'-dpng','-r300',[working_dir 'matrix_plot_eFC_hcp_mean.png'])
close all

% Individual correlation
for sub_i = 1:n_hcp
    eFC_temp = mat2vec(eFC_hcp(:,:,sub_i));
    eFC_temp = eFC_temp(SC_mask_vec_hcp>0);
    eFC_vec_hcp(sub_i,:) = eFC_temp;

    SC_temp = mat2vec(SC_hcp(:,:,sub_i));
    SC_temp = SC_temp(SC_mask_vec_hcp>0);
    SC_vec_hcp(sub_i,:) = SC_temp;

    corr_SC_eFC_hcp(sub_i,1) = corr(eFC_temp(SC_temp > 0)',log(SC_temp(SC_temp > 0))',"type","Pearson");
end

% Group mean correlation
eFC_vec_hcp_mean = mean(eFC_vec_hcp)';
SC_vec_hcp_mean = mean(SC_vec_hcp)';
hcp_eFC_SC.eFC = eFC_vec_hcp_mean;
hcp_eFC_SC.SC = log(SC_vec_hcp_mean);

[r,p] = corr(hcp_eFC_SC.eFC,hcp_eFC_SC.SC,"type","Pearson")

hcp_eFC_SC = struct2table(hcp_eFC_SC);
writetable(hcp_eFC_SC,[working_dir '/hcp_eFC_SC.csv'])

%% pFC-eFC
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPA_PredFC_FC_test.mat')
eFC = permute(HCPA_FC_test,[2,3,1]);
eFC = double(eFC);
pFC = permute(HCPA_PredFC_test,[2,3,1]);
pFC = double(pFC);

% plot the mean matrix
% pFC_mean = mean(pFC,3);
% plot_matrix(tanh(gp_HCP_pFC),net_label,net_order,half_flag)
caxis([-0.6,0.8]);
% print(gcf,'-dpng','-r300',[working_dir 'matrix_plot_pFC_hcp_mean.png'])
close all

% Individual correlation
[n_pred,~,~] = size(HCPA_FC_test);
intra_mask = eye(n_pred,n_pred);

clear eFC_vec_hcp
for sub_i = 1:n_pred
    eFC_vec_hcp(sub_i,:) = mat2vec(eFC(:,:,sub_i));
    pFC_vec_hcp(sub_i,:) = mat2vec(pFC(:,:,sub_i));
    corr_eFC_pFC_hcp(sub_i,1) = corr(eFC_vec_hcp(sub_i,:)',pFC_vec_hcp(sub_i,:)',"type","Pearson");
end

% Group mean correlation
eFC_vec_hcp_mean = mean(eFC_vec_hcp)';
pFC_vec_hcp_mean = mat2vec(gp_HCP_pFC)';

[r,p] = corr(pFC_vec_hcp_mean,eFC_vec_hcp_mean,"type","Pearson")
hcp_eFC_pFC.eFC = eFC_vec_hcp_mean;
hcp_eFC_pFC.pFC = pFC_vec_hcp_mean;
hcp_eFC_pFC = struct2table(hcp_eFC_pFC);
writetable(hcp_eFC_pFC,[working_dir '/hcp_eFC_pFC.csv'])

%%
hcp_SC_pFC_eFC_coupling.data = [corr_SC_eFC_hcp;corr_eFC_pFC_hcp];
hcp_SC_pFC_eFC_coupling.type = [repmat({'SC-eFC'},[n_hcp,1]);repmat({'pFC-eFC'},[n_pred,1])];
hcp_SC_pFC_eFC_coupling = struct2table(hcp_SC_pFC_eFC_coupling);
writetable(hcp_SC_pFC_eFC_coupling,[working_dir '/hcp_SC_pFC_eFC_coupling.csv'])
