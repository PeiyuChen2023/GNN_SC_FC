clear
clc

addpath(genpath('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/'))

working_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_SC_FC_coupling/';

load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/info/7net_label_schaefer400.mat')

net_order = [1 2 3 4 5 6 7]; %1 VIS 2 SMN 3 DAN 4 VAN 5 LIM 6 FPN 7 DMN.
half_flag = 0; % if plot the lower triangle of the matrix, set it to 1.

%% SC-eFC
data_hcpd = load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_SC_FC.mat');
SC_hcpd = data_hcpd.HCPD_SC;
[~,~,n_hcpd] = size(SC_hcpd);

W_thr = threshold_consistency(SC_hcpd, 0.75);
SC_mask_hcpd = double(W_thr > 0);
SC_mask_vec_hcpd = mat2vec(SC_mask_hcpd)'; 

eFC_hcpd = data_hcpd.HCPD_FC;

for sub_i = 1:n_hcpd
    SC_hcpd(:,:,sub_i) = SC_hcpd(:,:,sub_i) .* SC_mask_hcpd;
end

% plot the mean matrix
SC_hcpd_mean = mean(SC_hcpd,3) .* SC_mask_hcpd;
plot_matrix(log(SC_hcpd_mean),net_label,net_order,half_flag)
print(gcf,'-dpng','-r300',[working_dir 'matrix_plot_SC_hcpd_mean.png'])
close all

eFC_hcpd_mean = mean(eFC_hcpd,3);
plot_matrix(tanh(eFC_hcpd_mean),net_label,net_order,half_flag)
caxis([-0.6,0.8]);
print(gcf,'-dpng','-r300',[working_dir 'matrix_plot_eFC_hcpd_mean.png'])
close all

% Individual correlation
for sub_i = 1:n_hcpd
    eFC_temp = mat2vec(eFC_hcpd(:,:,sub_i));
    eFC_temp = eFC_temp(SC_mask_vec_hcpd>0);
    eFC_vec_hcpd(sub_i,:) = eFC_temp;

    SC_temp = mat2vec(SC_hcpd(:,:,sub_i));
    SC_temp = SC_temp(SC_mask_vec_hcpd>0);
    SC_vec_hcpd(sub_i,:) = SC_temp;

    corr_SC_eFC_hcpd(sub_i,1) = corr(eFC_temp(SC_temp > 0)',log(SC_temp(SC_temp > 0))',"type","Pearson");
end

% Group mean correlation
eFC_vec_hcpd_mean = mean(eFC_vec_hcpd)';
SC_vec_hcpd_mean = mean(SC_vec_hcpd)';
hcpd_eFC_SC.eFC = eFC_vec_hcpd_mean;
hcpd_eFC_SC.SC = log(SC_vec_hcpd_mean);

[r,p] = corr(hcpd_eFC_SC.eFC,hcpd_eFC_SC.SC,"type","Pearson")

hcpd_eFC_SC = struct2table(hcpd_eFC_SC);
writetable(hcpd_eFC_SC,[working_dir '/hcpd_eFC_SC.csv'])

%% pFC-eFC
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/group_pFC.mat')
load('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/data/HCPD_PredFC_FC_test.mat')
eFC = permute(HCPD_FC_test,[2,3,1]);
pFC = permute(HCPD_PredFC_test,[2,3,1]);
pFC = double(pFC);

% plot the mean matrix
% pFC_mean = mean(pFC,3);
plot_matrix(tanh(gp_HCPD_pFC),net_label,net_order,half_flag)
caxis([-0.6,0.8]);
print(gcf,'-dpng','-r300',[working_dir 'matrix_plot_pFC_hcpd_mean.png'])
close all

% Individual correlation
[n_pred,~,~] = size(HCPD_FC_test);
intra_mask = eye(n_pred,n_pred);

clear eFC_vec_hcpd
for sub_i = 1:n_pred
    eFC_vec_hcpd(sub_i,:) = mat2vec(eFC(:,:,sub_i));
    pFC_vec_hcpd(sub_i,:) = mat2vec(pFC(:,:,sub_i));

    corr_eFC_pFC_hcpd(sub_i,1) = corr(eFC_vec_hcpd(sub_i,:)',pFC_vec_hcpd(sub_i,:)',"type","Pearson");
end

% Group mean correlation
eFC_vec_hcpd_mean = mean(eFC_vec_hcpd)';
pFC_vec_hcpd_mean = mat2vec(gp_HCPD_pFC)';

[r,p] = corr(pFC_vec_hcpd_mean,eFC_vec_hcpd_mean,"type","Pearson")
hcpd_eFC_pFC.eFC = eFC_vec_hcpd_mean;
hcpd_eFC_pFC.pFC = pFC_vec_hcpd_mean;
hcpd_eFC_pFC = struct2table(hcpd_eFC_pFC);
writetable(hcpd_eFC_pFC,[working_dir '/hcpd_eFC_pFC.csv'])

%%
hcpd_SC_pFC_eFC_coupling.data = [corr_SC_eFC_hcpd;corr_eFC_pFC_hcpd];
hcpd_SC_pFC_eFC_coupling.type = [repmat({'SC-eFC'},[n_hcpd,1]);repmat({'pFC-eFC'},[n_pred,1])];
hcpd_SC_pFC_eFC_coupling = struct2table(hcpd_SC_pFC_eFC_coupling);
writetable(hcpd_SC_pFC_eFC_coupling,[working_dir '/hcpd_SC_pFC_eFC_coupling.csv'])
