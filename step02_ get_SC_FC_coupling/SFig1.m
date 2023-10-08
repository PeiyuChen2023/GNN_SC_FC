clear
clc

root_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/';
addpath(genpath(root_dir))

func_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/functions';
addpath(genpath(func_dir))


% working_dir = [root_dir 'step_02_connectional_hierarchy_of_inter_individual_fc_variability/'];
% data_dir = [root_dir 'data/fc_variability/'];

load('7net_label_schaefer400.mat')
% We excluded the limbic network in this study.
net_order = [1 2 3 4 5 6 7]; %1 VIS 2 SMN 3 DAN 4 VAN 5 LIM 6 FPN 7 DMN.
half_flag = 0; % if plot the lower triangle of the matrix, set it to 1.

% load([root_dir 'mean_test_SC_comm_FC_predFC.mat']);
load([root_dir 'result_plot/temp_data/mean_SC_FC_mask.mat']);

hcpa_fc = HCPA_mean_FC;
hcpd_fc = HCPD_mean_FC;
abcd_fc = ABCD_mean_FC;

hcpa_predfc = HCPA_mean_predFC;
hcpd_predfc = HCPD_mean_predFC;
abcd_predfc = ABCD_mean_predFC;


% plot the fc matrix
half_flag = 0;
plot_matrix(hcpd_fc,net_label,net_order,half_flag)
caxis([min(min(hcpd_fc)),max(max(hcpd_fc))]);
print(gcf,'-dpng','-r300',[root_dir 'hcpd_fc.png'])

plot_matrix(hcpa_fc,net_label,net_order,half_flag)
caxis([min(min(hcpa_fc)),max(max(hcpa_fc))]);
print(gcf,'-dpng','-r300',[root_dir 'hcpa_fc.png'])

plot_matrix(abcd_fc,net_label,net_order,half_flag)
caxis([min(min(abcd_fc)),max(max(abcd_fc))]);
print(gcf,'-dpng','-r300',[root_dir 'abcd_fc.png'])
% 
plot_matrix(hcpa_predfc,net_label,net_order,half_flag)
caxis([min(min(hcpa_predfc)),max(max(hcpa_predfc))]);
print(gcf,'-dpng','-r300',[root_dir 'hcpa_predfc.png'])

plot_matrix(hcpa_predfc,net_label,net_order,half_flag)
caxis([min(min(hcpd_predfc)),max(max(hcpd_predfc))]);
print(gcf,'-dpng','-r300',[root_dir 'hcpd_predfc.png'])

plot_matrix(abcd_predfc,net_label,net_order,half_flag)
caxis([min(min(abcd_predfc)),max(max(abcd_predfc))]);
print(gcf,'-dpng','-r300',[root_dir 'abcd_predfc.png'])


% plot the sc matrix
hcpa_sc = log(HCPA_mean_SC.*HCPA_SC_mask);
hcpd_sc = log(HCPD_mean_SC.*HCPD_SC_mask);
abcd_sc = log(ABCD_mean_SC.*ABCD_SC_mask);

% half_flag = 0;
plot_matrix(hcpa_sc,net_label,net_order,half_flag)
caxis([min(min(hcpa_sc(HCPA_SC_mask))),max(max(hcpa_sc(HCPA_SC_mask)))]);
print(gcf,'-dpng','-r300',[root_dir 'hcpa_sc.png'])

plot_matrix(hcpd_sc,net_label,net_order,half_flag)
caxis([min(min(hcpd_sc(HCPD_SC_mask))),max(max(hcpd_sc(HCPD_SC_mask)))]);
print(gcf,'-dpng','-r300',[root_dir 'hcpd_sc.png'])

plot_matrix(abcd_sc,net_label,net_order,half_flag)
caxis([min(min(abcd_sc(ABCD_SC_mask))),max(max(abcd_sc(ABCD_SC_mask)))]);
print(gcf,'-dpng','-r300',[root_dir 'abcd_sc.png'])


close all