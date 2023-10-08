clear
clc

root_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/';
addpath(genpath(root_dir))

func_dir = [root_dir 'functions'];
addpath(genpath(func_dir))


load([root_dir 'data/info/7net_label_schaefer400.mat']);
load([root_dir 'data/info/SC_mask_75.mat']);

net_order = [1 2 3 4 5 6 7]; %1 VIS 2 SMN 3 DAN 4 VAN 5 LIM 6 FPN 7 DMN.
half_flag = 0;

% load([root_dir 'mean_test_SC_comm_FC_predFC.mat']);
load([root_dir 'data/SC_FC_PredFC_matrix/HCPA_mean_SC_FC_predFC.mat']);
load([root_dir 'data/SC_FC_PredFC_matrix/HCPD_mean_SC_FC_predFC.mat']);
load([root_dir 'data/SC_FC_PredFC_matrix/ABCD_mean_SC_FC_predFC.mat']);

hcpa_fc = HCPA_mean_FC;
hcpd_fc = HCPD_mean_FC;
abcd_fc = ABCD_mean_FC;

hcpa_predfc = HCPA_mean_predFC;
hcpd_predfc = HCPD_mean_predFC;
abcd_predfc = ABCD_mean_predFC;


% plot the fc matrix

plot_matrix(hcpa_fc,net_label,net_order,half_flag)
caxis([min(min(hcpa_fc)),max(max(hcpa_fc))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/hcpa_fc.png'])

plot_matrix(hcpd_fc,net_label,net_order,half_flag)
caxis([min(min(hcpd_fc)),max(max(hcpd_fc))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/hcpd_fc.png'])


plot_matrix(abcd_fc,net_label,net_order,half_flag)
caxis([min(min(abcd_fc)),max(max(abcd_fc))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/abcd_fc.png'])

% plot the predfc matrix
plot_matrix(hcpa_predfc,net_label,net_order,half_flag)
caxis([min(min(hcpa_predfc)),max(max(hcpa_predfc))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/hcpa_predfc.png'])

plot_matrix(hcpd_predfc,net_label,net_order,half_flag)
caxis([min(min(hcpd_predfc)),max(max(hcpd_predfc))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/hcpd_predfc.png'])

plot_matrix(abcd_predfc,net_label,net_order,half_flag)
caxis([min(min(abcd_predfc)),max(max(abcd_predfc))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/abcd_predfc.png'])


% plot the sc matrix
HCPA_sc_mask = logical(HCPA_sc_mask);
HCPD_sc_mask = logical(HCPD_sc_mask);
ABCD_sc_mask = logical(ABCD_sc_mask);


hcpa_sc = log(HCPA_mean_SC.*HCPA_sc_mask);
hcpd_sc = log(HCPD_mean_SC.*HCPD_sc_mask);
abcd_sc = log(ABCD_mean_SC.*ABCD_sc_mask);

plot_matrix(hcpa_sc,net_label,net_order,half_flag)
caxis([min(min(hcpa_sc(HCPA_sc_mask))),max(max(hcpa_sc(HCPA_sc_mask)))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/hcpa_sc.png'])

plot_matrix(hcpd_sc,net_label,net_order,half_flag)
caxis([min(min(hcpd_sc(HCPD_sc_mask))),max(max(hcpd_sc(HCPD_sc_mask)))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/hcpd_sc.png'])

plot_matrix(abcd_sc,net_label,net_order,half_flag)
caxis([min(min(abcd_sc(ABCD_sc_mask))),max(max(abcd_sc(ABCD_sc_mask)))]);
print(gcf,'-dpng','-r300',[root_dir 'result_plot/SFig1/abcd_sc.png'])


close all