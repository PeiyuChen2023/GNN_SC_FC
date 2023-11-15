## plot surface
library(ggplot2)
library(ggseg)
library(ggsegSchaefer)

rm(list = ls())
source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/plot_surface.R')
source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/replace_outliers.R')

working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_04_regional_group_ind_effects/'
pFC_eFC_group_ind_effect <- read.csv(paste0(working_dir, 'pFC_eFC_group_ind_effect.csv'))

#################
pFC_eFC_group_ind_effect$group_effect_hcp <- replace_outliers(pFC_eFC_group_ind_effect$group_effect_hcp)$data
data <- pFC_eFC_group_ind_effect[,c(1,2)]
colnames(data) <- c('label','value')
plot_surface(data = data, color_type = 2, atlas = schaefer7_400,
             outpath = paste0(working_dir,'GNN_model/hcp/hcp_pFC_eFC_group_effect_surface'))

pFC_eFC_group_ind_effect$ind_effect_hcp <- replace_outliers(pFC_eFC_group_ind_effect$ind_effect_hcp)$data
data <- pFC_eFC_group_ind_effect[,c(1,3)]
colnames(data) <- c('label','value')
plot_surface(data = data, color_type = 2, atlas = schaefer7_400, 
             outpath = paste0(working_dir,'GNN_model/hcp/hcp_pFC_eFC_ind_effect_surface'))

pFC_eFC_group_ind_effect$group_effect_hcpd <- replace_outliers(pFC_eFC_group_ind_effect$group_effect_hcpd)$data
data <- pFC_eFC_group_ind_effect[,c(1,4)]
colnames(data) <- c('label','value')
plot_surface(data = data, color_type = 2, atlas = schaefer7_400,
             outpath = paste0(working_dir,'GNN_model/hcpd/hcpd_pFC_eFC_group_effect_surface'))

pFC_eFC_group_ind_effect$ind_effect_hcpd <- replace_outliers(pFC_eFC_group_ind_effect$ind_effect_hcpd)$data
data <- pFC_eFC_group_ind_effect[,c(1,5)]
colnames(data) <- c('label','value')
plot_surface(data = data, color_type = 2, atlas = schaefer7_400, 
             outpath = paste0(working_dir,'GNN_model/hcpd/hcpd_pFC_eFC_ind_effect_surface'))

