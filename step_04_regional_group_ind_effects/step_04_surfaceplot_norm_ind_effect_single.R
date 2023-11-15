## plot surface
library(ggplot2)
library(ggseg)
library(ggsegSchaefer)

rm(list = ls())
source('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/plot_surface.R')
source('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/replace_outliers.R')

working_dir <- 'F:/Cui_Lab/Projects/GNN_SC_FC/matlab/step_05_regional_group_ind_effects/'
pFC_eFC_group_ind_effect_norm <- read.csv(paste0(working_dir, 'pFC_eFC_group_ind_effect_norm.csv'))

#################
# pFC_eFC_group_ind_effect_norm$group_effect_hcp <- replace_outliers(pFC_eFC_group_ind_effect_norm$group_effect_hcp)$data
# data <- pFC_eFC_group_ind_effect_norm[,c(1,2)]
# colnames(data) <- c('label','value')
# plot_surface(data = data, 
#              outpath = paste0(working_dir,'GNN_model/hcp/hcp_pFC_eFC_group_effect_surface'), color_type = 2)

pFC_eFC_group_ind_effect_norm$ind_effect_norm_hcp <- replace_outliers(pFC_eFC_group_ind_effect_norm$ind_effect_norm_hcp)$data
data <- pFC_eFC_group_ind_effect_norm[,c(1,3)]
colnames(data) <- c('label','value')
plot_surface(data = data, 
             outpath = paste0(working_dir,'GNN_model/hcp/hcp_pFC_eFC_ind_effect_norm_surface'), color_type = 2)

# pFC_eFC_group_ind_effect_norm$group_effect_hcpd <- replace_outliers(pFC_eFC_group_ind_effect_norm$group_effect_hcpd)$data
# data <- pFC_eFC_group_ind_effect_norm[,c(1,4)]
# colnames(data) <- c('label','value')
# plot_surface(data = data, 
#              outpath = paste0(working_dir,'GNN_model/hcpd/hcpd_pFC_eFC_group_effect_surface'), color_type = 2)

pFC_eFC_group_ind_effect_norm$ind_effect_norm_hcpd <- replace_outliers(pFC_eFC_group_ind_effect_norm$ind_effect_norm_hcpd)$data
data <- pFC_eFC_group_ind_effect_norm[,c(1,5)]
colnames(data) <- c('label','value')
plot_surface(data = data, 
             outpath = paste0(working_dir,'GNN_model/hcpd/hcpd_pFC_eFC_ind_effect_norm_surface'), color_type = 2)

