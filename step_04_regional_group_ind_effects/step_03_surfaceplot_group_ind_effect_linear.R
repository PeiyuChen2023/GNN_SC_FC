## plot surface
library(ggplot2)
library(ggseg)
library(ggsegSchaefer)

rm(list = ls())
source('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/plot_surface.R')
source('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/replace_outliers.R')

working_dir <- 'F:/Cui_Lab/Projects/GNN_SC_FC/matlab/step_05_regional_group_ind_effects/'
SC_eFC_group_ind_effect <- read.csv(paste0(working_dir, 'SC_eFC_group_ind_effect.csv'))

#################
#data$ind_effect_hcpd <- rank(replace_outliers(data$ind_effect_hcpd))
SC_eFC_group_ind_effect$group_effect_hcp <- replace_outliers(SC_eFC_group_ind_effect$group_effect_hcp)$data
data <- SC_eFC_group_ind_effect[,c(1,2)]
colnames(data) <- c('label','value')
plot_surface(data = data, 
             outpath = paste0(working_dir,'Linear_model/hcp/hcp_SC_eFC_group_effect_surface'), color_type = 2)

SC_eFC_group_ind_effect$ind_effect_hcp <- replace_outliers(SC_eFC_group_ind_effect$ind_effect_hcp)$data
data <- SC_eFC_group_ind_effect[,c(1,3)]
colnames(data) <- c('label','value')
plot_surface(data = data, 
             outpath = paste0(working_dir,'Linear_model/hcp/hcp_SC_eFC_ind_effect_surface'), color_type = 2)

SC_eFC_group_ind_effect$group_effect_hcpd <- replace_outliers(SC_eFC_group_ind_effect$group_effect_hcpd)$data
data <- SC_eFC_group_ind_effect[,c(1,4)]
colnames(data) <- c('label','value')
plot_surface(data = data, 
             outpath = paste0(working_dir,'Linear_model/hcpd/hcpd_SC_eFC_group_effect_surface'), color_type = 2)

SC_eFC_group_ind_effect$ind_effect_hcpd <- replace_outliers(SC_eFC_group_ind_effect$ind_effect_hcpd)$data
data <- SC_eFC_group_ind_effect[,c(1,5)]
colnames(data) <- c('label','value')
plot_surface(data = data, 
             outpath = paste0(working_dir,'Linear_model/hcpd/hcpd_SC_eFC_ind_effect_surface'), color_type = 2)

