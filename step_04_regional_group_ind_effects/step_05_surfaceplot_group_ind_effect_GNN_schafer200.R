## plot surface
library(ggplot2)
library(ggseg)
library(ggsegSchaefer)

rm(list = ls())
source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/plot_surface_2.R')
source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/replace_outliers.R')

working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_04_regional_group_ind_effects/'
pFC_eFC_group_ind_effect <- read.csv(paste0(working_dir, 'pFC_eFC_group_ind_effect_schaefer200.csv'))

#################
pFC_eFC_group_ind_effect$ind_effect_hcp <- replace_outliers(pFC_eFC_group_ind_effect$ind_effect_hcp)$data
pFC_eFC_group_ind_effect$ind_effect_hcpd <- replace_outliers(pFC_eFC_group_ind_effect$ind_effect_hcpd)$data
ind_effect <- c(pFC_eFC_group_ind_effect$ind_effect_hcp[c(1:100)], pFC_eFC_group_ind_effect$ind_effect_hcpd[c(1:100)])

min_val <- min(ind_effect)
max_val <- max(ind_effect)
up <- quantile(ind_effect, probs=0.75, na.rm=T)
low <- quantile(ind_effect, probs=0.25, na.rm=T)
middle <- quantile(ind_effect, probs=0.5, na.rm=T)

pFC_eFC_group_ind_effect$ind_effect_hcp <- replace_outliers(pFC_eFC_group_ind_effect$ind_effect_hcp)$data
data <- pFC_eFC_group_ind_effect[,c(1,3)]
colnames(data) <- c('label','value')
plot_surface_2(data = data, color_type = 2, atlas = schaefer7_200, 
             outpath = paste0(working_dir,'GNN_model/hcp/hcp_pFC_eFC_ind_effect_surface_schaefer200'), min_val = min_val, max_val = max_val, low=low, middle=middle, up=up)

data <- pFC_eFC_group_ind_effect[,c(1,5)]
colnames(data) <- c('label','value')
plot_surface_2(data = data, color_type = 2, atlas = schaefer7_200, 
             outpath = paste0(working_dir,'GNN_model/hcpd/hcpd_pFC_eFC_ind_effect_surface_schaefer200'), min_val = min_val, max_val = max_val, low=low, middle=middle, up=up)


pFC_eFC_group_ind_effect$group_effect_hcpd <- replace_outliers(pFC_eFC_group_ind_effect$group_effect_hcpd)$data
pFC_eFC_group_ind_effect$ind_effect_hcpd <- replace_outliers(pFC_eFC_group_ind_effect$ind_effect_hcpd)$data
group_effect <- c(pFC_eFC_group_ind_effect$group_effect_hcp[c(1:100)], pFC_eFC_group_ind_effect$group_effect_hcpd[c(1:100)])

min_val <- min(group_effect)
max_val <- max(group_effect)
up <- quantile(group_effect, probs=0.75, na.rm=T)
low <- quantile(group_effect, probs=0.25, na.rm=T)
middle <- quantile(group_effect, probs=0.5, na.rm=T)

data <- pFC_eFC_group_ind_effect[,c(1,2)]
colnames(data) <- c('label','value')
plot_surface_2(data = data, color_type = 2, atlas = schaefer7_200,
               outpath = paste0(working_dir,'GNN_model/hcp/hcp_pFC_eFC_group_effect_surface_schaefer200'), min_val = min_val, max_val = max_val, low=low, middle=middle, up=up)

data <- pFC_eFC_group_ind_effect[,c(1,4)]
colnames(data) <- c('label','value')
plot_surface_2(data = data, color_type = 2, atlas = schaefer7_200,
             outpath = paste0(working_dir,'GNN_model/hcpd/hcpd_pFC_eFC_group_effect_surface_schaefer200'), min_val = min_val, max_val = max_val, low=low, middle=middle, up=up)
