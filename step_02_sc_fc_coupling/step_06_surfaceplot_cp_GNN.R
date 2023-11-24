## plot surface
library(ggplot2)
library(ggseg)
library(ggsegSchaefer)

rm(list = ls())

source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/plot_surface.R')
source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/replace_outliers.R')

working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/'
pFC_eFC_cp <- read.csv(paste0(working_dir, 'region_hcp_pFC_eFC_cp.csv'))

pFC_eFC_cp$GNN_cp <- replace_outliers(pFC_eFC_cp$GNN_cp)$data

data <- pFC_eFC_cp[,c(1,2)]
plot_surface(data = data, atlas = schaefer7_400, 
             outpath = paste0(working_dir,'Linear_model/hcp/hcp_pFC_eFC_cp_surface'), color_type = 2)

working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/'
pFC_eFC_cp <- read.csv(paste0(working_dir, 'region_hcpd_pFC_eFC_cp.csv'))

pFC_eFC_cp$GNN_cp <- replace_outliers(pFC_eFC_cp$GNN_cp)$data

data <- pFC_eFC_cp[,c(1,2)]
plot_surface(data = data, atlas = schaefer7_400, 
             outpath = paste0(working_dir,'Linear_model/hcpd/hcpd_pFC_eFC_cp_surface'), color_type = 2)