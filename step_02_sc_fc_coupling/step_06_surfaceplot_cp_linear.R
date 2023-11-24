## plot surface
library(ggplot2)
library(ggseg)
library(ggsegSchaefer)

rm(list = ls())

source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/plot_surface.R')
source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/replace_outliers.R')

working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/'
SC_eFC_cp <- read.csv(paste0(working_dir, 'region_hcp_SC_eFC_cp.csv'))

SC_eFC_cp$lin_cp <- replace_outliers(SC_eFC_cp$lin_cp)$data

data <- SC_eFC_cp[,c(1,2)]
plot_surface(data = data, atlas = schaefer7_400, 
             outpath = paste0(working_dir,'Linear_model/hcp/hcp_SC_eFC_cp_surface'), color_type = 2)

working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/'
SC_eFC_cp <- read.csv(paste0(working_dir, 'region_hcpd_SC_eFC_cp.csv'))

SC_eFC_cp$lin_cp <- replace_outliers(SC_eFC_cp$lin_cp)$data

data <- SC_eFC_cp[,c(1,2)]
plot_surface(data = data, atlas = schaefer7_400, 
             outpath = paste0(working_dir,'Linear_model/hcpd/hcpd_SC_eFC_cp_surface'), color_type = 2)