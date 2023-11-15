## plot surface
library(ggplot2)
library(ggseg)
library(ggsegSchaefer)

rm(list = ls())

source('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/plot_surface.R')

working_dir <- 'F:/Cui_Lab/Projects/GNN_SC_FC/matlab/step_05_regional_group_ind_effects/'
data <- read.csv(paste0(working_dir, 'hcp_pFC_eFC_group_ind_effect.csv'))

shaeferindex<-read.csv('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/data/schaefer400_sa_rank.csv')
shaeferindex <- shaeferindex[order(shaeferindex$index),]
shaeferindex$label<-shaeferindex$label_7network
shaeferindex$label[1:200]<-paste0("lh_", shaeferindex$label[1:200])
shaeferindex$label[201:400]<-paste0("rh_", shaeferindex$label[201:400])

data <- shaeferindex[,c(3,9)]
colnames(data) <- c('label','value')
plot_surface(data = data, 
             outpath = paste0(working_dir,'sa_rank'), color_type = 2)

# p <- ggplot(data=shaeferindex)+
#   geom_brain(aes(fill=finalrank.wholebrain,colour=finalrank.wholebrain), atlas=schaefer7_400)+
#   scale_fill_distiller(type="seq", palette = "RdBu", direction = -1)+
#   scale_color_distiller(type="seq", palette = "RdBu", direction = -1)+
#   theme_void()+ theme(legend.position="none")
# p
# 
# ggsave(paste0(working_dir,'sa_rank.png'),plot=p,bg='transparent',width = 10,height = 4,units = "cm",dpi = 600)