library(ggplot2)
library(see)
library(RColorBrewer)

rm(list = ls())
working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/'

##### HCP-YA #####
data <- read.csv(paste0(working_dir, 'hcp_SC_pFC_reSC_repFC_eFC_coupling.csv'))

dp <- ggplot(data, aes(x=type, y=data, fill=type)) + 
  geom_violinhalf(width=0.6, trim=FALSE,flip = TRUE,position = position_nudge(x = -0.16), scale='width') +
  geom_boxplot(width=0.14, fill="white", outlier.shape = NA,position = position_nudge(x = -0.16)) +
  geom_point(aes(x=type, y=data, colour=type),shape = 13,size = 0.2, alpha = 0.8, position=position_jitter(width=.07))+
  scale_color_manual(values=c("#C6DBEF", "#FEE1D8", "#9ECAE1", "#FCBBA1"))+
  theme_classic() +
  theme(axis.text = element_text(size = 16, color = 'black'),axis.title = element_text(size = 16),aspect.ratio = 0.8)+
  scale_fill_manual(values=c("#EFF3FF", "#FEE1D8", "#9ECAE1", "#FCBBA1"))+ scale_x_discrete(expand = c(0.25, 0), limits=c("Linear","GNN","Linear(rew)", "GNN(rew)"),
                                                                                            label=c("Linear","GNN","Linear\n(Rew)", "GNN\n(Rew)") )+
  scale_y_continuous(limits = c(0., 0.85),breaks = seq(0., 0.8, by = 0.2)) + 
  labs(y = "Individual coupling (r)", x = "") + theme(legend.position="none")
dp

ggsave(paste0(working_dir,'hcp_SC_pFC_reSC_repFC_eFC_coupling.png'),plot=dp,width = 12,height = 10,units = "cm",dpi = 600)

##### HCP-D #####
data <- read.csv(paste0(working_dir, 'hcpd_SC_pFC_reSC_repFC_eFC_coupling.csv'))

dp <- ggplot(data, aes(x=type, y=data, fill=type)) + 
  geom_violinhalf(width=0.6, trim=FALSE,flip = TRUE,position = position_nudge(x = -0.16), scale='width') +
  geom_boxplot(width=0.14, fill="white", outlier.shape = NA,position = position_nudge(x = -0.16)) +
  geom_point(aes(x=type, y=data, colour=type),shape = 13,size = 0.2, alpha = 0.8, position=position_jitter(width=.07))+
  scale_color_manual(values=c("#C6DBEF", "#FEE1D8", "#9ECAE1", "#FCBBA1"))+
  theme_classic() + 
  theme(axis.text = element_text(size = 16, color = 'black'),axis.title = element_text(size = 16),aspect.ratio = 0.8)+
  scale_fill_manual(values=c("#EFF3FF", "#FEE1D8", "#9ECAE1", "#FCBBA1"))+ scale_x_discrete(expand = c(0.25, 0), limits=c("Linear","GNN","Linear(rew)", "GNN(rew)"), label=c("Linear","GNN","Linear\n(Rew)", "GNN\n(Rew)")) +
  scale_y_continuous(limits = c(0., 0.85),breaks = seq(0., 0.8, by = 0.2)) + 
  labs(y = "Individual coupling (r)", x = "") + theme(legend.position="none")
dp

ggsave(paste0(working_dir,'hcpd_SC_pFC_reSC_repFC_eFC_coupling.png'),plot=dp,width = 12,height = 10,units = "cm",dpi = 600)