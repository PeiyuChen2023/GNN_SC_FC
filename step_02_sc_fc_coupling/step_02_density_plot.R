library(ggplot2)
library(dplyr)
library(viridis)
library(ggpointdensity)

rm(list = ls())
working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/'
data <- read.csv(paste0(working_dir, 'hcp_eFC_SC.csv'))

p <- ggplot(data = data, mapping = aes(x = SC, y = eFC)) +
  geom_pointdensity(size=0.2, alpha = 0.8, shape=16) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 16, color = 'black'),axis.title = element_text(size = 16),aspect.ratio = 1) + 
  labs(y = "Mean FC", x = "Log(Mean SC)")+ theme(legend.position="none") + 
  theme(axis.title.y = element_text(vjust = -1)) +
  scale_colour_gradientn(colours = rev(c("#9ECAE1", "#6BAED6", "#4292C6")))+
  scale_y_continuous(limits = c(-0.8, 1.3),breaks = seq(-0.5, 1, by = 0.5))+
  scale_x_continuous(limits = c(-11, 1),breaks = seq(-10, 0, by = 2))

p

ggsave(paste0(working_dir,'hcp_eFC_SC.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)

####################
data <- read.csv(paste0(working_dir, 'hcp_eFC_pFC.csv'))

p <- ggplot(data = data, mapping = aes(x = pFC, y = eFC)) +
  geom_pointdensity(size=0.2, alpha = 0.8, shape=16) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 16, color = 'black'),axis.title = element_text(size = 16),aspect.ratio = 1) + 
  labs(y = "Mean FC", x = "Predicted mean FC")+ theme(legend.position="none") + 
  theme(axis.title.y = element_text(vjust = -1)) +
  scale_colour_gradientn(colours = rev(c("#9ECAE1", "#6BAED6", "#4292C6"))) +
  scale_y_continuous(limits = c(-0.8, 1.3),breaks = seq(-0.5, 1, by = 0.5)) + 
  scale_x_continuous(limits = c(-0.8, 1.3),breaks = seq(-0.5, 1, by = 0.5))

p

ggsave(paste0(working_dir,'hcp_eFC_pFC.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)

working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_02_sc_fc_coupling/'
data <- read.csv(paste0(working_dir, 'hcpd_eFC_SC.csv'))

p <- ggplot(data = data, mapping = aes(x = SC, y = eFC)) +
  geom_pointdensity(size=0.2, alpha = 0.8, shape=16) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 16, color = 'black'),axis.title = element_text(size = 16),aspect.ratio = 1) + 
  labs(y = "Mean FC", x = "Log(Mean SC)")+ theme(legend.position="none") + 
  theme(axis.title.y = element_text(vjust = -1)) +
  scale_colour_gradientn(colours = rev(c("#9ECAE1", "#6BAED6", "#4292C6")))+
  scale_y_continuous(limits = c(-0.8, 1.3),breaks = seq(-0.5, 1, by = 0.5)) + 
  scale_x_continuous(limits = c(-9, 3),breaks = seq(-8, 2, by = 2))

p

ggsave(paste0(working_dir,'hcpd_eFC_SC.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)

####################
data <- read.csv(paste0(working_dir, 'hcpd_eFC_pFC.csv'))

p <- ggplot(data = data, mapping = aes(x = pFC, y = eFC)) +
  geom_pointdensity(size=0.2, alpha = 0.8, shape=16) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 16, color = 'black'),axis.title = element_text(size = 16),aspect.ratio = 1) + 
  labs(y = "Mean FC", x = "Predicted mean FC")+ theme(legend.position="none") + 
  theme(axis.title.y = element_text(vjust = -1)) +
  scale_colour_gradientn(colours = rev(c("#9ECAE1", "#6BAED6", "#4292C6"))) +
  scale_y_continuous(limits = c(-0.8, 1.3),breaks = seq(-0.5, 1, by = 0.5)) + 
  scale_x_continuous(limits = c(-0.8, 1.3),breaks = seq(-0.5, 1, by = 0.5))

p

ggsave(paste0(working_dir,'hcpd_eFC_pFC.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)

#"#F7FBFF" "#DEEBF7" "#C6DBEF" "#9ECAE1" "#6BAED6" "#4292C6" "#2171B5" "#08519C" "#08306B"