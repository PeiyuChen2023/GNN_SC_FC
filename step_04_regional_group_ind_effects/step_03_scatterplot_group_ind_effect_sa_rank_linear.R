library('ggplot2')

rm(list = ls())
working_dir <- '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_04_regional_group_ind_effects/'
source('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/GNN_SC_FC/functions/replace_outliers.R')

####################
data <- read.csv(paste0(working_dir, 'hcp_SC_eFC_group_ind_effect.csv'))
data$group_effect <- data$group_effect
data$ind_effect <- data$ind_effect

outlier <- !replace_outliers(data$group_effect)$outlier
rho <- cor.test(data[outlier,1], data[outlier,3], method = 'spearman',exact=FALSE)$estimate

p <- ggplot(data = data[outlier,], mapping = aes(x = sa_rank, y = group_effect)) +
  geom_point(size=1.25, alpha=1, shape=19, aes(color=sa_rank)) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 12, color = 'black'),axis.title = element_text(size = 12),aspect.ratio = 0.7) + 
  labs(y = "Group-common \n structure-function coupling (r)", x = "Sensorimotor-association axis rank")+ theme(legend.position="none") + 
  #theme(axis.title.y = element_text(vjust = -1)) +
  scale_color_gradient2(low = "#4393C3", high = "#D6604D", mid = "#F6FBFF",
                        midpoint = 200, limit = c(0,400)) +
  scale_y_continuous(limits = c(0, 0.6),breaks = seq(0, 0.6, by = 0.1)) +
  scale_x_continuous(limits = c(0, 400),breaks = seq(0, 400, by = 100))

p

ggsave(paste0(working_dir,'Linear_model/corr_hcp_SC_eFC_group_effect_sa_rank.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)

#
outlier <- !replace_outliers(data$ind_effect)$outlier
rho <- cor.test(data[outlier,2], data[outlier,3], method = 'spearman',exact=FALSE)$estimate

p <- ggplot(data = data[outlier,], mapping = aes(x = sa_rank, y = ind_effect)) +
  geom_point(size=1.25, alpha=1, shape=19, aes(color=sa_rank)) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 12, color = 'black'),axis.title = element_text(size = 12),aspect.ratio = 0.7) + 
  labs(y = "Individual-specific \n structure-function coupling (r)", x = "Sensorimotor-association axis rank")+ theme(legend.position="none") + 
  #theme(axis.title.y = element_text(vjust = -1)) +
  scale_color_gradient2(low = "#4393C3", high = "#D6604D", mid = "#F6FBFF",
                        midpoint = 200, limit = c(0,400)) + 
  #scale_y_continuous(limits = c(-3, 3),breaks = seq(-3, 3, by = 1)) +
  scale_x_continuous(limits = c(0, 400),breaks = seq(0, 400, by = 100))

p

ggsave(paste0(working_dir,'Linear_model/corr_hcp_SC_eFC_ind_effect_sa_rank.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)

####################
data <- read.csv(paste0(working_dir, 'hcpd_SC_eFC_group_ind_effect.csv'))
data$group_effect <- data$group_effect
data$ind_effect <- data$ind_effect

outlier <- !replace_outliers(data$group_effect)$outlier
rho <- cor.test(data[outlier,1], data[outlier,3], method = 'spearman',exact=FALSE)$estimate

p <- ggplot(data = data[outlier,], mapping = aes(x = sa_rank, y = group_effect)) +
  geom_point(size=1.25, alpha=1, shape=19, aes(color=sa_rank)) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 12, color = 'black'),axis.title = element_text(size = 12),aspect.ratio = 0.7) + 
  labs(y = "Group-common \n structure-function coupling (r)", x = "Sensorimotor-association axis rank")+ theme(legend.position="none") + 
  #theme(axis.title.y = element_text(vjust = -1)) +
  scale_color_gradient2(low = "#4393C3", high = "#D6604D", mid = "#F6FBFF",
                        midpoint = 200, limit = c(0,400)) + 
  scale_y_continuous(limits = c(0, 0.6),breaks = seq(0, 0.6, by = 0.1)) +
  scale_x_continuous(limits = c(0, 400),breaks = seq(0, 400, by = 100))

p

ggsave(paste0(working_dir,'Linear_model/corr_hcpd_SC_eFC_group_effect_sa_rank.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)

#
outlier <- !replace_outliers(data$ind_effect)$outlier
rho <- cor.test(data[outlier,2], data[outlier,3], method = 'spearman',exact=FALSE)$estimate

p <- ggplot(data = data[outlier,], mapping = aes(x = sa_rank, y = ind_effect)) +
  geom_point(size=1.25, alpha=1, shape=19, aes(color=sa_rank)) + theme_classic() + 
  geom_smooth(method = "lm", colour = "black",linewidth = 0.7, fullrange = TRUE) +
  theme(axis.text = element_text(size = 12, color = 'black'),axis.title = element_text(size = 12),aspect.ratio = 0.7) + 
  labs(y = "Individual-specific \n structure-function coupling (r)", x = "Sensorimotor-association axis rank")+ theme(legend.position="none") + 
  #theme(axis.title.y = element_text(vjust = -1)) +
  scale_color_gradient2(low = "#4393C3", high = "#D6604D", mid = "#F6FBFF",
                        midpoint = 200, limit = c(0,400)) + 
  # scale_y_continuous(limits = c(-3, 3),breaks = seq(-3, 3, by = 1)) +
  scale_x_continuous(limits = c(0, 400),breaks = seq(0, 400, by = 100))

p

ggsave(paste0(working_dir,'Linear_model/corr_hcpd_SC_eFC_ind_effect_sa_rank.png'),plot=p,width = 10,height = 10,units = "cm",dpi = 600)
