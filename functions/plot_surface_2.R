plot_surface_2 <- function(data,outpath,color_type,atlas, min_val, low, middle, up, max_val) {
  
  if (color_type == 1) {
    values<-scales::rescale(c(1:100), to=c(min(data$value),median(data$value)))
    values<-c(values, scales::rescale(c(1:100), to=c(median(data$value),max(data$value))))
    values<-scales::rescale(values, to=c(0,1))
  }
  
  if (color_type == 2) {
    maxup <- max_val 
    minup <- min_val
    
    limit.up <- quantile(data$value, probs=0.75, na.rm=T)
    limit.low <- quantile(data$value, probs=0.25, na.rm=T)
    limit.middle <- quantile(data$value, probs=0.5, na.rm=T)
    #limit.middle <- mean(c(limit.up, limit.low))

    values1 <- scales::rescale(c(1:100), to=c(limit.up,maxup))
    values2 <- scales::rescale(c(1:100), to=c(limit.middle,limit.up))
    values3 <- scales::rescale(c(1:100), to=c(limit.low,limit.middle))
    values4 <- scales::rescale(c(1:100), to=c(minup,limit.low))
    values <- c(values4,values3,values2,values1)
    values <- scales::rescale(values, to=c(0,1))
  }

  cmap <- rev(c("#D6604D" ,"#F4A582" ,"#FDDBC7", "#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3"))

  p <- ggplot(data=data)+
    geom_brain(aes(fill=value,colour=value), atlas=atlas, hemi='left', side='lateral')+
    scale_fill_gradientn(colours = cmap,values=values) + 
    scale_color_gradientn(colours = cmap,values=values) + 
    theme_void() + theme(legend.position="none")
  
  ggsave(paste0(outpath,'_ll.png'),bg='transparent',plot = p, width = 4,height = 4,units = "cm",dpi = 600)

  p <- ggplot(data=data)+
    geom_brain(aes(fill=value,colour=value), atlas=atlas, hemi='left', side='medial')+
    scale_fill_gradientn(colours = cmap,values=values) +
    scale_color_gradientn(colours = cmap,values=values) +
    theme_void() + theme(legend.position="none")

  ggsave(paste0(outpath,'_lm.png'),bg='transparent',plot = p, width = 4,height = 4,units = "cm",dpi = 600)

  p <- ggplot(data=data)+
    geom_brain(aes(fill=value,colour=value), atlas=atlas, hemi='right', side='lateral')+
    scale_fill_gradientn(colours = cmap,values=values) +
    scale_color_gradientn(colours = cmap,values=values) +
    theme_void() + theme(legend.position="none")

  ggsave(paste0(outpath,'_rl.png'),bg='transparent',plot = p, width = 4,height = 4,units = "cm",dpi = 600)

  p <- ggplot(data=data)+
    geom_brain(aes(fill=value,colour=value), atlas=atlas, hemi='right', side='medial')+
    scale_fill_gradientn(colours = cmap,values=values) +
    scale_color_gradientn(colours = cmap,values=values) +
    theme_void() + theme(legend.position="none")

  ggsave(paste0(outpath,'_rm.png'),bg='transparent',plot = p, width = 4,height = 4,units = "cm",dpi = 600)

}