replace_outliers <- function(x) {
  mean_value <- mean(x)
  std_value <- sd(x)
  r_threshold <- mean_value + 3 * std_value
  l_threshold <- mean_value - 3 * std_value
  outliers <- x > r_threshold | x < l_threshold
  
  non_outliers <- x[!outliers]
  max_non_outliers <- max(non_outliers)
  min_non_outliers <- min(non_outliers)
  
  x[x > r_threshold] <- max_non_outliers
  x[x < l_threshold] <- min_non_outliers
  
  results <- list()
  results$data <- x
  results$outlier <- outliers
  
  return(results)
}