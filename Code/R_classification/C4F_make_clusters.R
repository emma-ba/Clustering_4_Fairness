##### INIT #############
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 
library(plyr)
library(tidyverse)
library(stats)
library(graphics)
library(precrec)
library(factoextra)
library(mltools)
library(data.table)
library(cowplot)

n_clust = 15
algo = 'kmeans'
thres <- 7.5

##### LOAD & PREP DATA #############
data <- read.csv('compas_clustering_prep.csv') %>% 
  as_tibble 
data %>% glimpse

##### PREP AS CLASSIFICATION #############
evalmod(scores = data$decile_score, labels = data$is_recid, 
        mode='rocprc', x_bins=10) %>%
  autoplot(type='b')

data <- data %>%
  mutate(true_class = is_recid) %>%
  select(-is_recid) %>%
  mutate(predicted_class = ifelse(decile_score < thres, 0, 1)) %>% 
  mutate(error = ifelse(predicted_class != true_class, 1, 0)) %>%
  mutate(error_type = ifelse(predicted_class == 1 & true_class == 1, 'TP', '')) %>%
  mutate(error_type = ifelse(predicted_class == 0 & true_class == 0, 'TN', error_type)) %>%
  mutate(error_type = ifelse(predicted_class == 1 & true_class == 0, 'FP', error_type)) %>%
  mutate(error_type = ifelse(predicted_class == 0 & true_class == 1, 'FN', error_type)) 
  
data %>% write_csv2(paste0('compas_clustering_prep_thres_',thres,'.csv'))
##### KMEANS CLUSTERING FUNCTION #############
make_clust <- function(data, plot_dir='./Results/Viz/', make_viz=F,
                        algo='kmeans', n_clust=7,
                         only_sensitive=F, error_as_input=F, pred_as_input=F,
                         error_type='fnr', iter = c(1:(n_clust-1)),
                         feat_scaling = 'zscore'){
  #### Check arguments ###
  # Error to target
  arg_error_type = c('accuracy', 'fnr', 'fpr', 'precision', 'npr')
  if(! error_type %in% arg_error_type){
    return(paste('error_type must be either:', paste(arg_error_type, collapse=', ')))
  }
  # Feature scaling
  arg_feat_scaling = c('none', 'zscore', 'bower')
  if(! feat_scaling %in% arg_feat_scaling){
    return(paste('feat_scaling must be either:', paste(feat_scaling, collapse=', ')))
  }
  # Use error as input for clustering
  if(! is_logical(error_as_input) ){
    return('error_as_input must be True/False')
  }
  
  ### Remove data points to focus on specific errors ###
  # Filter specific errors
  if(error_type == 'fnr'){
    data <- data %>% filter(true_class==1)
  } else if(error_type == 'fpr'){
    data <- data %>% filter(true_class==0)
  } else if(error_type == 'precision'){
    data <- data %>% filter(predicted_class==1)
  } else if(error_type == 'npr'){
    data <- data %>% filter(predicted_class==0)
  }
  
  ### Prepare data for clustering ###
  # One-hot encoding
  data_clust <- data %>%
    mutate(sex = ifelse(sex=='Male', 0, 1)) %>%
    mutate(sex = as.integer(sex)) %>%
    mutate(c_charge_degree = ifelse(c_charge_degree=='M', 0, 1)) %>%
    mutate(c_charge_degree = as.integer(c_charge_degree)) %>%
    mutate(race = ifelse(race=='African-American', 'Afr.Am.', race)) %>%
    mutate(race = ifelse(race=='Native American', 'Nat.Am.', race)) %>%
    mutate(race = race %>% as_factor) 
  
  data_clust <- data_clust %>% as.data.table %>% one_hot %>% as_tibble
  
  # Remove input columns
  data_clust <- data_clust %>% select(-true_class, -predicted_class, -error_type) 
  if(only_sensitive){
    data_clust <- data_clust %>% select(-juv_fel_count, -juv_misd_count, -juv_other_count,
                                        -priors_count, -c_charge_degree)
  }
  if(!error_as_input){
    data_clust <- data_clust %>% select(-error)
  } 
  if(!pred_as_input){
    data_clust <- data_clust %>% select(-decile_score)
  } 
  if(feat_scaling == 'z_score'){
    data_clust <- data_clust %>% scale %>% as_tibble
  }
  
  ### Do clustering ### 
  clust_res_all <- clust_pca_all <- clust_silh_all <- list()
  for(i in iter){
    clust_res <- data_clust %>% eclust(algo, k=i+1, nstart=20, graph=F)
    clust_res$data_full <- data %>% mutate(cluster = clust_res$cluster)
    clust_res_all[[i]] <- clust_res
    
    if(make_viz){
      cluster_pca <- clust_res %>% fviz_cluster(main = '',
                                   star.plot = F, geom = 'point', 
                                   palette = "jco", ggtheme = theme_minimal()) 
      cluster_silh <- clust_res %>% fviz_silhouette(main = '', print.summary=F,
                                      palette = "jco", ggtheme = theme_minimal()) 
      
      clust_pca_all[[i]] <- cluster_pca
      clust_silh_all[[i]] <- cluster_silh
    }
  }
  
  # Make grid plots
  if(make_viz){
    plot_grid(plotlist = clust_pca_all, ncol=3) %>% ggsave(
      filename = paste0( plot_dir, paste0(algo, n_clust, '_pca_'),
                         error_type, '_', feat_scaling,
                         ifelse(only_sensitive,'_sens',''),
                         ifelse(pred_as_input,'_pred',''),
                         ifelse(error_as_input,'_err',''), 
                         '.pdf'),
      device = "pdf",
      height = 8, width = 12, units = "in")
    
    plot_grid(plotlist = clust_silh_all, ncol=3) %>% ggsave(
      filename = paste0( plot_dir, paste0(algo, n_clust, '_silh_'),
                         error_type, '_', feat_scaling,
                         ifelse(only_sensitive,'_sens',''),
                         ifelse(pred_as_input,'_pred',''),
                         ifelse(error_as_input,'_err',''), 
                         '.pdf'),
      device = "pdf",
      height = 8, width = 12, units = "in")
  }
  
  return(clust_res_all)
}

##### ANALYSE CLUSTERS FUNCTION ###############
print_clust_err <- function(res){
  res <- res[[res %>% length]]
  print('')
  res$data %>% colnames %>% print
  
  print('')
  paste('Global Error Rate:',
        res$data_full %>% select(error) %>% as_vector %>% abs %>% mean) %>% print
  
  c_list <- res$cluster %>% unique %>% sort
  for(c in c_list){
    print('')
    paste('Cluster', c) %>% print
    d_clust <- res$data_full %>% filter(cluster == c)
    
    d <- d_clust
    paste0('Error Rate = ', d %>% filter(error==1) %>% nrow, '/', d %>% nrow, 
           ' = ', d %>% select(error) %>% as_vector %>% abs %>% mean %>% round(digits=4)
    ) %>% print
  }
}
get_clust_err <- function(res_data){
  c_list <- res_data$cluster %>% unique
  c_size <- c_err <- c_rate <- c()
  for(c in c_list){
    d <- res_data %>% filter(cluster == c)
    c_size <- c(c_size, d %>% nrow)
    c_err <- c(c_err, d %>% filter(error==1) %>% nrow)
    c_rate <- c(c_rate, d %>% select(error) %>% as_vector %>% abs %>% mean %>% round(digits=4))
  }
  res_tbl <- tibble(cluster=c_list, size=c_size, 
                    error=c_err, error_rate=c_rate) %>%  
    arrange(error_rate) %>%
    mutate(c_rank=c(1:length(c_list)))
  return(res_tbl)
}
##### RUN CLUSTERING FUNCTION ###############
run_clustering <- function(error_type_list = c('accuracy', 'fnr', 'fpr', 'precision', 'npr'), 
                           n_clust=7, make_viz=F, algo='kmeans'){
  # Init Recap Table
  recap_col_names <- c(
    paste0('r', c(1:n_clust), '_n'),
    paste0('r', c(1:n_clust), '_error'),
    paste0('r', c(1:n_clust), '_rate') ) %>% sort
  
  vec <- setNames(rep("", 4+n_clust*3), c('error_type',
    'only_sens', 'with_pred', 'with_err', recap_col_names)
  )
  whole_recap <- bind_rows(vec)[0, ]
  row_index <- col_index <- 0
  
  for(error_type in error_type_list){
    whole_res_data <- tibble()
    for(only_sensitive in c(T,F)){
      for(pred_as_input in c(T,F)){
        for(error_as_input in c(T,F)){
          row_index <- row_index + 1
          
          # Get data
          res <- data %>% 
            make_clust(algo=algo, n_clust=n_clust,
                         pred_as_input=pred_as_input,
                         only_sensitive=only_sensitive, 
                         error_as_input=error_as_input, 
                         error_type=error_type,
                         iter = c(n_clust-1),
                         make_viz=make_viz)
          
          res <- res[[res %>% length]]
          
          # Get recap
          new_recap <- res$data_full %>% get_clust_err()
          
          whole_recap[row_index, 1] <- error_type
          whole_recap[row_index, 2] <- only_sensitive %>% as.character
          whole_recap[row_index, 3] <- pred_as_input %>% as.character
          whole_recap[row_index, 4] <- error_as_input %>% as.character
          col_index <- 4
          
          for(i in 1:n_clust){
            col_index <- col_index + 1
            whole_recap[row_index, col_index] <- new_recap$error[i] %>% as.character
            col_index <- col_index + 1
            whole_recap[row_index, col_index] <- new_recap$size[i] %>% as.character
            col_index <- col_index + 1
            whole_recap[row_index, col_index] <- new_recap$error_rate[i] %>% as.character
          }
          
          # Concat data
          clust_col_name <- paste0('c_', ifelse(only_sensitive, 'sens', 'all'),
                                  ifelse(pred_as_input, '_pred', ''),
                                  ifelse(error_as_input, '_err', ''))
          rank_col_name <- paste0('r_', ifelse(only_sensitive, 'sens', 'all'),
                                  ifelse(pred_as_input, '_pred', ''),
                                  ifelse(error_as_input, '_err', ''))
          
          if(whole_res_data %>% nrow == 0){
            whole_res_data <- res$data_full # %>%
              # mutate(charge_degree = c_charge_degree) %>%
              # select(-c_charge_degree) 
          } else {
            whole_res_data <- whole_res_data %>% mutate(cluster = res$data_full$cluster)
          }
          
          whole_res_data <- whole_res_data %>% mutate(rank = cluster) 
          whole_res_data$rank = whole_res_data$rank %>% 
                          mapvalues(from=new_recap$cluster, to=new_recap$c_rank)
          
                              
          whole_data_col_names <- whole_res_data %>% colnames
          whole_data_col_names[(whole_data_col_names %>% length)-1] = clust_col_name
          whole_data_col_names[(whole_data_col_names %>% length)] = rank_col_name
          whole_res_data <- whole_res_data %>% setNames(whole_data_col_names) 
        }
      }
    }
    whole_res_data %>% write_csv2(paste0('./Results/res_', algo, n_clust, '_data_',error_type,'.csv'))
  }
  
  whole_recap <- whole_recap %>% mutate(across(recap_col_names, as.numeric))
  whole_recap %>% write_csv2(paste0('./Results/res_', algo, n_clust, '_recap.csv'))
  whole_recap %>% print(n=whole_recap%>%nrow)
  return(whole_recap)
}

##### RUN CLUSTERING ###############
whole_recap <- run_clustering(n_clust=n_clust, algo=algo)

##### basic error rates #############
# Accuracy
data$error %>% abs %>% mean
# FNR
data %>% filter(true_class==1) %>% select(error) %>% as_vector %>% abs %>% mean
# FPR
data %>% filter(true_class==0) %>% select(error) %>% as_vector %>% abs %>% mean
# 1 - Precision
data %>% filter(predicted_class==1) %>% select(error) %>% as_vector %>% abs %>% mean
# NPR
data %>% filter(predicted_class==0) %>% select(error) %>% as_vector %>% abs %>% mean

  # # FNR
  # d <- d_clust %>% filter(true_class==1)
  # paste0('FNR = ', d %>% filter(error==1) %>% nrow, '/', d %>% nrow, 
  #        ' = ', d %>% select(error) %>% as_vector %>% abs %>% mean %>% round(digits=4)
  # ) %>% print
  # 
  # # FPR
  # d <- d_clust %>% filter(true_class==0)
  # paste0('FPR = ', d %>% filter(error==1) %>% nrow, '/', d %>% nrow, 
  #        ' = ', d %>% select(error) %>% as_vector %>% abs %>% mean %>% round(digits=4)
  # ) %>% print
  # 
  # # FNR
  # d <- d_clust %>% filter(predicted_class==1)
  # paste0('FNR = ', d %>% filter(error==1) %>% nrow, '/', d %>% nrow, 
  #        ' = ', d %>% select(error) %>% as_vector %>% abs %>% mean %>% round(digits=4)
  # ) %>% print
  # 
  # # FPR
  # d <- d_clust %>% filter(predicted_class==0)
  # paste0('FPR = ', d %>% filter(error==1) %>% nrow, '/', d %>% nrow, 
  #        ' = ', d %>% select(error) %>% as_vector %>% abs %>% mean %>% round(digits=4)
  # ) %>% print
