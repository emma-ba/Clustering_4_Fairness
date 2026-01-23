##### INIT #############
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 
library(tidyverse)

n_clust <- 15
algo <- 'kmeans'
thres <- 7.5

##### VIZ HISTO CLUSTER ######################

for(error_type in c('accuracy', 'fnr', 'fpr', 'precision', 'npr')){
  res_data <- read_csv2(paste0('./Results/res_', algo, n_clust, '_data_',error_type,'.csv'),
                        show_col_types=F) %>%
    mutate(charge_degree = c_charge_degree) %>%
    select(-c_charge_degree)
  # res_data %>% glimpse
  
  clust_col_name <- rank_col_name <- c()
  for(only_sensitive in c(T,F)){
    for(pred_as_input in c(T,F)){
      for(error_as_input in c(T,F)){
        clust_col_name <- c(clust_col_name, paste0('c_', 
                                 ifelse(only_sensitive, 'sens', 'all'),
                                 ifelse(pred_as_input, '_pred', ''),
                                 ifelse(error_as_input, '_err', '')) )
        rank_col_name <- c(rank_col_name, paste0('r_', 
                                ifelse(only_sensitive, 'sens', 'all'),
                                ifelse(pred_as_input, '_pred', ''),
                                ifelse(error_as_input, '_err', '')) )
      }
    }
  }
  
  factor_col <- c('sex', 'race', 'charge_degree', 'input_feature', 'cluster_id')
  
  res_data_long <- res_data %>% select(-all_of(clust_col_name)) %>%
    pivot_longer(
      cols = starts_with("r_"),
      names_to = "input_feature",
      names_prefix = 'r_',
      values_to = "cluster_id",
      values_drop_na = TRUE
    ) %>%
    select(-true_class, -predicted_class, -error_type) %>% 
    mutate_at(factor_col, as.factor)
  
  p <- res_data_long %>% mutate(error = as_factor(error)) %>%
    ggplot(aes(x=cluster_id)) + 
    geom_bar(aes(fill=error)) +
    facet_wrap(~input_feature, ncol=4) +
    theme_light() +
    scale_fill_brewer(palette = "Set1", direction = -1) + 
    guides(fill="none")
  
  p %>% print
  
  p %>% 
    ggsave(
      filename = paste0( './Results/Viz/res_', algo, n_clust, '_',error_type,'.pdf'),
      device = "pdf",
      height = 4, width = 8, units = "in")
  
}

##### VIZ HISTO BASIC ############
col_as_x <- c('sex', 'race', 'age', 
              'juv_misd_count', 'juv_fel_count', 'juv_other_count',
              'priors_count', 'charge_degree', 'decile_score')

for(error_type in c('accuracy', 'fnr', 'fpr', 'precision', 'npr')){
  basic_data <- read_csv2(paste0('./Results/res_', algo, n_clust, '_data_',error_type,'.csv'),
                        show_col_types=F) %>%
  mutate(charge_degree = c_charge_degree) %>%
  select(-c_charge_degree) %>%
  select(-true_class, -predicted_class, -error_type) %>%
  mutate(race = ifelse(race=='African-American', 'Afr.Am.', race)) %>%
  mutate(race = ifelse(race=='Native American', 'Nat.Am.', race)) %>%
  mutate(charge_degree = ifelse(charge_degree=='M', 'Misd.', 'Felony')) %>%
  mutate_all(as.factor) 

  # basic_data %>% glimpse
  
  data_long <- basic_data %>% 
    pivot_longer(
      cols = col_as_x,
      names_to = "feature",
      values_to = "feat_value",
      values_drop_na = TRUE
    ) 
  
  # data_long %>% glimpse
  all_val <- data_long$feat_value %>% unique %>% as.character %>% sort %>% rev
  new_level <- c(all_val[1:10], 0:100)
  
  p <- data_long %>% mutate(error = as_factor(error)) %>%
    mutate(feat_value = factor(feat_value, levels=new_level)) %>%
    mutate(feature = factor(feature, levels=col_as_x)) %>%
    ggplot(aes(x=feat_value)) + 
    geom_bar(aes(fill=error)) +
    facet_wrap(~feature, ncol=3, scale='free') +
    theme_light() +
    scale_fill_brewer(palette = "Set1", direction = -1) + 
    guides(fill="none")
  
  
  p %>% print
  
  p %>% 
    ggsave(
      filename = paste0( './Results/Viz/res_', algo, n_clust,'_basic_dist_',error_type,'.pdf'),
      device = "pdf",
      height = 12, width = 16, units = "in")
}
