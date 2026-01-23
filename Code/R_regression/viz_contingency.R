##### INIT #############
# setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 
# library(tidyverse)
# library(GGally)
# library(patchwork)
# 
# n_clust <- 20
# algo <- 'kmeans'
# thres <= 7.5
# error_type_list <- c('accuracy', 'fnr', 'fpr', 'precision', 'npr')
# error_type_list <- c('accuracy', 'fnr', 'fpr')

##### COL LIST ######################
rank_col <- clust_col <- c()
for(only_sensitive in c(T,F)){
  for(pred_as_input in c(T,F)){
    for(error_as_input in c(T,F)){
      clust_col <- c(rank_col, paste0('c_',
                                      ifelse(only_sensitive, 'sens', 'all'),
                                      ifelse(pred_as_input, '_pred', ''),
                                      ifelse(error_as_input, '_err', '')) )
      rank_col <- c(rank_col, paste0('r_',
                                     ifelse(only_sensitive, 'sens', 'all'),
                                     ifelse(pred_as_input, '_pred', ''),
                                     ifelse(error_as_input, '_err', '')) )
    }
  }
}

sens_feat <- c('sex', 'age', 'race')
basic_feat <- c('juv_fel_count', 'juv_misd_count', 'juv_other_count',
                'priors_count', 'charge_degree', 'decile_score')
all_feat <- c(sens_feat, basic_feat)


##### VIZ OVERLAPS ######################
for(error_type in error_type_list){
  res_data <- read_csv2(paste0('./Results/res_', algo, n_clust, '_data_',error_type,'.csv'), 
                        show_col_types=F) %>%
    mutate(charge_degree = c_charge_degree) %>%
    select(-c_charge_degree) 
  
  p <- res_data %>% 
    select(all_of(rank_col)) %>%
    mutate_all(as.factor) %>%
    ggpairs(
      upper = list(discrete = "count"), 
      diag = list(discrete = "countDiag"), 
      lower = 'blank'
    ) +
    theme_bw() + 
    theme(
      panel.grid.minor.x = element_blank(),
      panel.grid.minor.y = element_blank()
    )
  
  p %>% print
  
  p %>% 
    ggsave(
      filename = paste0( './Results/Viz/res_', algo, n_clust, '_',error_type,'_contingency.pdf'),
      device = "pdf",
      height = 9, width = 9, units = "in")
}