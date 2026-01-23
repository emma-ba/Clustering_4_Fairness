##### INIT #############
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 
library(tidyverse)
library(GGally)
library(patchwork)

n_clust <- 15
algo <- 'kmeans'

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
                'priors_count', 'charge_degree')
all_feat <- c(sens_feat, basic_feat)

##### VIZ OVERLAPS ######################
for(error_type in c('accuracy', 'fnr', 'fpr', 'precision', 'npr')){
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

##### VIZ DISTRIB ######################
for(error_type in c('accuracy', 'fnr', 'fpr', 'precision', 'npr')){
  res_data <- read_csv2(paste0('./Results/res_', algo, n_clust, '_data_',error_type,'.csv'), 
                        show_col_types=F) %>%
    mutate(charge_degree = c_charge_degree) %>%
    select(-c_charge_degree)
  # res_data %>% glimpse
  
  res_data_prep <- res_data %>% 
    select(all_of(c(all_feat, rank_col))) %>%
    mutate(charge = charge_degree) %>%
    select(-charge_degree) %>%
    mutate(priors = priors_count) %>%
    mutate(juv_fel = juv_fel_count) %>%
    mutate(juv_misd = juv_misd_count) %>%
    mutate(juv_other = juv_other_count) %>%
    select(-priors_count, -juv_fel_count, -juv_misd_count, -juv_other_count) %>%
    mutate(charge = ifelse(charge == 'M', 'Mis.', 'Fel.')) %>%
    mutate(charge = factor(charge)) %>%
    mutate(race = ifelse(race == 'African-American', 'Afr.Am.', race)) %>%
    mutate(race = ifelse(race == 'Native American', 'Nat.Am.', race)) %>%
    mutate(race = ifelse(race == 'Hispanic', 'Hisp.', race)) %>%
    mutate(race = ifelse(race == 'Caucasian', 'Cauc.', race)) %>%
    mutate_at(c('sex', 'race', 'charge', rank_col), as.factor)
  # res_data_prep %>% glimpse
  
  res_data_long <- res_data_prep %>% 
    pivot_longer(
      cols = starts_with("r_"),
      names_to = "input_clust",
      names_prefix = 'r_',
      values_to = "cluster",
      values_drop_na = TRUE
    ) 
  res_data_long <- res_data_long %>% 
    mutate(input_clust = factor(input_clust, levels=c(res_data_long$input_clust %>% unique %>% sort)))
  # res_data_long %>% glimpse
  
  p_sex <- res_data_long %>% ggplot(aes(cluster, sex)) + 
    geom_count(alpha=0.3, stroke=0) + 
    scale_radius(range=c(0,5)) +
    facet_grid(.~input_clust)  +
    theme_light() + 
    theme(
      panel.grid.minor.x = element_blank(),
      panel.border = element_blank()
    ) +
    guides(size = "none") + 
    labs(x = "")
  
  p_age <- res_data_long %>% ggplot(aes(cluster, age)) + 
    geom_count(alpha=0.3, stroke=0) + 
    scale_radius(range=c(0.2,5)) +
    facet_grid(.~input_clust)  +
    theme_minimal() + 
    theme(
      panel.grid.minor.x = element_blank()
    ) +
    guides(size = "none") +
    theme(strip.text = element_blank()) + 
    labs(x = "")
  
  p_race <- res_data_long %>% ggplot(aes(cluster, race)) + 
    geom_count(alpha=0.4, stroke=0) + 
    scale_radius(range=c(0,5)) +
    facet_grid(.~input_clust)  +
    theme_minimal() + 
    theme(
      panel.grid.minor.x = element_blank()
    ) +
    guides(size = "none") +
    theme(strip.text = element_blank()) + 
    labs(x = "")
  
  p_priors <- res_data_long %>% ggplot(aes(cluster, priors)) + 
    geom_count(alpha=0.4, stroke=0) + 
    scale_radius(range=c(0.2,5)) +
    facet_grid(.~input_clust)  +
    theme_minimal() + 
    theme(
      panel.grid.minor.x = element_blank()
    ) +
    guides(size = "none") +
    theme(strip.text = element_blank()) + 
    labs(x = "")
  
  p_charge <- res_data_long %>% ggplot(aes(cluster, charge)) + 
    geom_count(alpha=0.4, stroke=0) + 
    scale_radius(range=c(0,5)) +
    facet_grid(.~input_clust)  +
    theme_minimal() + 
    theme(
      panel.grid.minor.x = element_blank()
    ) + 
    guides(size = "none") +
    theme(strip.text = element_blank()) +
    labs(x = "")
  
  p_juv_fel <- res_data_long %>% ggplot(aes(cluster, juv_fel)) + 
    geom_count(alpha=0.4, stroke=0) + 
    scale_radius(range=c(0.2,5)) +
    facet_grid(.~input_clust)  +
    theme_minimal() + 
    theme(
      panel.grid.minor.x = element_blank()
    ) +
    guides(size = "none") +
    theme(strip.text = element_blank()) + 
    labs(x = "")
  
  p_juv_misd <- res_data_long %>% ggplot(aes(cluster, juv_misd)) +
    geom_count(alpha=0.4, stroke=0) + 
    scale_radius(range=c(0.2,5)) +
    facet_grid(.~input_clust)  +
    theme_minimal() + 
    theme(
      panel.grid.minor.x = element_blank()
    ) +
    guides(size = "none") +
    theme(strip.text = element_blank()) + 
    labs(x = "")
  
  p_juv_other <- res_data_long %>% ggplot(aes(cluster, juv_other)) + 
    geom_count(alpha=0.4, stroke=0) + 
    scale_radius(range=c(0.2,5)) +
    facet_grid(.~input_clust)  +
    theme_minimal() + 
    theme(
      panel.grid.minor.x = element_blank()
    ) +
    guides(size = "none") +
    theme(strip.text = element_blank())
    
  p_all <- p_sex / p_race / p_age / p_priors / p_charge /
    p_juv_fel / p_juv_misd / p_juv_other +
    plot_layout(heights = c(1, 4, 5, 4, 1, 2, 2, 2))
  # p_all %>% print
  
  p_all %>% 
    ggsave(
      filename = paste0( './Results/Viz/res_', algo, n_clust, '_',error_type,'_distrib.pdf'),
      device = "pdf",
      height = 12, width = 9, units = "in")
}
