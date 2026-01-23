##### INIT #############
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 
library(tidyverse)

n_clust <- 15
algo <- 'kmeans'
thres <- 7.5

basic_data <- read_csv2(paste0('compas_clustering_prep_thres_',thres,'.csv'),
                        show_col_types=F)

# Get overall error rates
basic_acc <- basic_data$error %>% abs %>% mean
basic_fnr <- basic_data %>% filter(true_class==1) %>% select(error) %>% as_vector %>% abs %>% mean
basic_fpr <- basic_data %>% filter(true_class==0) %>% select(error) %>% as_vector %>% abs %>% mean
basic_prec <- basic_data %>% filter(predicted_class==1) %>% select(error) %>% as_vector %>% abs %>% mean
basic_npr <- basic_data %>% filter(predicted_class==0) %>% select(error) %>% as_vector %>% abs %>% mean

recap <- read_csv2(file=paste0('./Results/res_', algo, n_clust, '_recap.csv'),
                   show_col_types=F)


##### VIZ RECAP ######################
recap %>% glimpse

recap_col_names <- c(
  paste0('r', c(1:n_clust), '_rate') ) %>% sort

recap_col_remove <- c(
  paste0('r', c(1:n_clust), '_n'),
  paste0('r', c(1:n_clust), '_error') ) %>% sort

error_type <- recap$error_type %>% unique %>% sort
hline <- data.frame(error_type, basic_error = c(basic_acc, basic_fnr, basic_fpr, basic_npr, basic_prec))

recap_long <- recap %>% select(-all_of(recap_col_remove)) %>%
  rename_with(., ~str_replace_all(., '_rate', '')) %>%
  pivot_longer(
    cols = starts_with("r"),
    names_to = "cluster_ranked",
    names_prefix = 'r',
    values_to = "error_rate",
    values_drop_na = TRUE
  ) %>% 
  mutate(only_sens = ifelse(only_sens, 'sens_only', 'all_feat')) %>%
  mutate(with_pred = ifelse(with_pred, 'with_pred', 'no_pred')) %>%
  mutate(with_err = ifelse(with_err, 'with_error', 'no_error')) 

factor_col <- recap_long %>% select(-error_rate) %>% colnames

recap_long <- recap_long %>% 
  mutate_at(factor_col, as.factor) %>%
  mutate(error_rate = as.numeric(error_rate))
recap_long %>% glimpse

p <- recap_long %>% ggplot(aes(cluster_ranked, error_rate, color=error_type,
                          alpha=0.5)) + 
  geom_point(aes(shape= with_err)) + 
  geom_hline(aes(yintercept=basic_error, colour=error_type, alpha=0.5), hline, linetype=2) +
  facet_grid(error_type ~ only_sens + with_pred, margins="am", scales='free') +
  theme_light() +
  theme(legend.position = "bottom") 

p %>% 
  ggsave(
    filename = paste0( './Results/Viz/res_', algo, n_clust, '_recap.pdf'),
    device = "pdf",
    height = 8, width = 8, units = "in")

p %>% print
