##### INIT #############
library(tidyverse)
library(corrplot)
library(GGally)
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 

##### LOAD & CHECK DATA #############
data_orig <- read.csv('compas-scores-two-years.csv') %>% as_tibble

# data_orig %>% glimpse
# data_orig$c_charge_desc %>% unique
colnames(data_orig)

data_orig %>% 
  filter(decile_score != decile_score.1) %>%
  glimpse

data_orig %>% 
  filter(priors_count != priors_count.1) %>%
  glimpse

data_orig %>% 
  filter(juv_fel_count + juv_misd_count + juv_other_count > priors_count) %>%
  glimpse

##### PREP DATA #############
data <- data_orig %>% 
  mutate(priors_count = priors_count.1) %>%
  select(sex, age, race, 
         juv_fel_count, juv_misd_count, juv_other_count, priors_count,
         c_charge_degree, 
         decile_score, is_recid,
         ) %>% 
  drop_na

data %>% glimpse
write_csv(data, 'compas_clustering_prep.csv')

##### PREP AS CLASSIFICATION #############
thres <- 5.5

data_classpb <- data %>% 
  mutate(true_class = is_recid) %>%
  select(-is_recid) %>%
  mutate(predicted_class = ifelse(decile_score < thres, 0, 1)) %>% 
  mutate(error = ifelse(predicted_class != true_class, 1, 0)) %>%
  mutate(TP = ifelse(predicted_class == 1 & true_class == 1, 1, 0)) %>%
  mutate(TN = ifelse(predicted_class == 0 & true_class == 0, 1, 0)) %>%
  mutate(FP = ifelse(predicted_class == 1 & true_class == 0, 1, 0)) %>%
  mutate(FN = ifelse(predicted_class == 0 & true_class == 1, 1, 0)) 

data_classpb %>% glimpse

write_csv(data_classpb, 'compas_clustering_classpb.csv')

##### PREP AS REGRESSION #############
data_regpb <- data %>% 
  mutate(true_score = is_recid*9 + 1) %>%
  mutate(predicted_score = decile_score) %>%
  select(-is_recid, -decile_score) %>%
  mutate(error = true_score - predicted_score) 

data_regpb %>% glimpse

write_csv(data_regpb, 'compas_clustering_regpb.csv')

##########################
##### BASIC DATA EXPLORATION #############
data %>% 
  mutate(sex = ifelse(sex=='Female', 1, 0)) %>%
  mutate(charge = ifelse(c_charge_degree=='F', 1, 0)) %>%
  select(-race, -c_charge_degree) %>%
  rename(j_fel = juv_fel_count) %>%
  rename(j_misd = juv_misd_count) %>%
  rename(j_other = juv_other_count) %>%
  rename(priors = priors_count) %>%
  rename(recid = is_recid) %>%
  rename(compas = decile_score) %>%
  cor %>% 
  corrplot.mixed(upper='ellipse', order='hclust')

data %>% 
  mutate(across(where(is.character), as.factor)) %>%
  rename(j_fel = juv_fel_count) %>%
  rename(j_misd = juv_misd_count) %>%
  rename(j_other = juv_other_count) %>%
  rename(priors = priors_count) %>%
  rename(charge = c_charge_degree) %>%
  rename(recid = is_recid) %>%
  rename(compas = decile_score) %>%
  ggpairs(mapping = aes(color = recid, alpha=0.1),
          lower = list(continuous = wrap('points', size=0.01),
                       combo = wrap('box_no_facet', size=0.01),
                       mapping = aes(color = recid, alpha=0.1, size=0.01)),
          upper = list(continuous = wrap('points', size=0.01),
                       combo = wrap('facetdensity', size=0.01),
                       mapping = aes(color = recid, alpha=0.1, size=0.01))
  ) + theme_light()

