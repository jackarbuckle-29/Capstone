
library(tidyverse)
library(DescTools)
library(janitor)
library(ggplot2)
library(jpeg)
library(chron)
library(lubridate)
library(anytime)
library(ggpubr)
library(devtools) 
#install_github("AppliedDataSciencePartners/xgboostExplainer")
library(xgboost) # Load XGBoost
library(caret) # Load Caret
library(xgboostExplainer) # Load XGboost Explainer
library(pROC) # Load proc
library(SHAPforxgboost) # Load shap for XGBoost
library(plyr)
library(Metrics)

# FOR WEB SCRAPING
library(httr) 
library(rvest) 
library(polite) 
library(jsonlite) 

# setting ND colors 
nd_navy <- "#0C2340"
nd_green <- "#00843D"
nd_gold <- "#C99700"

calc_num_zeros_by_col <- function(df_title) {
  cols_vec <- colnames(df_title)
  zeros_count_vec <- rep("", length=length(cols_vec))
  zeros_test_vec <- rep("", length=length(cols_vec))

  for (i in 1:length(cols_vec)){
    zeros_count_vec[i] <- sum(df_title[,i] == 0)
    zeros_test_vec[i] <- sum(df_title[,i] == 0 | df_title[,i] == "0")
  }

  zero_count_df <- cbind.data.frame(cols_vec, as.numeric(zeros_count_vec))
  names(zero_count_df) <- c("column_name", "number_zeros")

  # Append df_title to the dataframe name
  assign(paste0("zero_count_df_", deparse(substitute(df_title))), zero_count_df,
         envir = .GlobalEnv)
}

#soc1_data <- read.csv("SOCdata1.csv") #reading in data

high_speed_dist_var_name <- "total_high_speed_distance"

soc0_data <- read.csv("soc0_data_cleaned.csv") 
soc1_data <- read.csv("soc1_data_cleaned.csv") 

not_include_indoor_vec <- c("duration", "meta", "total_effort", 
                            "heart_rate", "hr", "velocity", "acceleration",
                            "deceleration", "metre", "meterage",
                            "exertion_index","max_vel_max", 
                            "footstrikes", "running_series_count", 
                            "running_imbalance"#, #distance
                            )

length(names(soc0_data)) #826 columns
length(names(soc1_data)) #701 columns


# indoor_diff_01 <- setdiff(names(soc0_data), names(soc1_data))
# indoor_diff_10 <- setdiff(names(soc1_data), names(soc0_data))
# full_indoor_diff <- c(indoor_diff_01, indoor_diff_10)

# testing_vec <- c("velocity_band_6_distance_band_1_total_efforts", 
#                  "velocity_band_7_distance_band_1_total_efforts", 
#                  "velocity_band_7_recovery_band_2_total_efforts" , 
#                  "velocity_b2_average_effort_distance_gen_2_set_2",
#                  "velocity_b2_total_efforts_gen_2_set_2")

#names(soc1_data) <- str_replace_all(names(soc1_data), "_b([0-9])", "_band_\\1")

# diff01 <- setdiff(names(soc0_data), names(soc1_data))
# diff10 <- setdiff(names(soc1_data), names(soc0_data))
# 
# full_diff <- c(diff01, diff10)
# length(full_diff) #245 originally

cols_exclude <- c("mii", "metabolic", "heart_rate", "_set_2")
soc0_data <- soc0_data %>% 
  select(!contains(cols_exclude))

soc1_data <- soc1_data %>% 
  select(!contains(cols_exclude))

soc1_data <- soc1_data %>%
  dplyr::rename(
    velocity_band_1_total_distance = velocity_band_1_0_45mph_total_distance,
    velocity_band_2_total_distance = velocity_band_2_45_4_05mph_total_distance,
    velocity_band_3_total_distance = velocity_band_3_4_05_7_14mph_total_distance,
    velocity_band_4_total_distance = velocity_band_4_7_14_9_64mph_total_distance,
    velocity_band_5_total_distance = velocity_band_5_9_64_12_12mph_total_distance,
    velocity_band_6_total_distance = velocity_band_6_12_12_14_61mph_total_distance,
    velocity_band_7_total_distance = velocity_band_7_14_61_mph_total_distance, 
    high_speed_distance = total_high_speed_distance, 
    pl_hi=pl_hi_session
  )

soc0_data$med_and_hi_ima_r <- soc0_data$ima_co_d_right_medium_1_0+soc0_data$ima_co_d_right_high_1_0
soc0_data$med_and_hi_ima_l <- soc0_data$ima_co_d_left_medium_1_0+soc0_data$ima_co_d_left_high_1_0
#names(soc1_data)

fr0 <- soc0_data %>% select(contains("ima_free_running")) %>% names()
fr1 <- soc1_data %>% select(contains("ima_free_running")) %>% names()

free_run_exclude <- setdiff(fr0, fr1)

soc0_data <- soc0_data %>% select(-contains(c(free_run_exclude)))

diff01 <- setdiff(names(soc0_data), names(soc1_data))
diff10 <- setdiff(names(soc1_data), names(soc0_data))


full_diff <- c(diff01, diff10)
length(full_diff) #245 originally

soc1_indoor_full_df <- soc1_data %>%
  select(-contains(not_include_indoor_vec))

soc0_indoor_full_df <- soc0_data %>%
  select(-contains(not_include_indoor_vec))


# should only look at differences for variables available indoors
indoor_diff_01 <- setdiff(names(soc0_indoor_full_df), names(soc1_indoor_full_df))
indoor_diff_10 <- setdiff(names(soc1_indoor_full_df), names(soc0_indoor_full_df))
full_indoor_diff <- c(indoor_diff_01, indoor_diff_10)

exclude_to_combine <- full_indoor_diff
# soc0_indoor_full_df <- soc0_indoor_full_df %>% select(-contains(c(exclude_to_combine)))
# soc1_indoor_full_df <- soc1_indoor_full_df 

soc1_indoor_full_df <- soc1_indoor_full_df %>%
  #select(-contains(not_include_indoor_vec)) %>% 
  select(-contains(c(exclude_to_combine)))

soc0_indoor_full_df <- soc0_indoor_full_df %>%
  #select(-contains(not_include_indoor_vec)) %>% 
  select(-contains(c(exclude_to_combine)))

# should only look at differences for variables available indoors
indoor_diff_01 <- setdiff(names(soc0_indoor_full_df), names(soc1_indoor_full_df))
indoor_diff_10 <- setdiff(names(soc1_indoor_full_df), names(soc0_indoor_full_df))
full_indoor_diff <- c(indoor_diff_01, indoor_diff_10)


# confirm names are sorted alphabetically
# names(soc0_mod_sort)
# names(soc1_mod_sort)


dist_cols_remove0 <- soc0_indoor_full_df %>% 
  select(contains("distance")) %>% names()
dist_cols_remove1 <- soc1_indoor_full_df %>% 
  select(contains("distance")) %>% names()
dist_cols_remove <- c(dist_cols_remove0, dist_cols_remove1)

dist_cols_remove <- dist_cols_remove[dist_cols_remove != "high_speed_distance"]  

soc0_indoor_full_df <- soc0_indoor_full_df %>%
  select(-contains(dist_cols_remove))

soc1_indoor_full_df <- soc1_indoor_full_df %>%
  select(-contains(dist_cols_remove))
#soc_full_mod_df$high_speed_distance

soc0_indoor_full_df$team_gender <- 0
soc1_indoor_full_df$team_gender <- 1

soc0_mod_sort <- soc0_indoor_full_df %>% select(sort(names(.)))
soc1_mod_sort <- soc1_indoor_full_df %>% select(sort(names(.)))

dim(soc0_mod_sort)
dim(soc1_mod_sort) 



#binding dataframes
soc_full_mod_df <- rbind.data.frame(soc0_mod_sort, soc1_mod_sort)

# set.seed(33) #setting seed for reproducing same samples
# 
# #selecting only outdoor rows to predict high speed distance
# soc1_outdoor_df <- soc1_data %>%
#   filter(location == "outdoor")
# sample_rows_2k <- sample(dim(soc1_outdoor_df)[1], 2000)
# 
# num_col_names_indoor <- num_col_names[num_col_names %in% names(soc1_indoor_full_df)] 
# 
# to_remove_vec <- c("unix", "position_name")
# 
# soc1_for_mod_df <- soc1_data %>% 
#   select(all_of(c("unique_session", "name_id", "total_high_speed_distance",
#                   num_col_names_indoor))) %>%
#   select(-contains(to_remove_vec))

players <- unique(soc_full_mod_df$name_id)

z_prep_mod_df <- soc_full_mod_df

full_df <- as.data.frame(matrix(ncol=dim(z_prep_mod_df)[2]))
names(full_df) <- names(z_prep_mod_df)

for(i in 1:length(players)){
  player_df <- z_prep_mod_df[z_prep_mod_df$name_id == players[i],]
  
  df_non_num <- player_df %>% select_if(is.character)
  df_num <- player_df %>% select_if(is.numeric)
  
  df_num <- scale(df_num)
  combined <- cbind(df_non_num, df_num)
  full_df <- rbind(full_df, combined)
}

full_df <- full_df[-1,]
names(full_df) <- paste0(names(full_df), "_z") 

z_score_df <- full_df %>% select(-contains(c("name_id")))

comb_z_df <- merge(soc_full_mod_df, z_score_df, 
                   by.x = "unique_session", by.y = "unique_session_z",
                   all.x=TRUE)
#head(comb_z_df)





num_col_names_indoor <- num_col_names[num_col_names %in% names(soc_full_mod_df)]

to_remove_vec <- c("unix", "position_name")

test_df <- soc_full_mod_df %>%
  select(all_of(c("unique_session","name_id", "high_speed_distance",
                  num_col_names_indoor))) %>%
  select(-contains(to_remove_vec))

comb_z_df_outdoor <- comb_z_df %>%
  filter(high_speed_distance != 0)

low_lim <- quantile(comb_z_df_outdoor$high_speed_distance, .10)
up_lim <- quantile(comb_z_df_outdoor$high_speed_distance, .90)

include_mod_cols <- c("high_speed_distance",
                      names(test_df), names(z_score_df))

comb_z_df_mod <- comb_z_df_outdoor[, names(comb_z_df_outdoor) %in% include_mod_cols]

model_df <- comb_z_df_mod %>% 
  select(!contains(c("unique_session", "name_id", "high_speed_distance_z", "team_gender_z", 
                     "activity_type_z", "date_z", "location_z"))) %>%
  #filter(high_speed_distance >= low_lim) %>% #& total_high_speed_distance <= up_lim)  %>%
  select(all_of("high_speed_distance"), everything())

set.seed(33) # Set Seed

# Split the data into training and testing sets
split_ratio <- 0.75
split_index <- floor(nrow(model_df) * split_ratio)
train_data <- model_df[1:split_index, ]
test_data <- model_df[(split_index + 1):nrow(model_df), ]

train_y = train_data[,1]
train_x = train_data[,-1]

test_y = test_data[,1]
test_x = test_data[,-1]

response <- model_df$high_speed_distance

train_response <- response[0:split_index]
test_response <- response[(split_index+1):length(response)]


# Create training data XGBOOST
dtrain <- xgb.DMatrix(data = as.matrix(train_data[ ,-1]), label = train_response)
# Create test data XGBOOST
dtest <- xgb.DMatrix(data = as.matrix(test_data[ ,-1]),label = test_response)

selected_sport <- "Combined Soccer"
xgb_viz_title <- "XGBoost Model Actual vs Predicted High Speed Distance (covered)"

bst_split_mod_pre_tune <- xgboost(
    data = as.matrix(train_data[, -1]),  # Exclude the response variable
    label = train_data$high_speed_distance,
    booster = "gblinear",  # Use linear booster for regression
    objective = "reg:linear",  # Specify regression as the objective
    eval_metric = "rmse",  # Evaluation metric (Root Mean Squared Error)
    nrounds = 2000,  # Number of boosting rounds (you can adjust this)
    print_every_n = 20)
  
# Make predictions on the test data
bst_preds_pre_tune <- predict(bst_split_mod_pre_tune, as.matrix(test_data[, -1]))
  
bst_actual_pre_tune <- test_data$high_speed_distance
  
# Calculate RMSE (Root Mean Squared Error) for model evaluation
rmse <- sqrt(mean((bst_preds_pre_tune - test_data$high_speed_distance)^2))
#You can also inspect the model's feature importance if needed
importance <- xgb.importance(feature_names = colnames(as.matrix(train_data[, -1])),
                             model = bst_split_mod_pre_tune)

#preds1 <- predict(bst_1, dtest)
bst_split_pred_data_pre_tune <- cbind.data.frame(bst_preds_pre_tune, bst_actual_pre_tune)
  
names(bst_split_pred_data_pre_tune) <- c("predicted", "actual")

# calc difference as actual-predicted
bst_split_pred_data_pre_tune$difference <- bst_split_pred_data_pre_tune$actual - bst_split_pred_data_pre_tune$predicted 
bst_split_pred_data_pre_tune$time <- "pre-tune"

pre_tune_diff_summary_soc1 <- summary(bst_split_pred_data_pre_tune$difference)

# bst_split_pred_data_pre_tune %>% 
#   filter(bst_split_pred_data_pre_tune$difference == max(bst_split_pred_data_pre_tune$difference) | 
#          bst_split_pred_data_pre_tune$difference == min(bst_split_pred_data_pre_tune$difference))  

# DATA VIZ
pred_viz_pre_tune <- ggplot(bst_split_pred_data_pre_tune, aes(x = predicted, y = actual)) +
  geom_point(color=nd_navy) +
  geom_smooth(color=nd_gold) + theme_minimal() +
  labs(title = xgb_viz_title, 
    subtitle = paste(selected_sport, "Data;",
                     "High Speed Distance between:", 
                     low_lim, "&", up_lim, sep=" "))


ggplot(bst_split_pred_data_pre_tune, aes(x = predicted, y = actual)) +
  geom_point(color=nd_navy) +
  geom_smooth(color=nd_gold) + theme_minimal() +
  labs(title = xgb_viz_title, 
    subtitle = paste(selected_sport, "Data;",
                     "High Speed Distance between:", 
                     low_lim, "&", up_lim, sep=" ")) + 
  ylim(0,5000) + xlim(-5000,8000)

viz_time <- "pre-tuning"
diff_hist_pre_tune <- ggplot(bst_split_pred_data_pre_tune,
       aes(x = difference, #fill = color_condition_wt,
           y = after_stat(count / sum(count)))) +
  geom_histogram(fill = nd_navy, alpha=0.5) +
  labs(title = "Histogram of Difference between Prediction vs Actual",
       subtitle= paste0(selected_sport, " High Speed Distance ", viz_time),
       x = "Difference between Prediction and Actual",
       y = "Percent Frequency") +
  theme_minimal() + ylim(0, .25) + xlim(-750, 500)

# Be Careful - This can take a very long time to run
max_depth_vals <- c(3, 5, 7, 10) # Create vector of max depth values
min_child_weight <- c(1,3,5,7, 10) # Create vector of min child values

# Expand grid of parameter values
cv_params <- expand.grid(max_depth_vals, min_child_weight)
names(cv_params) <- c("max_depth", "min_child_weight")
# Create results vector
rmse_vec  <- rep(NA, nrow(cv_params)) 
# Loop through results

for(i in 1:nrow(cv_params)){
  set.seed(111111)
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     
                     nfold = 5, # Use 5 fold cross-validation
                     
                     eta = 0.1, # Set learning rate
                     max.depth = cv_params$max_depth[i], # Set max depth
                     min_child_weight = cv_params$min_child_weight[i], # Set minimum number of samples in node to split
                     
                     
                     nrounds = 1000, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
                     
  ) # Set evaluation metric to use
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}


# Join results in dataset
res_db <- cbind.data.frame(cv_params, rmse_vec)
names(res_db)[3] <- c("rmse") 
res_db$max_depth <- as.factor(res_db$max_depth) # Convert tree number to factor for plotting
res_db$min_child_weight <- as.factor(res_db$min_child_weight) # Convert node size to factor for plotting

# Print AUC heatmap
g_2 <- ggplot(res_db, aes(y = max_depth, x = min_child_weight, fill = rmse)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = nd_green, # Choose low color
                       mid = "white", # Choose mid color
                       high = nd_navy, # Choose high color
                       midpoint =mean(res_db$rmse), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Minimum Child Weight", y = "Max Depth", fill = "RMSE") # Set labels
g_2 # Generate plot

# RESULTS -- Best Max.Depth = 5
# RESULTS -- BEST Min.Child.Weight = 1

bst_max_depth <- res_db[res_db$rmse==min(res_db$rmse),]$max_depth
bst_min_child <- res_db[res_db$rmse==min(res_db$rmse),]$min_child_weight

gamma_vals <- c(0, 0.05, 0.1, 0.15, 0.2, .25) # Create vector of gamma values

# Be Careful - This can take a very long time to run
set.seed(111111)
rmse_vec  <- rep(NA, length(gamma_vals))
for(i in 1:length(gamma_vals)){
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     nfold = 5, # Use 5 fold cross-validation
                     eta = 0.1, # Set learning rate
                     max.depth = bst_max_depth, # Set max depth
                     min_child_weight = bst_min_child, # Set min n of samples in node to split
                     gamma = gamma_vals[i], # Set minimum loss reduction for split
                     nrounds = 1000, # Set number of rounds
                     early_stopping_rounds = 20, #number of rounds to stop if no improvement
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
  ) # Set evaluation metric to use
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
}

# Lets view our results to identify the value of gamma to use:

# Gamma results
# Join gamma to values
gam_df <- cbind.data.frame(gamma_vals, rmse_vec)

bst_gamma <- gam_df[gam_df$rmse_vec==min(gam_df$rmse_vec),]$gamma_vals

# Be Careful - This can take a very long time to run
subsample <- c(0.6, 0.7, 0.8, 0.9, 1) # Create vector of subsample values
colsample_by_tree <- c(0.6, 0.7, 0.8, 0.9, 1) # Create vector of col sample values

# Expand grid of tuning parameters
cv_params <- expand.grid(subsample, colsample_by_tree)
names(cv_params) <- c("subsample", "colsample_by_tree")
# Create vectors to store results
rmse_vec <- rep(NA, nrow(cv_params)) 
# Loop through parameter values
for(i in 1:nrow(cv_params)){
  set.seed(111111)
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     nfold = 5, # Use 5 fold cross-validation
                     eta = 0.1, # Set learning rate
                     max.depth = bst_max_depth, # Set max depth
                     min_child_weight = bst_min_child, #Set min child depth
                     gamma = bst_gamma, # Set minimum loss reduction for split
                     subsample = cv_params$subsample[i], #prop of training data used in tree
                     colsample_bytree = cv_params$colsample_by_tree[i], 
                     #num of variables to use in each tree
 
                     nrounds = 1000, # Set number of rounds
                     early_stopping_rounds = 20, #rounds to stop at if no improvement
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
  ) # Set evaluation metric to use
  
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}

# visualise tuning sample params

res_db <- cbind.data.frame(cv_params, rmse_vec)
names(res_db)[3] <- c("rmse") 
res_db$subsample <- as.factor(res_db$subsample) # Convert tree number to factor for plotting
res_db$colsample_by_tree <- as.factor(res_db$colsample_by_tree) # Convert node size to factor for plotting

g_4 <- ggplot(res_db, aes(y = colsample_by_tree, x = subsample, fill = rmse)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = nd_green, # Choose low color
                       mid = "white", # Choose mid color
                       high = nd_navy, # Choose high color
                       midpoint =mean(res_db$rmse), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Subsample", y = "Column Sample by Tree", fill = "RMSE") # Set labels
g_4 # Generate plot

bst_col_samp <- res_db[res_db$rmse == min(res_db$rmse), ]$colsample_by_tree
bst_sub_samp <- res_db[res_db$rmse == min(res_db$rmse), ]$subsample

# Be Careful - This can take a very long time to run
lambda_vals <- c(.01, .1, .25, .5) # Create vector of lambda values
alpha_vals <- c(.01, .1, .25, .5) # Create vector of alpha values

# Expand grid of tuning parameters
cv_params <- expand.grid(lambda_vals, alpha_vals)
names(cv_params) <- c("lambda", "alpha")
# Create vectors to store results
rmse_vec <- rep(NA, nrow(cv_params)) 
# Loop through parameter values
for(i in 1:nrow(cv_params)){
  set.seed(111111)
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     nfold = 5, # Use 5 fold cross-validation
                     eta = 0.1, # Set learning rate
                     max.depth = bst_max_depth, # Set max depth
                     min_child_weight = bst_min_child, #Set min child depth
                     gamma = bst_gamma, # Set minimum loss reduction for split
                     subsample = bst_sub_samp, #prop of training data used in tree
                     colsample_bytree = bst_col_samp, 
                     #num of variables to use in each tree
                     
                     reg_lambda = cv_params$lambda[i], 
                     reg_alpha = cv_params$alpha[i],
 
                     nrounds = 1000, # Set number of rounds
                     early_stopping_rounds = 20, #rounds to stop at if no improvement
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
  ) # Set evaluation metric to use
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
}

# visualise tuning sample params

res_db <- cbind.data.frame(cv_params, rmse_vec)
names(res_db)[3] <- c("rmse") 
res_db$lambda <- as.factor(res_db$lambda) # Convert tree number to factor for plotting
res_db$alpha <- as.factor(res_db$alpha) # Convert node size to factor for plotting

g_5 <- ggplot(res_db, aes(y = lambda, x = alpha, fill = rmse)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = nd_green, # Choose low color
                       mid = "white", # Choose mid color
                       high = nd_navy, # Choose high color
                       midpoint =mean(res_db$rmse), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Alpha", y = "Lambda", fill = "RMSE") # Set labels
g_5 # Generate plot

bst_lambda <- res_db[res_db$rmse == min(res_db$rmse), ]$lambda
bst_alpha <- res_db[res_db$rmse == min(res_db$rmse), ]$alpha

# Use xgb.cv to run cross-validation inside xgboost

etas_vec <- c(.3, .1, .05, .01, .005)

set.seed(111111)
bst_mod_1 <- xgb.cv(data = dtrain, # Set training data
                    nfold = 5, # Use 5 fold cross-validation
                    eta = etas_vec[1], # Set learning rate
                    max.depth = bst_max_depth, # use best max depth
                    min_child_weight = bst_min_child, 
                    gamma = bst_gamma, # use best gamma
                    subsample = bst_sub_samp, # use best subsample
                    colsample_bytree =  bst_col_samp, # use best col sample
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20,
                    reg_lambda = bst_lambda,
                    reg_alpha=bst_alpha,
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20) 

set.seed(111111)
bst_mod_2 <- xgb.cv(data = dtrain, # Set training data
                    nfold = 5, # Use 5 fold cross-validation
                    eta = etas_vec[2], # Set learning rate
                    max.depth = bst_max_depth, # use best max depth
                    min_child_weight = bst_min_child, 
                    gamma = bst_gamma, # use best gamma
                    subsample = bst_sub_samp, # use best subsample
                    colsample_bytree =  bst_col_samp, # use best col sample
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, 
                    reg_lambda = bst_lambda,
                    reg_alpha=bst_alpha,
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20) 

set.seed(111111)
bst_mod_3 <- xgb.cv(data = dtrain, # Set training data
                    nfold = 5, # Use 5 fold cross-validation
                    eta = etas_vec[3], # Set learning rate
                    max.depth = bst_max_depth, # use best max depth
                    min_child_weight = bst_min_child, 
                    gamma = bst_gamma, # use best gamma
                    subsample = bst_sub_samp, # use best subsample
                    colsample_bytree =  bst_col_samp, # use best col sample
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, 
                    reg_lambda = bst_lambda,
                    reg_alpha=bst_alpha,
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20) 

set.seed(111111)
bst_mod_4 <- xgb.cv(data = dtrain, # Set training data
                    nfold = 5, # Use 5 fold cross-validation
                    eta = etas_vec[4], # Set learning rate
                    max.depth = bst_max_depth, # use best max depth
                    min_child_weight = bst_min_child, 
                    gamma = bst_gamma, # use best gamma
                    subsample = bst_sub_samp, # use best subsample
                    colsample_bytree =  bst_col_samp, # use best col sample
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, 
                    reg_lambda = bst_lambda,
                    reg_alpha=bst_alpha,
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20) 

set.seed(111111)
bst_mod_5 <- xgb.cv(data = dtrain, # Set training data
                    nfold = 5, # Use 5 fold cross-validation
                    eta = etas_vec[5], # Set learning rate
                    max.depth = bst_max_depth, # use best max depth
                    min_child_weight = bst_min_child, 
                    gamma = bst_gamma, # use best gamma
                    subsample = bst_sub_samp, # use best subsample
                    colsample_bytree =  bst_col_samp, # use best col sample
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, 
                    reg_lambda = bst_lambda,
                    reg_alpha=bst_alpha,
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20) 


# eta plots

# Extract results for model with eta = 0.3
pd1 <- cbind.data.frame(bst_mod_1$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.3, nrow(bst_mod_1$evaluation_log)))
names(pd1)[3] <- "eta"
# Extract results for model with eta = 0.1
pd2 <- cbind.data.frame(bst_mod_2$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.1, nrow(bst_mod_2$evaluation_log)))
names(pd2)[3] <- "eta"
# Extract results for model with eta = 0.05
pd3 <- cbind.data.frame(bst_mod_3$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.05, nrow(bst_mod_3$evaluation_log)))
names(pd3)[3] <- "eta"
# Extract results for model with eta = 0.01
pd4 <- cbind.data.frame(bst_mod_4$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.01, nrow(bst_mod_4$evaluation_log)))
names(pd4)[3] <- "eta"
# Extract results for model with eta = 0.005
pd5 <- cbind.data.frame(bst_mod_5$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.005, nrow(bst_mod_5$evaluation_log)))
names(pd5)[3] <- "eta"
# Join datasets
plot_data <- rbind.data.frame(pd1, pd2, pd3, pd4, pd5)
# Converty ETA to factor
plot_data$eta <- as.factor(plot_data$eta)
# Plot points
g_6 <- ggplot(plot_data, aes(x = iter, y = test_rmse_mean, color = eta))+
  geom_point(alpha = 0.5) +
  theme_bw() + # Set theme
  theme(panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Number of Trees", title = "RMSE v Number of Trees",
       y = "RMSE", color = "Learning \n Rate")  # Set labels
g_6

# Plot lines
g_7 <- ggplot(plot_data, aes(x = iter, y = test_rmse_mean, color = eta))+
  geom_smooth(alpha = 0.5) +
  theme_bw() + # Set theme
  theme(panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Number of Trees", title = "RMSE v Number of Trees",
       y = "RMSE", color = "Learning \n Rate")  # Set labels
g_7


# fit final xgb model
bst_eta <- .1

set.seed(111111)
bst_final_mod <- xgboost(
    data = as.matrix(train_data[, -1]),  # Exclude the response variable
    label = train_data$high_speed_distance,
    booster = "gblinear",  # Use linear booster for regression
    objective = "reg:linear",  # Specify regression as the objective
    eval_metric = "rmse",  # Evaluation metric (Root Mean Squared Error)
    nfold = 5, # Use 5 fold cross-validation
    eta = bst_eta, # Set learning rate
    max.depth = bst_max_depth, # use best max depth
    min_child_weight = bst_min_child, 
    gamma = bst_gamma, # use best gamma
    subsample = bst_sub_samp, # use best subsample
    colsample_bytree =  bst_col_samp, # use best col sample
    nrounds = 2000, # Set number of rounds
    early_stopping_rounds = 20, 
    verbose = 1, # 1 - Prints out fit
    nthread = 1, # Set number of parallel threads
    reg_lambda = bst_lambda,
    reg_alpha=bst_alpha,
    print_every_n = 20)
  
# Make predictions on the test data
bst_final_preds <- predict(bst_final_mod, as.matrix(test_data[, -1]))
  
bst_final_actual<- test_data$high_speed_distance
  
# Calculate RMSE (Root Mean Squared Error) for model evaluation
rmse <- sqrt(mean((bst_final_preds - test_data$high_speed_distance)^2))
#You can also inspect the model's feature importance if needed
importance <- xgb.importance(feature_names = colnames(as.matrix(train_data[, -1])),
                             model = bst_final_mod)

#preds1 <- predict(bst_1, dtest)
bst_pred_data <- cbind.data.frame(bst_final_preds, bst_final_actual)
  
names(bst_pred_data) <- c("predicted", "actual")
bst_pred_data$difference <- bst_pred_data$actual - bst_pred_data$predicted

bst_pred_data$time <- "post-tune"


imp_mat <- xgb.importance(model = bst_final_mod) # Extract importance
xgb.plot.importance(imp_mat, top_n = 5, main = "Top 5 XGBoost Important Variables") # Plot importance (top 10 variables)

xgb_imp_df <- data.frame(imp_mat) %>% 
  select(all_of(c("Feature", "Importance")))

imp_viz_post_tune <- ggplot(data=xgb_imp_df[1:5,], aes(x=Importance, 
                                  y=reorder(Feature, -Importance, decreasing = TRUE))) + 
  theme_minimal() + geom_col(fill=nd_navy) +
  labs(title = "XG Boost Model Top 5 Most Important Variables", 
       x = "Importance", y = "Variable") 

soc1_full_preds_df <- rbind(bst_split_pred_data_pre_tune, bst_pred_data)

ggplot(soc1_full_preds_df, aes(x = difference,
                               y = after_stat(count / sum(count)),
                               fill = time)) +
  geom_histogram(data = subset(soc1_full_preds_df,
                               time == "pre-tune"),
                 alpha = 0.7, position = "identity", bins = 30) +
  geom_histogram(data = subset(soc1_full_preds_df,
                               time == "post-tune"),
                 alpha = 0.5, position = "identity", bins = 30) +
  scale_fill_manual(values = c("post-tune" = nd_green, "pre-tune" = nd_navy)) + 
  xlim(-750, 500) +
  ylim(0, .15) +
  labs(title = "Histogram of Difference between Prediction vs Actual",
       subtitle = "Distribution of Prediction Differences Pre & Post Tune",
       x = "Difference between Actual and Prediction",
       y = "Percent Frequency") +
  theme_minimal()


# soc1 Preds DATA VIZ
post_tune_mod_pred_viz <- ggplot(bst_pred_data, aes(x = predicted, y = actual)) +
  geom_point(color=ifelse(abs(bst_pred_data$difference) >= 100, nd_navy, nd_green)) +
  geom_smooth(color=nd_gold) + theme_minimal() + ylim(0,1000) + xlim(0,1000)+
  labs(title = xgb_viz_title, 
    subtitle = paste(selected_sport, "Data;",
                     "High Speed Distance between:", 
                     low_lim, "&", up_lim, sep=" "))



# soc1 Preds - ADJUSTED AXES
post_tune_mod_pred_viz_2 <- ggplot(bst_pred_data, aes(x = predicted, y = actual)) +
  geom_point(color=ifelse(abs(bst_pred_data$difference) >= 100, nd_navy, nd_green)) +
  geom_smooth(color=nd_gold) + theme_minimal() + ylim(0,750) + xlim(0,750)+
  labs(title = xgb_viz_title, 
    subtitle = paste(selected_sport, "Data;",
                     "High Speed Distance between:", 
                     low_lim, "&", up_lim, sep=" "))




# perc_diff_over_100 <- sum(abs(bst_pred_data$difference) >= 100)/length(bst_pred_data$difference)
# 
# perc_diff_over_150 <- sum(abs(bst_pred_data$difference) >= 150)/length(bst_pred_data$difference)
# 
# yards_to_comp <- c(50, 100, 150, 200)
# results_vec <- c()
# 
# for (i in seq_along(yards_to_comp)){
#   within_dist <- sum(abs(bst_pred_data$difference) <= i)/length(bst_pred_data$difference)
#   results_vec <- c(results_vec, within_dist)
# }
# 
# results_vec

summary(bst_pred_data$difference)

quantile(bst_pred_data$difference, .10)
hist(bst_pred_data$difference, freq=FALSE, breaks = 50, xlim = c(-750, 750))

viz_time <- "post-tuning"

post_tune_diff_hist <- ggplot(bst_pred_data,
       aes(x = difference, #fill = color_condition_wt,
           y = after_stat(count / sum(count)))) +
  geom_histogram(fill = nd_green, alpha =.5) + xlim(-750,500) + ylim(0, .25)+
  labs(title = "Histogram of Difference between Prediction vs Actual",
       subtitle= paste0("High Speed Distance ", viz_time),
       x = "Difference between Prediction and Actual",
       y = "Percent Frequency") +
  theme_minimal() 


# hist_comp <- ggarrange(pre_tune_diff_hist + labs(x="", y="Perc Freq", 
#                                     title = "Pre-Tuning", 
#                                     subtitle=""),
#           post_tune_diff_hist + labs(title="Post-Tuning", 
#                                      subtitle = "", 
#                                      x="", y="Perc Freq"),
#           ncol = 1, 
#           nrow = 2)
# annotate_figure(hist_comp, top = text_grob("Difference between Predicted and Actual Value", 
#                color = nd_navy, face = "bold", size = 14))


# ABSOLUTE VALUE HISTOGRAM
abs_val_hist_post_tune <- ggplot(bst_pred_data,
       aes(x = abs(difference), #fill = color_condition_wt,
           y = after_stat(count / sum(count)))) +
  geom_histogram(fill = nd_green, binwidth = 25) + xlim(0, 500)+
  labs(title = "Histogram of Difference between Prediction vs Actual",
       subtitle= paste0("High Speed Distance ", viz_time),
       x = "Difference between Prediction and Actual",
       y = "Percent Frequency") +
  theme_minimal() 

sorted_diffs <- sort(abs(bst_pred_data$difference))

preds_75perc_val <- quantile(sorted_diffs)[4] #180.39

