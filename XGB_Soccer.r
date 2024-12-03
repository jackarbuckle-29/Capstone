# MSBA-SA CAPSTONE 
# SPRING 2024
# MODELING HIGH SPEED RUNNING WITH CATAPULT DATA

# XGBoost Model Tuning - Soccer
# warning: code takes very long time to run given the tuning parameters. 

############################# SET UP ############################# 
library(dplyr)
library(ggplot2)
library(xgboost)
library(stringr)
library(readr)
library(Metrics)

# setting ND colors 
nd_navy <- "#0C2340"
nd_green <- "#00843D"
nd_gold <- "#C99700"

comb_z_df_mod <- read.csv("comb_soc_data_cleaned_with_z.csv")

############################# MODELING PREP ############################# 
model_df <- comb_z_df_mod %>% 
  select(!contains(c("unique_session", "name_id", "high_speed_distance_z",
                     "team_gender_z", 
                     "activity_type_z", "date_z", "location_z"))) %>%
  select(is.numeric) %>%
  select(all_of("high_speed_distance"), everything())

set.seed(33) # Set Seed
split_ratio <- 0.7
train_row_ind <- sample(x=nrow(model_df), 
                        size=floor(nrow(model_df) * split_ratio))

# Split the data into training and testing sets
train_data <- model_df[train_row_ind, ]
test_data <- model_df[-train_row_ind, ]

train_response = train_data[,1]
train_x = train_data[,-1]
test_response = test_data[,1]
test_x = test_data[,-1]

response <- model_df$high_speed_distance

# Create training data XGBOOST
dtrain <- xgb.DMatrix(data = as.matrix(train_data[ ,-1]), label = train_response)
# Create test data XGBOOST
dtest <- xgb.DMatrix(data = as.matrix(test_data[ ,-1]),label = test_response)

############################# XG BOOST PRELIMINARY MODELING ############################# 
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
prelim_importance <- xgb.importance(feature_names = colnames(as.matrix(train_data[, -1])),
                             model = bst_split_mod_pre_tune)

#preds1 <- predict(bst_1, dtest)
bst_split_pred_data_pre_tune <- cbind.data.frame(bst_preds_pre_tune, bst_actual_pre_tune)
names(bst_split_pred_data_pre_tune) <- c("predicted", "actual")

# calc difference as actual-predicted
bst_split_pred_data_pre_tune$difference <- bst_split_pred_data_pre_tune$actual - bst_split_pred_data_pre_tune$predicted 
bst_split_pred_data_pre_tune$time <- "pre-tune"

pre_tune_diff_summary_soc1 <- summary(bst_split_pred_data_pre_tune$difference)

# DATA VIZ
pred_viz_pre_tune <- ggplot(bst_split_pred_data_pre_tune, aes(x = predicted, y = actual)) +
  geom_point(color=nd_navy) +
  geom_smooth(color=nd_gold) + theme_minimal() +
  labs(title = xgb_viz_title, 
    subtitle = paste(selected_sport, "Data;",
                     "High Speed Distance:", sep=" "))

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

save(prelim_importance, bst_split_pred_data_pre_tune, 
     pred_viz_pre_tune, diff_hist_pre_tune, pre_tune_diff_summary_soc1, 
     file = "pre_tune_files.rda")



############################# MAX DEPTH MIN CHILD ############################# 
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

save(res_db, bst_max_depth, bst_min_child, g_2, 
     file = "min_child_max_depth.rda")



############################# GAMMA ############################# 
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

save(gam_df, bst_gamma, 
     file = "gamma.rda")



############################# SAMPLE & COLSAMPLE ############################# 
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

save(res_db, bst_col_samp, bst_sub_samp, g_4, 
     file = "sample_colsample.rda")



############################# ALPHA & LAMBDA ############################# 
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

save(res_db, bst_lambda, bst_alpha, g_5, 
     file = "alpha_lambda.rda")






############################# ETA ############################# 
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

save(bst_mod_1, bst_mod_2, bst_mod_3, bst_mod_4, bst_mod_5,
     g_6, g_7, plot_data, 
     file = "eta.rda")


############################# TUNED MODEL ############################# 
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
tuned_importance <- xgb.importance(feature_names = colnames(as.matrix(train_data[, -1])),
                             model = bst_final_mod)

#preds1 <- predict(bst_1, dtest)
bst_pred_data <- cbind.data.frame(bst_final_preds, bst_final_actual)
names(bst_pred_data) <- c("predicted", "actual")
bst_pred_data$difference <- bst_pred_data$actual - bst_pred_data$predicted
bst_pred_data$time <- "post-tune"

imp_mat <- xgb.importance(model = bst_final_mod) # Extract importance
imp_plot <- xgb.plot.importance(imp_mat, top_n = 5, main = "Top 5 XGBoost Important Variables") # Plot importance (top 10 variables)

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
                     "High Speed Distance"))

# soc1 Preds - ADJUSTED AXES
post_tune_mod_pred_viz_2 <- ggplot(bst_pred_data, aes(x = predicted, y = actual)) +
  geom_point(color=ifelse(abs(bst_pred_data$difference) >= 100, nd_navy, nd_green)) +
  geom_smooth(color=nd_gold) + theme_minimal() + ylim(0,750) + xlim(0,750)+
  labs(title = xgb_viz_title, 
    subtitle = paste(selected_sport, "Data;",
                     "High Speed Distance")) 

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

save(bst_final_mod, bst_pred_data, tuned_importance, imp_plot, xgb_imp_df,
     soc1_full_preds_df, post_tune_mod_pred_viz, post_tune_mod_pred_viz_2,post_tune_diff_hist,
     file = "bst_final_mod.rda")
