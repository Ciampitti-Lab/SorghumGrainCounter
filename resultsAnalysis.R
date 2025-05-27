# Importing libraries ------
library(tidyverse)
library(metrica)
library(caret)
library(ggplot2)

# Density maps -----
## Loading data -----
modelAnalised <- "CerealNet_VGG16"
data <- read.table(paste0("results/", modelAnalised, ".csv", sep = ""))

## Computing metrics -----

### RMSE -----
rmse <- metrica::RMSE(data, data$Observed, data$Predicted)
print(paste("RMSE:", rmse))

### MAE ------
mae <- metrica::MAE(data, data$Observed, data$Predicted)
print(paste("MAE:", mae))

### Kling-Gupta Coefficient -----
kge <- metrica::KGE(data, data$Observed, data$Predicted)
print(paste("KGE:", kge))

## 1:1 Graph ------
ggplot(data, aes(x = Predicted, y = Observed)) +
  geom_point(colour = "red", size = 3) +
  geom_abline(intercept = 0, slope = 1, linewidth = 2) +
  theme_minimal() +
  labs(
    x = 'Predicted (number of grains)',
    y = 'Observed (number of grains)',
    title = 'Predicted vs. Actual number of grains')

# Estimating to the whole panicle ------

## Loading data ------

modelAnalised <- "CerealNet_VGG16"
data <- read.csv(paste0("results/Fit_to_observed_", modelAnalised, ".csv", sep=""))
colnames(data)[3] <- "Observed"

## Comparing both sides -----
data$Picture1 <- data$Predicted[seq(1, nrow(data), by = 2)]
data$Picture2 <- data$Predicted[seq(2, nrow(data), by = 2)]

### Metrics -----
rmse <- metrica::RMSE(data, data$Picture1, data$Picture2)
print(paste("RMSE:", rmse))

mape <- metrica::MAPE(data, data$Picture1, data$Picture2)
print(paste("MAPE:", mape))

mae <- metrica::MAE(data, data$Picture1, data$Picture2)
print(paste("MAE:", mae))

## Loading and observing output from the density maps vs whole panicle counting -----
ggplot(data, aes(x = Predicted, y = Observed)) +
  geom_point(colour = "red", size = 4) +
  theme_minimal() +
  labs(x='Predicted (number of grains from the density maps)', y='Observed (number of grains)', title='Predicted vs. Actual number of grains')

### Splitting the data -----
set.seed(666)
trainIndex <- sample(seq_len(nrow(data)), 0.7 * nrow(data))
DM_train <- predicted[trainIndex]
DM_test <- predicted[-trainIndex]
obs_train <- observed[trainIndex]
obs_test <- observed[-trainIndex]

### Ratio approach ------
factor <- mean(obs_train / DM_train)
DM_pred <- DM_test * factor

### Linear Regression --------
model <- lm(obs_train ~ DM_train)

test_data <- data.frame(DM_train = DM_test)
DM_pred <- predict(model, newdata = test_data)
DM_pred <- as.integer(DM_pred)

### Metrics -----------
rmse <- metrica::RMSE(obs = obs_test, pred = DM_pred)
print(paste("RMSE:", rmse))

mape <- metrica::MAPE(obs = obs_test, pred = DM_pred)
print(paste("MAPE:", mape))

mae <- metrica::MAE(obs = obs_test, pred = DM_pred)
print(paste("MAE:", mae))

kge <- metrica::KGE(obs = obs_test, pred = DM_pred)
print(paste("KGE:", kge))

### Plotting the scatter plot -------
ggplot() +
  geom_point(aes(x = predicted, y = observed), size = 3, colour = "red") +
  geom_abline(intercept = 0, slope = 1, linewidth = 3) +
  labs(title = "Observed vs Predicted counting of sorghum kernels",
       x = "Density map estimated sorghum grain number (grains)",
       y = "Manually counted sorghum grain number (grains)") +
  theme_minimal()
