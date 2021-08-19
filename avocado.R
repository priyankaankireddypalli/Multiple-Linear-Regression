# 4
library(readr)
# Importing the dataset
avocado <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Multiple Linear Regression\\Avacado_Price.csv')
avocado <-avocado[,c(1:6)]
View(avocado)
str(avocado)
attach(avocado)
# Performing EDA
# Normal distribution
qqnorm(AveragePrice)
qqline(AveragePrice)
summary(avocado)
# Scatter plot
plot(Total_Volume, AveragePrice) # Plot relation ships between each X with Y
plot(tot_ava1, AveragePrice)
plot(tot_ava2,AveragePrice)
plot(tot_ava3, AveragePrice)
plot(Total_Bags,AveragePrice)
# Or make a combined plot
pairs(avocado)   # Scatter plot for all pairs of variables
plot(avocado)
cor(Total_Volume, AveragePrice)
cor(tot_ava1, AveragePrice)
cor(tot_ava2,AveragePrice)
cor(tot_ava3, AveragePrice)
cor(Total_Bags,AveragePrice)
cor(avocado)
# The Linear Model of interest
model.avocado <- lm(AveragePrice ~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags , data = avocado) # lm(Y ~ X)
summary(model.avocado)
model.avocadoT <- lm( AveragePrice~ Total_Volume)
summary(model.avocadoT)
model.avocado1 <- lm(AveragePrice ~ tot_ava1)
summary(model.avocado1)
model.avocado2 <-lm(AveragePrice~tot_ava2)
summary(model.avocado2)
model.avocadoc3 <- lm( AveragePrice~ tot_ava3)
summary(model.avocadoc3)
model.avocado4 <- lm(AveragePrice ~ Total_Bags)
summary(model.avocado4)
model.avocadoALL <- lm(AveragePrice ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags )
summary(model.avocadoALL)
# Scatter plot matrix with Correlations inserted in graph
library(GGally)
ggpairs(avocado)
# Partial Correlation matrix
library(corpcor)
library(car)
cor(avocado)
cor2pcor(cor(avocado))
plot(model.avocado)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance
qqPlot(model.avocado, id.n = 5) # QQ plots of studentized residuals, helps identify outliers
# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.avocado, id.n = 3) # Index Plots of the influence measures
influencePlot(model.avocado, id.n = 3) # A user friendly representation of the above
# Regression after deleting the 77th observation
model.avocado1 <- lm(AveragePrice ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags, data = avocado[-15561, ])
summary(model.avocado1)
# Variance Inflation Factors
vif(model.avocado)  # VIF is > 10 => collinearity
# Added Variable Plots 
avPlots(model.avocado, id.n = 2, id.cex = 0.8, col = "red")
# Linear Model without Total_Volume
model.final <- lm(AveragePrice ~tot_ava1+tot_ava2+tot_ava3+Total_Bags , data = avocado)
summary(model.final)
# Linear model without Total bags and influential observation
model.final1 <- lm(AveragePrice ~Total_Volume+tot_ava1+tot_ava2+tot_ava3 , data = avocado[-810, ])
summary(model.final1)
# Added Variable Plots
avPlots(model.final1, id.n = 2, id.cex = 0.8, col = "red")
# Variance Influence Plot
vif(model.final1)
# Evaluation Model Assumptions
plot(model.final1)
plot(model.final1$fitted.values, model.final1$residuals)
qqnorm(model.final1$residuals)
qqline(model.final1$residuals)
# Subset selection
library(leaps)
lm_best <- regsubsets(AveragePrice ~ ., data = avocado, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 5)
lm_forward <- regsubsets(AveragePrice ~ ., data = avocado, nvmax = 15, method = "forward")
summary(lm_forward)
# Data Partitioning
n <- nrow(avocado)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- avocado[-train, ]
# Model Training
model <- lm(AveragePrice ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags, avocado[train, ])
summary(model)
pred <- predict(model, newdata = test)
actual <- test$AveragePrice
error <- actual - pred
test.rmse <- sqrt(mean(error**2))
test.rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse
# Step AIC
library(MASS)
stepAIC(model.avocado)

