# 2
library(readr)
# Importing the dataset
computer <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\Multiple Linear Regression\\Computer_Data.csv')
computer <-computer[,c(2,3,4,5)]
View(computer)
str(computer)
attach(computer)
# Performing EDA
# Normal distribution
qqnorm(price)
qqline(price)
summary(computer)
# Scatter plot
plot(speed, price) # Plot relation ships between each X with Y
plot(hd, price)
plot(ram,price)
# Or make a combined plot
pairs(computer)   # Scatter plot for all pairs of variables
plot(computer)
cor(speed, price)
cor(hd,price)
cor(ram,price)
cor(computer) # correlation matrix
# The Linear Model of interest
model.comp <- lm(price ~ hd+ram+speed, data = computer) # lm(Y ~ X)
summary(model.comp)
model.starH <- lm( price~ hd)
summary(model.starH)
model.starR <- lm(price ~ ram)
summary(model.starR)
model.starS <-lm(price~speed)
summary(model.starS)
model.starHRS <- lm(price ~hd+ram+speed )
summary(model.starHRS)
# Scatter plot matrix with Correlations inserted in graph
library(GGally)
ggpairs(computer)
# Partial Correlation matrix
library(corpcor)
library(car)
cor(computer)
cor2pcor(cor(computer))
plot(model.comp)# Residual Plots, QQ-Plot, Std. Residuals vs Fitted, Cook's distance
qqPlot(model.comp, id.n = 5) # QQ plots of studentized residuals, helps identify outliers
# Deletion Diagnostics for identifying influential obseravations
influenceIndexPlot(model.comp, id.n = 3) # Index Plots of the influence measures
influencePlot(model.comp, id.n = 3) # A user friendly representation of the above
# Regression after deleting the 77th observation
model.comp1 <- lm(price ~ hd+ram+speed , data = computer[-901, ])
summary(model.comp1)
# Diagnostic Plots
library(car)
# Variance Inflation Factors
vif(model.comp)  # VIF is > 10 => collinearity
# Regression model to check R^2 on Independent variales
VIFHD <- lm(hd ~ ram+speed)
VIFRAM <- lm(ram ~ hd+speed)
VIFSP<- lm(speed~hd+ram)
summary(VIFHD)
summary(VIFRAM)
summary(VIFSP)
# Added Variable Plots 
avPlots(model.comp, id.n = 2, id.cex = 0.8, col = "red")
# Linear Model without SPEED
model.final <- lm(price ~ hd+ram, data = computer)
summary(model.final)
# Linear model without SPEED and influential observation
model.final1 <- lm(price ~hd+ram , data = computer[-901, ])
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
lm_best <- regsubsets(price ~ ., data = computer, nvmax = 15)
summary(lm_best)
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)
coef(lm_best, 3)
lm_forward <- regsubsets(price ~ ., data = computer, nvmax = 15, method = "forward")
summary(lm_forward)
# Data Partitioning
n <- nrow(computer)
n1 <- n * 0.7
n2 <- n - n1
train <- sample(1:n, n1)
test <- computer[-train, ]
# Model Training
model <- lm(price ~ hd+speed+ram, computer[train, ])
summary(model)
pred <- predict(model, newdata = test)
actual <- test$price
error <- actual - pred
test.rmse <- sqrt(mean(error**2))
test.rmse
train.rmse <- sqrt(mean(model$residuals**2))
train.rmse
# Step AIC
library(MASS)
stepAIC(model.comp)
