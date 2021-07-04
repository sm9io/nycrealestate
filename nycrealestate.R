# loading libraries and setting significant digits to 2
library(dplyr)
library(rpart)
library(randomForest)
library(corrplot)
library(stringr)
options(digits = 2)

# loading data into the environment and saving in Rda format for future reference
nyc_rolling_sales <- read.csv("nyc-rolling-sales.csv")
save(nyc_rolling_sales, file = "nyc_rolling_sales.Rda")

# exploring the dataset and cleaning the dataset
str(nyc_rolling_sales)
dim(nyc_rolling_sales)
names(nyc_rolling_sales) <- tolower(str_replace_all(names(nyc_rolling_sales), "\\.", "_"))
names(nyc_rolling_sales)

# changing certain variables classes
nyc_rolling_sales <- nyc_rolling_sales %>%
  mutate(x = NULL,
         borough = as.factor(borough),
         neighborhood = as.factor(neighborhood),
         building_class_category = as.factor(building_class_category),
         tax_class_at_present = as.factor(tax_class_at_present),
         ease_ment = NULL,
         building_class_at_present = as.factor(building_class_at_present),
         address = NULL,
         apartment_number = NULL,
         zip_code = as.integer(zip_code),
         land_square_feet = as.integer(land_square_feet),
         gross_square_feet = as.integer(gross_square_feet),
         tax_class_at_time_of_sale = as.factor(tax_class_at_time_of_sale),
         building_class_at_time_of_sale = as.factor(building_class_at_time_of_sale),
         sale_price = as.integer(sale_price),
         sale_date = as.Date(sale_date))

# looking at the correlations for numberic valiables
nyc_rolling_sales %>% select_if(is.numeric) %>%
  cor(use = "pairwise.complete.obs") %>% corrplot(method = "number")

# NAs and odd observations
# note that if you directly go for filtering based on values, NAs will be removed
nyc_rolling_sales_new <- nyc_rolling_sales %>%
  filter(!is.na(land_square_feet) & !is.na(gross_square_feet) & ! is.na(sale_price))
nyc_rolling_sales_new %>% select_if(is.numeric) %>%
  cor(use = "pairwise.complete.obs") %>% corrplot(method = "number")
nyc_rolling_sales_new <- nyc_rolling_sales_new %>%
  filter(sale_price > 100 & land_square_feet > 100 & gross_square_feet > 100
         & zip_code > 0)

nyc_rolling_sales_new %>%
  mutate(age = as.integer(format(sale_date,"%Y")) - year_built,
         sale_date = as.integer(format(sale_date,"%Y"))) %>%
  select_if(is.numeric) %>%
  cor(use = "pairwise.complete.obs") %>% corrplot(method = "number")
plot(nyc_rolling_sales_new$zip_code, nyc_rolling_sales_new$sale_price,
     xlab = "Zip code", ylab = "Selling price")
plot(nyc_rolling_sales_new$year_built, nyc_rolling_sales_new$sale_price,
     xlab = "Year built", ylab = "Selling price")

# selecting variables as needed
nyc_rolling_sales_new <- nyc_rolling_sales %>%
  filter(sale_price > 100 & gross_square_feet > 100
         & zip_code > 0) %>%
  mutate(sale_price = sale_price / 10^6,
         lot = NULL,
         block = NULL,
         zip_code = as.factor(zip_code),
         residential_units = NULL,
         commercial_units = NULL,
         total_units = NULL,
         land_square_feet = NULL,
         year_built = as.factor(year_built),
         sale_date = NULL)
dim(nyc_rolling_sales_new)

# creating test and train sets
set.seed(1, sample.kind = "Rounding")
test_index <- caret::createDataPartition(y = nyc_rolling_sales_new$sale_price, times = 1, p = 0.2, list = FALSE)
train <- nyc_rolling_sales_new[-test_index,]

temp <- nyc_rolling_sales_new[test_index,]
test <- temp %>% 
  semi_join(train, by = "borough") %>%
  semi_join(train, by = "neighborhood") %>%
  semi_join(train, by = "building_class_category") %>%
  semi_join(train, by = "tax_class_at_present") %>%
  semi_join(train, by = "building_class_at_present") %>%
  semi_join(train, by = "zip_code") %>%
  semi_join(train, by = "year_built") %>%
  semi_join(train, by = "tax_class_at_time_of_sale") %>%
  semi_join(train, by = "building_class_at_time_of_sale")

removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(test_index, temp, removed)

# modelling with lm, rpart and rf
lmfit <- lm(sale_price~., data = train)
rpartfit <- rpart(sale_price~., data = train)
# rf cannot handle categorical predictors with more than 53 categories
# making some adjustments for this
nyc_rolling_sales_rf <- nyc_rolling_sales_new %>%
  mutate(neighborhood = NULL,
         zip_code = as.integer(zip_code),
         year_built = as.integer(year_built),
         building_class_at_present = NULL,
         building_class_at_time_of_sale = NULL,
         borough = NULL,
         tax_class_at_present = NULL) %>%
  filter(zip_code != 0)

set.seed(1, sample.kind = "Rounding")
test_index <- caret::createDataPartition(y = nyc_rolling_sales_rf$sale_price, times = 1, p = 0.2, list = FALSE)
train_rf <- nyc_rolling_sales_rf[-test_index,]
temp <- nyc_rolling_sales_rf[test_index,]

test_rf <- temp %>% 
  semi_join(train_rf, by = "building_class_category") %>%
  semi_join(train_rf, by = "tax_class_at_time_of_sale")

removed <- anti_join(temp, test_rf)
train_rf <- rbind(train_rf, removed)

rm(test_index, temp, removed)

rffit <- randomForest(sale_price~., data = train_rf)

# r2 for lm
summary(lmfit)$r.squared
summary(lmfit)$adj.r.squared

# checking RMSEs of the 3 models
rmse <- function(pred, act){
  if(length(pred) == length(act)){
    sqrt(sum((pred - act)^2)/length(act))} else{
    print("Lengths of vectors do not match")
  }
}

rmse_lm <- rmse(predict(lmfit, test),test$sale_price)
rmse_rpart<- rmse(predict(rpartfit, test),test$sale_price)
rmse_rf <- rmse(predict(rffit, test_rf),test_rf$sale_price)

rmse_all <- tibble(RMSE = c(rmse_lm, rmse_rpart, rmse_rf))
rmse_all <- tibble(cbind(data.frame(model = c("lm", "rpart", "rf")),
                         rmse_all,
                         data.frame(RMSE_by_SD = c(
                                        rmse_all$RMSE[1]/sd(test$sale_price),
                                        rmse_all$RMSE[2]/sd(test$sale_price),
                                        rmse_all$RMSE[3]/sd(test_rf$sale_price)))))
rmse_all
plot(rffit)

# THANK YOU