if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(fastDummies)) install.packages("fastDummies")
if(!require(lubridate)) install.packages("lubridate")
if(!require(stringi)) install.packages("stringi")
if(!require(OneR)) install.packages("OneR")
library(tidyverse)
library(caret)
library(data.table)
library(fastDummies)
library(lubridate)
library(stringi)
library(OneR)



############################
# Creating MovieLens Dataset
############################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#creating date, year, day of week variables (NEW VARIABLES)
movielens <- movielens %>% mutate(date = as_datetime(timestamp), yearstamp = year(as_datetime(timestamp)), daystamp = wday(date))

#creating year of release variable (NEW VARIABLE)
extract <- function(x) {
  stri_extract_last(x, regex = "\\d{4}")
}
movielens <- movielens %>% mutate(year_released = as.numeric(sapply(title, extract)))

#creating year difference variable (NEW VARIABLE)
movielens <- movielens %>% mutate(year_diff = yearstamp - year_released)

#creating number of ratings variable (NEW VARIABLE)
movielens <- movielens %>% group_by(movieId) %>% summarize(n_ratings = n()) %>% left_join(movielens, by = "movieId")

#creating number of months since first rating variable (NEW VARIABLE)
movielens <- movielens %>% mutate(timestamp_round = round_date(as_datetime(timestamp), unit = "month"))
movielens <- movielens %>% group_by(userId) %>% mutate(months_since_first = timestamp_round - min(timestamp_round)) 

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
temp <- movielens[test_index,]
edx <- movielens[-test_index,]


# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(list = setdiff(ls(), c("edx", "validation")))

################################
# Implementing model from course
################################

#defining function to calculate RMSE
calc_RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#generating train and test sets
set.seed(17, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
temp <- edx[test_index,]
edx_train <- edx[-test_index,]

edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)
rm(temp, removed)

#baseline "naive" RMSE
mu_hat <- mean(edx_train$rating)
naive_rmse <- calc_RMSE(edx_test$rating, mu_hat)
rmse_table <- data.frame(method = "Using the average", RMSE = naive_rmse)

#including movie and user effects
movie_avgs <- edx_train %>% group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))
user_avgs <- edx_train %>% left_join(movie_avgs, by = "movieId") %>% 
  group_by(userId) %>% summarize(b_u = mean(rating - mu_hat - b_i))
edx_train <- edx_train %>% left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId")
edx_test <- edx_test %>% left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% mutate(pred = mu_hat + b_i + b_u)
rmse_0 <- calc_RMSE(edx_test$pred, edx_test$rating)
rmse_table <- rbind(rmse_table, data.frame(method = "2 effects", RMSE = rmse_0))

####################
# Adding new effects
####################

#view year differences
edx_train %>% ggplot(aes(year_diff)) + geom_histogram()

#include year difference effect
year_avgs <- edx_train %>% group_by(year_diff) %>% 
  summarize(b_y = mean(rating - mu_hat - b_u - b_i))
edx_train <- edx_train %>% left_join(year_avgs, by = "year_diff")
edx_test <- edx_test %>% left_join(year_avgs, by = "year_diff") %>% 
  mutate(pred_1 = mu_hat + b_i + b_u + b_y)
rmse_1 <- calc_RMSE(edx_test$pred_1, edx_test$rating)
rmse_table <- rbind(rmse_table, data.frame(method = "3 effects", RMSE = rmse_1))


#view number of ratings per movie
edx_train %>% ggplot(aes(n_ratings)) + geom_histogram()

#include number of ratings per movie effect
n_rating_avgs <- edx_train %>% group_by(n_ratings) %>% 
  summarize(b_n = mean(rating - mu_hat - b_u - b_i - b_y))
edx_train <- edx_train %>% left_join(n_rating_avgs, by = "n_ratings")
edx_test <- edx_test %>% left_join(n_rating_avgs, by = "n_ratings") %>% 
  mutate(pred_2 = mu_hat + b_i + b_u + b_y + b_n)
rmse_2 <- calc_RMSE(edx_test$pred_2, edx_test$rating)
rmse_table <- rbind(rmse_table, data.frame(method = "4 effects", RMSE = rmse_2))


#view months since user's first rating
edx_train %>% 
  ggplot(aes(log(as.numeric(months_since_first)))) + 
  geom_histogram() + 
  xlab("log(months_since_first)")

#include months since first rating effect
since_first_avgs <- edx_train %>% group_by(months_since_first) %>% 
  summarize(b_m = mean(rating - mu_hat - b_u - b_i - b_y - b_n))
edx_train <- edx_train %>% left_join(since_first_avgs, by = "months_since_first")
edx_test <- edx_test %>% left_join(since_first_avgs, by = "months_since_first") %>% 
  mutate(pred_3 = mu_hat + b_i + b_u + b_y + b_n + b_m)
rmse_3 <- calc_RMSE(edx_test$pred_3, edx_test$rating)
rmse_table <- rbind(rmse_table, data.frame(method = "5 effects", RMSE = rmse_3))


#view crude genres
length(unique(edx_train$genres))

#include crude genre effect
genre_avgs <- edx_train %>% group_by(genres) %>% 
  summarize(b_g = mean(rating - mu_hat - b_u - b_i - b_y - b_n - b_m))
edx_train <- edx_train %>% left_join(genre_avgs, by = "genres")
edx_test <- edx_test %>% left_join(genre_avgs, by = "genres") %>% 
  mutate(pred_4 = mu_hat + b_i + b_u + b_y + b_n + b_m + b_g)
rmse_4 <- calc_RMSE(edx_test$pred_4, edx_test$rating)
rmse_table <- rbind(rmse_table, data.frame(method = "6 effects", RMSE = rmse_4))


#view day of week
t <- prop.table(table(edx_train$daystamp, edx_train$rating))
table(t)

#include day of week effect
day_avgs <- edx_train %>% group_by(daystamp) %>% 
  summarize(b_d = mean(rating - mu_hat - b_u - b_i - b_y - b_n - b_m))
edx_train <- edx_train %>% left_join(day_avgs, by = "daystamp")
edx_test <- edx_test %>% left_join(day_avgs, by = "daystamp")  %>% 
  mutate(pred_5 = mu_hat + b_i + b_u + b_y + b_n + b_m + b_g + b_d)
rmse_5 <- calc_RMSE(edx_test$pred_5, edx_test$rating)
rmse_table <- rbind(rmse_table, data.frame(method = "7 effects", RMSE = rmse_5))


#view b_i by number of ratings
movie_titles <- edx %>% select(movieId, title) %>% distinct()
movie_avgs %>% left_join(movie_titles, by = "movieId") %>% 
  arrange(desc(b_i)) %>% head(10)
movie_avgs %>% left_join(movie_titles, by = "movieId") %>% 
  arrange(b_i) %>% head(10)
edx_train %>% select(b_i, n_ratings) %>% distinct() %>% 
  ggplot(aes(n_ratings, b_i)) + geom_point() + geom_smooth(method = "lm", formula = y ~ x)

#regularize movie effect
lambdas <- seq(0,10,0.25)
rmse_6_seq <- sapply(lambdas, function(x) {
  movie_sums <- edx_train %>% group_by(movieId) %>% 
    summarize(s = sum(rating - mu_hat), n_i = n())
  predicted_ratings <- edx_test %>% left_join(movie_sums, by = "movieId") %>% 
    mutate(b_i_reg = s / (n_i + x)) %>% 
    mutate(pred_reg = mu_hat + b_i_reg + b_u + b_y + b_n + b_m + b_g + b_d) %>% ungroup()
  return(calc_RMSE(predicted_ratings$pred_reg, edx_test$rating))
})

#get optimal lambda and RMSE for regularized movie effect
qplot(lambdas, rmse_6_seq)
lambda_tune <- lambdas[which.min(rmse_6_seq)]
rmse_table <- rbind(rmse_table, data.frame(
  method = "7 effects, reg. movie effect", RMSE = min(rmse_6_seq)))

#add regularized b_i back into data sets
movie_sums <- edx_train %>% group_by(movieId) %>% 
  summarize(s = sum(rating - mu_hat), n_i = n())
edx_train <- edx_train %>% left_join(movie_sums, by = "movieId") %>% 
  mutate(b_i_reg = s / (n_i + lambda_tune)) %>% ungroup()
edx_test <- edx_test %>% left_join(movie_sums, by = "movieId") %>% 
  mutate(b_i_reg = s / (n_i + lambda_tune)) %>% 
  mutate(pred_reg = mu_hat + b_i_reg + b_u + b_y + b_n + b_m + b_g + b_d) %>% 
  ungroup()


#view b_u by number of users
users <- edx %>% select(userId) %>% distinct()
user_n <- edx %>% group_by(userId) %>% summarize(n_user = n(), userId = mean(userId))
user_avgs %>% inner_join(user_n, by = "userId") %>% 
  ggplot(aes(n_user, b_u)) + geom_point() + geom_smooth(method = "lm", formula = y ~ x)

#regularize user effect
rmse_7_seq <- sapply(lambdas, function(x) {
  user_sums <- edx_train %>% group_by(userId) %>% summarize(n_u = n())
  predicted_ratings <- edx_test %>% left_join(user_sums, by = "userId") %>% 
    mutate(b_u_reg = (b_u*n_u)/(n_u + x)) %>% 
    mutate(pred_reg_1 = mu_hat + b_i_reg + b_u_reg + b_y + b_n + b_m + b_g + b_d) %>% 
    ungroup()
  return(calc_RMSE(predicted_ratings$pred_reg_1, edx_test$rating))
})

#get optimal lambda and RMSE for regularized user effect
qplot(lambdas, rmse_7_seq)
lambda_tune_1 <- lambdas[which.min(rmse_7_seq)]
rmse_table <- rbind(rmse_table, data.frame(
  method = "7 effects, reg. movie/user effects", RMSE = min(rmse_7_seq)))

#add regularized b_u back into data sets
user_sums <- edx_train %>% group_by(userId) %>% summarize(n_u = n())
edx_train <- edx_train %>% left_join(user_sums, by = "userId") %>% 
  mutate(b_u_reg = (b_u*n_u)/(n_u + lambda_tune_1))
edx_test <- edx_test %>% left_join(user_sums, by = "userId") %>% 
  mutate(b_u_reg = (b_u*n_u)/(n_u + lambda_tune_1)) %>% 
  mutate(pred_reg_1 = mu_hat + b_i_reg + b_u_reg + b_y + b_n + b_m + b_g + b_d) %>% 
  ungroup()


#final validation test
validation <- validation %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  left_join(year_avgs, by = "year_diff") %>% 
  left_join(n_rating_avgs, by = "n_ratings") %>% 
  left_join(since_first_avgs, by = "months_since_first") %>% 
  left_join(genre_avgs, by = "genres") %>% 
  left_join(day_avgs, by = "daystamp")  %>% 
  left_join(movie_sums, by = "movieId") %>% 
  mutate(b_i_reg = s / (n_i + lambda_tune)) %>% ungroup() %>%
  left_join(user_sums, by = "userId") %>% 
  mutate(b_u_reg = (b_u*n_u)/(n_u + lambda_tune_1)) %>% ungroup %>%
  mutate(pred_final = mu_hat + b_i_reg + b_u_reg + b_y + b_n + b_m + b_g + b_d)
calc_RMSE(validation$pred_final, validation$rating)


  
