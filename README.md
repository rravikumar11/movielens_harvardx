# movielens_harvardx

Predictive model guessing the rating over a movie with user and time data. Part of the capstone for HarvardX's Data Science certification.

Starting with the fundamentals established in the HarvardX Data Science: Machine Learning course, we develop the model further by including more effects and incorporating regularization. Through these efforts, we achieve a residual mean square error (RMSE), essentially the average error of a given prediction, of 0.864525.

This model utilizes the following variables (some included in the initial MovieLens dataset, and some created from existing variables):

**rating:** The value we are trying to predict; the rating left by a user on a movie out of 5 points.  
**movieID:** A unique numerical identifier for each movie.  
**userID:** A unique numerical identifier for each user.  
**n_ratings:** The number of reviews left on each movie.  
**timestamp:** The exact time each review was left.  
**genres:** A collection of the genres to which a movie belongs.    
**date:** The timestamp variable, converted to the useful "Date" format.  
**yearstamp:** The year in which each review was left.  
**year_released:** The year in which each movie was released.  
**timestamp_round:** The date on which a review was left, rounded down to the first of the month.  
**months_since_first:** A measure of the time between a user's first review and the current one (in months).  
**year_diff:** The number of years between the time of the review and the release of the movie (in years).  
**daystamp:** A measure of the day of the week on which the review was left (where Sunday is 1, Monday is 2, and so on.)  
