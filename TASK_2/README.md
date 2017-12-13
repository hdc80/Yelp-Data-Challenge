## TASK 2

## ML based approach
__Ignore ML_based_approach_old.py__. It is only using attributes of the reviews as features.
*ML based approach (1).py* use different types of features like usefulness, coolness, funny attributes of reviews also tfidf scores of the reviews. Applied PCA on the features and used 3 principal components as they covered 97.5% of the variance of the data. Used different ML models like Naive Bayes, Linear Regression, Logistic Regression, SVM and Random Forest to fit the data.

## LSTM model (Task2.py)

### Dataset used:  sorted_part2_update.csv

This code generates the prediction of review count based on the neural net (LSTM). It runs for 2000 epoch with 1 hidden layer and 10 hidden unit. The look back chosen is 4 based on the analysis done. 

Result: 1.4614 RMSE

## ARIMA model (ARIMA_VERSION.IPYNB)
### Dataset used:  sorted_part2.csv

This code generate the prediction based on the ARIMA model. 

Result: 0.274

