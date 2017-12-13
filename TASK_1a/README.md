TASK 1 (a) - Recommendation based on Aspect Opinion Mining 

## Sentiment_Code_Update.ipynb
This code will calculate sentiment score for reviews on sentence level and will generate 'output_sent_update.csv' which will be ued as input to other program code.

## Categorical_sentiment_analysis_part_1a.ipynb:

### Dataset used: 'bible_word2vec_gensim', 'output_sent_update.csv'

This code will first load the word2vec model 'bible_word2vec_gensim' which is trained on yelp reviews.
It will generate category_new_update.csv which contains category wise sentiment analysis based on word2vect model.

 ## evaluation_part_1a.ipynb:
 
 ### Dataset used: 'category_new_update.csv'

This code will take input from previous code "category_new_update.csv" and will evaluate our algorithm for task1a.

Evaluation Results: RMSE : 1.63
