The above code uses random forest classifier and logistic regression for predicting the sentiment in the given data sets The first approach I used is to split up the datasets into two training(80%) and test/validation(20%) from the given dataset the model trains by looking the reviews and sentiments of 80% of the data and predicts the sentiment of rest 20% and check with the actual results so with this method we can get confusion matrix, accuracy, f1 score and more it is like the model studies pattern of 80% of data and tries itself for the rest and check with results and confusion matrix accuracy are just its tests outcomes inform of percentage or exact numbers then we vectorize data as the dataset consists of string models cant train with this efficiently so we vectorize it with most occurring words(for a movie lets say good bad worst decent are most occurring words in a review then the model takes it as vector and vectorize it to array just turns string to numbers then we use randomforestclassifier(It works by building many different decision trees on random parts of the data, then combining their predictions to get the final answer.)with an estimate of 200 (more estimates would improve accuracy but takes more time) and we fit up our vectorized array in this model
and then we generate accuracy,f1 score ,results and also a confusion matric(inform of array and graph) and then we use logistic regression model so this model is ideal for string type datasets with less no of integers We have a dataset of text (e.g., reviews) and labels (e.g., 1 for positive, 0 for negative).
We transform the text into numerical features using TF-IDF vectorization.
Logistic Regression  learns to predict the probability of a review being positive or negative based on these features.
it has a threshold probability of 0.5
and then we get results of logistic regression model and compare its output with randomforestclassifier(in our code the logistic regression performs a bit better)
and then we compare training and validation accuracy for both models
at last we get a user input and try both the models and generates answers from both the models
key libraries imported:
matplotlib
 numpy
sklearn
seaborn
