import joblib as jl
import pandas, xgboost, numpy, textblob, string
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)
    
    
X = jl.load('Alice/X_train')
Y = jl.load('Alice/Y_train')

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['X'] = X
trainDF['Y'] = Y
print(trainDF)

#X_train = 
#Y_train = 
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=42)
classifier = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),
        ('length', Pipeline([
            ('count', FunctionTransformer(get_text_length, validate=False)),
        ]))
    ])),
    ('clf', LinearSVC(max_iter=50000))])

classifier.fit(X_train, y_train)
print('Score on train data:', classifier.score(X_train, y_train))
print('Score on test data:', classifier.score(X_test, y_test))

s = input("Check your standup!\n")
print(classifier.predict([s]))