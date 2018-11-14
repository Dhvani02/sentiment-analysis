# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:18:30 2018

@author: dhvanikansara
"""
#import libraries
import pandas as pd 
import numpy as np      
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#import dataset
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

#function for cleaning
def review_to_words( raw_review ):
    # 1. remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    import re    
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. keep only root words
    ps = PorterStemmer()
    tense_word = [ps.stem(word) for word in meaningful_words if not word in set(stopwords.words('english'))]

    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( tense_word ))   
    

num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

#apply cleaning on all reviews
for i in range( 0, num_reviews ):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append( review_to_words( train["review"][i] ) )

#create bag of words    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)   
train_data_features = cv.fit_transform(clean_train_reviews).toarray()

y = train.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(train_data_features,y,test_size=0.2,random_state=0)

#Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

print('Accuracy of Logistic Regression classifier on test set: {:.4f}'.format(forest.score(X_test, y_test)))

#Naive BAyes classifier
from sklearn.naive_bayes import GaussianNB
classifier_n = GaussianNB()
classifier_n.fit(X_train, y_train)
y_pred = classifier_n.predict(X_test)
print('Accuracy of Naive Bayes classifier on test set: {:.4f}'.format(classifier_n.score(X_test, y_test)))

#Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
forest = forest.fit(X_train,y_train)
y_pred = forest.predict(X_test)
print('Accuracy of Random Forest classifier on test set: {:.4f}'.format(forest.score(X_test, y_test)))

#Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier_d = DecisionTreeClassifier(criterion='entropy') 
classifier_d.fit(X_train, y_train)
y_pred = classifier_d.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.4f}'.format(classifier_d.score(X_test, y_test)))

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier_k = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2) 
classifier_k.fit(X_train,y_train)
y_pred = classifier_k.predict(X_test)
print('Accuracy of KNN classifier on test set: {:.4f}'.format(classifier_k.score(X_test, y_test)))






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    