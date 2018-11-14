#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:51:24 2018

@author: dhvanikansara
"""

import pandas as pd 
import numpy as np      
train = pd.read_csv("imdb.tsv", header=0, delimiter="\t", quoting=3)

from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def review_to_words( raw_review ):
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
    ps = PorterStemmer()
    tense_word = [ps.stem(word) for word in meaningful_words if not word in set(stopwords.words('english'))]

    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( tense_word ))

num_reviews = train["review"].size
clean_train_reviews = []


for i in range( 0, num_reviews ):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append( review_to_words( train["review"][i] ) )

word_reviews = []
word_unlabeled = []
all_words = []
#split sentences into words
for review in clean_train_reviews:
    word_reviews.append(review.lower().split())
    for word in review.split():
        all_words.append(word.lower())

#Assign integer numbers to each word
from collections import Counter
counter = Counter(all_words)
vocab = sorted(counter, key=counter.get, reverse=True)
vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}

#Create Word Vectors for each review
reviews_to_ints = []
for review in word_reviews:
    reviews_to_ints.append([vocab_to_int[word] for word in review])
        
y = train.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(reviews_to_ints,y,test_size=0.2,random_state=0)

#make inputs of same length by padding or truncating
from keras.preprocessing import sequence
X_train = sequence.pad_sequences(X_train,maxlen=300)
X_test = sequence.pad_sequences(X_test,maxlen=300)

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM
#RNN_LSTM-------------------------
model = Sequential()
model.add(Embedding(input_dim=30000,output_dim=128))
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=32,epochs=1,verbose=2,validation_data=(X_test,y_test))
# Final evaluation of the model
acc = model.evaluate(X_test,y_test,verbose=0)
print("Accuracy of LSTM: %.2f%%" % (acc[1]*100))

from keras.layers import Flatten
#CNN_LSTM-----------------------
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
model = Sequential()
model.add(Embedding(input_dim=30000,output_dim=100))
model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, verbose=2)
# Final evaluation of the model
acc_cl = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of CNN-LSTM: %.2f%%" % (acc_cl[1]*100))

#CNN----------------------------
model = Sequential()
model.add(Embedding(input_dim=30000,output_dim=100,input_length=300))
model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32, verbose=2)
# Final evaluation of the model
acc_c = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of CNN: %.2f%%" % (acc_c[1]*100))






