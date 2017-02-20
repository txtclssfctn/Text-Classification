from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import os
import pickle

directory = 'C:/Users/Kieran/FYP/bbc/bbc/'

topics = {'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}

doc_data = []
doc_topic = []

# Collect File Content and Topics
for topic, value in topics.items():

    topic_directory = directory + topic + '/'
    for files in os.walk(topic_directory):
        file_array = files[2]                           #files[2] is where the array of file names is stored
        for file in file_array:
            file = open(topic_directory + file, "r")

            text = file.read()
            doc_data.append(text)
            doc_topic.append(value)

            file.close()

# Build Naive Bayes
NB_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=2, max_df=0.99)),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB())])
NB_clf = NB_clf.fit(doc_data, doc_topic)

# Save Naive Bayes
f = open('NaiveBayes.pickle', 'wb')
pickle.dump(NB_clf, f)
f.close()

#Build SVM
SGD_clf = Pipeline([('vect', CountVectorizer(stop_words='english', min_df=10, max_df=0.5)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42))])
SGD_clf = SGD_clf.fit(doc_data, doc_topic)

#Save SVM
f = open('SGD.pickle', 'wb')
pickle.dump(SGD_clf, f)
f.close()

#Build a basic text box website and then go onto bootstrap.

#1. Basic HTML Website built.