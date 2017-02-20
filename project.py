from flask import Flask
from flask import request
from flask import render_template
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

f = open('NaiveBayes.pickle', 'rb')
classifier = pickle.load(f)
f.close()

tfidf_transformer = TfidfTransformer()
count_vect = CountVectorizer()

@app.route('/')
def my_form():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def my_form_post():

    if request.method == 'POST':
        text = request.form['text']

        X_new_counts = count_vect.transform(text)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        predicted = classifier.predict(X_new_tfidf)

        print(predicted)

if __name__ == '__main__':
    app.run()