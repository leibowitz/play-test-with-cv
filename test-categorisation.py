# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print(twenty_train.data[0])
print(twenty_train.target)
print(twenty_train.target_names)
# bag of words, tokenizing
count_vect = CountVectorizer()
# words frequency vector 
X_train_counts = count_vect.fit_transform(twenty_train.data)

#Term Frequency times Inverse Document Frequency
tfidf_transformer = TfidfTransformer(use_idf=False)
#tf_transformer = tfidf_transformer.fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# train a classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# categorise a new text
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


# test accuracy
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

docs_test = twenty_test.data
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
