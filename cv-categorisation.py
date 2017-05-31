# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
import sys
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def get_data(key, categories):
    dataset = []
    target = []
    for idx, category in enumerate(categories):
        for root, dirs, files in os.walk(os.path.join('cv', key, category)):
            for f in files:
                with open(os.path.join(root, f)) as inputfile:
                    dataset.append(inputfile.read())
                    target.append(idx)
    return dataset, target

categories = ['datascience', 'software']
(train_dataset, train_target) = get_data('train', categories)

# bag of words, tokenizing
count_vect = CountVectorizer()
# words frequency vector 
X_train_counts = count_vect.fit_transform(train_dataset)

#Term Frequency times Inverse Document Frequency
tfidf_transformer = TfidfTransformer(use_idf=False)
#tf_transformer = tfidf_transformer.fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# train a classifier
clf = MultinomialNB().fit(X_train_tfidf, train_target)

# test accuracy
import numpy as np
(test_dataset, test_target) = get_data('test', categories)

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(train_dataset, train_target)
predicted = text_clf.predict(test_dataset)
print(predicted, test_target)
print(np.mean(predicted == test_target))

sys.exit(1)
# categorise a new text
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


