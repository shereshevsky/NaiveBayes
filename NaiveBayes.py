from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from scipy.sparse import coo_matrix

ng = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

X_train = ng.data
y_train = ng.target
target_names = ng.target_names
test_data = fetch_20newsgroups(subset='test')


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.prior_class_prob = None
        self.cond_prob = None
        self.total_labels = None

    def fit(self, x, y):
        # get label names and count per label
        label_names, label_counts = np.unique(y, return_counts=True)
        self.total_labels = label_counts.size
        # calculate prior per label (class)
        self.prior_class_prob = label_counts / self.total_labels
        # Calculating conditional probabilities using Laplace smoothing
        self.cond_prob = np.vstack([np.array((x[y_train == i].sum(axis=0) + 1.))[0] /
                                    (x[y_train == i].sum() + x.shape[1]) for i in label_names])

    def predict(self, x):
        res = []
        for doc in x:
            # convert each document to class x features matrix
            matrix_with_cond = coo_matrix(doc.toarray() * self.cond_prob)
            # replace all matrix values with log(value)
            matrix_with_cond.data = np.log(matrix_with_cond.data)
            # get top class for the document
            res.append(
                np.argmax(matrix_with_cond.sum(axis=1).reshape(1, self.total_labels) + np.log(self.prior_class_prob)))
        return res

    def predict_log_proba(self, x):
        res = []
        for doc in x:
            matrix_with_cond = coo_matrix(doc.toarray() * self.cond_prob)
            matrix_with_cond.data = np.log(matrix_with_cond.data)
            prob = matrix_with_cond.sum(axis=1).reshape(1, self.total_labels) + np.log(self.prior_class_prob)
            res.append(prob / 100.)
        return np.vstack(res)


pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english', max_features=10000, strip_accents='ascii')),
    ('tfidf', TfidfTransformer()),
    # ('clf', MultinomialNB()),
    ('clf', NaiveBayes()),
])

pipeline.fit(X_train, y_train)

predicted = pipeline.predict(test_data.data)
print("Accuracy=", sum(test_data.target == predicted) / test_data.target.size)
