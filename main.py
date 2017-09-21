import pandas as pd
from gensim import corpora
from collections import defaultdict
from sklearn import svm
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
import scipy.stats as st
import sys
import math
import numpy as np
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
 
 
class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self

def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)
    


def converT2V(vec):
    temp = [0]*1000
    for each in vec:
        if each[0] < 1000:
            temp[each[0]]=each[1]
    return temp    
        

df = pd.read_csv('train.csv')
response = df["Is_Response"].tolist()
documents = df["Description"].tolist()


print ('Procesing the data')
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
freq = defaultdict(int)
for text in texts:
    for token in text:
        freq[token] += 1

texts = [[token for token in text if freq[token] > 1] for text in texts]
dictionary = corpora.Dictionary(texts)
print ('Dictionary created')

print('started loading the training data')
X = []
for i,each in enumerate(documents):
    vec = dictionary.doc2bow(each.lower().split())
    vec = converT2V(vec)
    X.append(vec)

y = []
for each in response:
    if each=='happy':
        y.append(1)
    else:
        y.append(0)    
print 'Done procesing the data'



clf = XGBoostClassifier(
        eval_metric = 'auc',
        num_class = 2,
        nthread = 4,
        silent = 1,
        )
parameters = {
        'num_boost_round': [100, 250, 500],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
}
clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
    
clf.fit(X,y)
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('score:', score)
for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))


print('loading the testing data')
df_test = pd.read_csv('test.csv')
user_id = df_test["User_ID"].tolist()
desc    = df_test["Description"].tolist()
X_test = []
for i,each in enumerate(desc):
    vec = dictionary.doc2bow(each.lower().split())
    vec = converT2V(vec)
    X_test.append(vec)

print 'Done with loding data'
predictions = clf.predict(X_test) 
print 'Done predictions'


"""


clf = svm.SVC()
clf.fit(X, y)
print 'Done fitting the data into classifier'


df_test = pd.read_csv('test.csv')
user_id = df_test["User_ID"].tolist()
desc    = df_test["Description"].tolist()


print('loading the testing data')
X_test = []
for i,each in enumerate(desc):
    vec = dictionary.doc2bow(each.lower().split())
    vec = converT2V(vec)
    X_test.append(vec)

print 'Done with loding data'
predictions = clf.predict(X_test) 
print 'Done predictions'

"""

print ('Wrting to the data')
with open('submit.csv','w') as f:
    f.write('User_ID,Is_Response\n')
f.close()    
for i in range(len(user_id)):
    with open('submit1.csv','a') as f:
        if predictions[i]==1:
            f.write(str(user_id[i])+','+'happy\n')
        else:
            f.write(str(user_id[i])+','+'not_happy\n')
    f.close()

print 'Successfully completed the program execution\n see the output in submit.csv'    


