import csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

past_data_size = 120

training_data = pd.read_csv('data.txt', header=None)
X = np.array(training_data.iloc[:, 0:past_data_size])
y = np.array(training_data.iloc[:, past_data_size])

to_predict_data = pd.read_csv('topredict.txt', header=None)
to_predict = np.array(to_predict_data.iloc[:, 0:past_data_size])
predictions = {}

# KNN (with different K and taking the biggest predicted value + with K = 30)
p_knn_30 = 0

for n in range(30,len(training_data)):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X, y)

    prediction = knn.predict(to_predict)
    
    if n == 30:
        p_knn_30 = prediction[0]
        
    if prediction[0] in predictions.keys():
        predictions[prediction[0]] += 1
    else:
        predictions[prediction[0]] = 1

p_knn_max = max(predictions, key=predictions.get)

# SVM
clf = svm.SVC()
clf.fit(X, y)
p_svm = clf.predict(to_predict)[0]

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X, y)
p_naive_bayes = gnb.predict(to_predict)[0]

# Decision tree
decissiontree = tree.DecisionTreeClassifier()
decissiontree = decissiontree.fit(X, y)
p_decission_tree = decissiontree.predict(to_predict)[0]

# Random forests
rf = RandomForestClassifier(max_depth=2, random_state=0)
rf = rf.fit(X, y)
p_random_forests = rf.predict(to_predict)[0]

pred = [list(to_predict[0])]
pred[0].append(p_knn_30)
pred[0].append(p_knn_max)
pred[0].append(p_svm)
pred[0].append(p_naive_bayes)
pred[0].append(p_decission_tree)
pred[0].append(p_random_forests)

with open("predictions.txt", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pred)