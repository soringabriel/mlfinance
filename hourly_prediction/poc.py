import csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

past_data_size = 60

training_data = pd.read_csv('data.txt', header=None)
X = np.array(training_data.iloc[:, 0:past_data_size])
y = np.array(training_data.iloc[:, past_data_size])

to_predict_data = pd.read_csv('topredict.txt', header=None)
to_predict = np.array(to_predict_data.iloc[:, 0:past_data_size])
predictions = {}

p_knn_30 = 0

for n in range(30,900):
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

clf = svm.SVC()
clf.fit(X, y)
p_svm = clf.predict(to_predict)[0]

pred = [list(to_predict[0])]
pred[0].append(p_knn_30)
pred[0].append(p_knn_max)
pred[0].append(p_svm)

with open("predictions.txt", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pred)