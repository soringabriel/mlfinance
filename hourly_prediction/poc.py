import csv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

training_data = pd.read_csv('data.txt', header=None)
X = np.array(training_data.iloc[:, 0:60])
y = np.array(training_data.iloc[:, 60])

to_predict_data = pd.read_csv('topredict.txt', header=None)
to_predict = np.array(to_predict_data.iloc[:, 0:60])
predictions = {}

for n in range(30,900):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X, y)

    prediction = knn.predict(to_predict)
    if prediction[0] in predictions.keys():
        predictions[prediction[0]] += 1
    else:
        predictions[prediction[0]] = 1

p = max(predictions, key=predictions.get)

pred = [list(to_predict[0])]
pred[0].append(p)

with open("predictions.txt", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(pred)