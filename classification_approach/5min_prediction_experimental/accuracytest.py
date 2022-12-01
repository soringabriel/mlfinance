import numpy as np
import pandas as pd

past_data_size = 480

predictions = pd.read_csv('predictions.txt', header=None)
predictions = np.array(predictions.iloc[:, 0:past_data_size + 6])

data = pd.read_csv('data.txt', header=None)
data = np.array(data.iloc[-len(predictions):, 0:past_data_size + 1])

acc_knn_30 = []
acc_knn_max = []
acc_svm = []
acc_naive_bayes = []
acc_decision_tree = []
acc_rf = []
acc_max = []

for x in range(len(predictions)):
    classification = data[x][past_data_size]
    knn_30 = predictions[x][past_data_size]
    knn_max = predictions[x][past_data_size + 1]
    svm = predictions[x][past_data_size + 2]
    naive_bayes = predictions[x][past_data_size + 3]
    decision_tree = predictions[x][past_data_size + 4]
    rf = predictions[x][past_data_size + 5]
    acc_knn_30.append(1 if knn_30 == classification else 0)
    acc_knn_max.append(1 if knn_max == classification else 0)
    acc_svm.append(1 if svm == classification else 0)
    acc_naive_bayes.append(1 if naive_bayes == classification else 0)
    acc_decision_tree.append(1 if decision_tree == classification else 0)
    acc_rf.append(1 if rf == classification else 0)
    sum_correct = acc_knn_30[-1] + acc_knn_max[-1] + acc_svm[-1] + acc_naive_bayes[-1] + acc_decision_tree[-1] + acc_rf[-1]
    acc_max.append(1 if sum_correct > 3 else 0)
    print(data[x][0],predictions[x][0],classification,knn_30,knn_max,svm,naive_bayes,decision_tree,rf,sum_correct)
    
print("Accuracy for knn 30", sum(acc_knn_30) / len(acc_knn_30))
print("Accuracy for knn max", sum(acc_knn_max) / len(acc_knn_max))
print("Accuracy for svm", sum(acc_svm) / len(acc_svm))
print("Accuracy for naive bayes", sum(acc_naive_bayes) / len(acc_naive_bayes))
print("Accuracy for decision tree", sum(acc_decision_tree) / len(acc_decision_tree))
print("Accuracy for rf", sum(acc_rf) / len(acc_rf))
print("Accuracy for max", sum(acc_max) / len(acc_max))



