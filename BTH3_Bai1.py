from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = load_iris()

for target in set(data.target):e
    x = [data.data[i, 0]
         for i in range(len(data.target)) if data.target[i] == target]
    y = [data.data[i, 1]
         for i in range(len(data.target)) if data.target[i] == target]
    plt.scatter(x, y, color=['red', 'blue', 'green']
                [target], label=data.target_names[target])
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title('Scatter plot')
plt.legend(data.target_names, loc='lower right')
plt.show()

features = data.data[:len(data.data),:2]
labels = data.target

train_x, train_y = features[:len(features)], labels[:len(labels)]
test_x, test_y = features[0:], labels[0:]

print("Gaussian Naive Bayesian Classifier")

clf = GaussianNB()
clf.fit(train_x, train_y)

predictions = clf.predict(test_x)
# print("pred -> o/p")
# for i in range(len(predictions)):
#     print(predictions[i], "->", test_y[i])

#print(classification_report(test_y, predictions))
ArrFault = confusion_matrix(test_y, predictions) 
#print(ArrFault)
print("Phân đúng lớp Setosa", ArrFault[0][0], "sai ở lớp Versicolor" ,ArrFault[0][1], "và sai lớp ở Virginica" ,ArrFault[0][2])
print("Phân đúng lớp Versicolor", ArrFault[1][1], "sai ở lớp Setosa" ,ArrFault[1][0], "và sai lớp ở Virginica" ,ArrFault[1][2])
print("Phân đúng lớp Virginica", ArrFault[2][2], "sai ở lớp Setosa" ,ArrFault[2][0], "và sai lớp ở Versicolor" ,ArrFault[2][1])
print("Lỗi ở lớp Setosa", ArrFault[0][1]+ArrFault[0][2])
print("Lỗi ở lớp Versicolor", ArrFault[1][0]+ArrFault[1][2])
print("Lỗi ở lớp Virginica", ArrFault[2][0]+ArrFault[2][1])
print("Tổng lỗi", ArrFault[0][1]+ArrFault[0][2]+ArrFault[1][0]+ArrFault[1][2]+ArrFault[2][0]+ArrFault[2][1])

acc = metrics.accuracy_score(test_y, predictions)
print("Độ chính xác", acc)

print("Multinomial Naive Bayesian Classifier")

clf1 = MultinomialNB()
clf1.fit(train_x, train_y)

predictions1 = clf1.predict(test_x)
# print("pred -> o/p")
# for i in range(len(predictions1)):
#     print(predictions1[i], "->", test_y[i])

#print(classification_report(test_y, predictions1))
ArrFault1 = confusion_matrix(test_y, predictions1) 
#print(ArrFault)
print("Phân đúng lớp Setosa", ArrFault1[0][0], "sai ở lớp Versicolor" ,ArrFault1[0][1], "và sai lớp ở Virginica" ,ArrFault1[0][2])
print("Phân đúng lớp Versicolor", ArrFault1[1][1], "sai ở lớp Setosa" ,ArrFault1[1][0], "và sai lớp ở Virginica" ,ArrFault1[1][2])
print("Phân đúng lớp Virginica", ArrFault1[2][2], "sai ở lớp Setosa" ,ArrFault1[2][0], "và sai lớp ở Versicolor" ,ArrFault1[2][1])
print("Lỗi ở lớp Setosa", ArrFault1[0][1]+ArrFault1[0][2])
print("Lỗi ở lớp Versicolor", ArrFault1[1][0]+ArrFault1[1][2])
print("Lỗi ở lớp Virginica", ArrFault1[2][0]+ArrFault1[2][1])
print("Tổng lỗi", ArrFault1[0][1]+ArrFault1[0][2]+ArrFault1[1][0]+ArrFault1[1][2]+ArrFault1[2][0]+ArrFault1[2][1])

acc1 = metrics.accuracy_score(test_y, predictions1)
print(acc1)

print('RESULT IRIS CLASSIFIER')
train_x = train_x.T
merged = np.array([train_x[0], train_x[1], train_y,
                   predictions, predictions1])
mt = merged.T
name = data.target_names


def rename(Ip):
    newl = []
    for i in range(len(Ip)):
        if Ip[i] == np.float64(0.0):
            newl.append(name[0])
        elif Ip[i] == np.float64(1.0):
            newl.append(name[1])
        else:
            newl.append(name[2])
    return np.array(newl)


train_y = rename(merged[2])
pred = rename(merged[3])
pred1 = rename(merged[4])

merged = mt.T

dataFinal = {'Sepal Length': merged[0].T, 'Sepal width': merged[1].T,
             'Species': train_y.T, 'Gaussian Naive Bayesian Classifier': pred.T, 'Multinomial Naive Bayesian Classifier': pred1.T}
dataFinalFrame = pd.DataFrame.from_dict(dataFinal)
print(dataFinalFrame)

dataFinalFrame.to_csv(r'IrisClassifier.csv')