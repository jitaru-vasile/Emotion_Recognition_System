from FaceDetector import FaceDetector
import numpy as np
import pickle
from Reader import Reader
from kNN import KNearestNeighbors
from distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

reader = Reader()
(trainingData, testingData) = reader.read_from('D:/Facultate/Licenta/CS229-master/JAFFE')
dataset = reader.read_from_with_target('D:/Facultate/Licenta/CS229-master/JAFFE')

count = 0
X = []
y = []
for data in dataset:
    faceDetector = FaceDetector()
    hogImage = faceDetector.detect(data)
    X.append(hogImage.fd)
    y.append(hogImage.target)

X = np.array(X, dtype=object)
y = np.array(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# knn train model
model = KNearestNeighbors(k=1, distance_metric=euclidean)
model.train(X_train, y_train)
knnPickle = open('knnpickle_file', 'wb')
pickle.dump(model, knnPickle)
knnPickle.close()
print(model.predict(X_test))
print(model.score(X_test, y_test))

# svm model
C = 2.0
svmModel = svm.LinearSVC(C=C, max_iter=10000)
svmModel.fit(X_train, y_train)
print(svmModel.predict(X_test))
print(svmModel.score(X_test, y_test))

# FR
randomForrest = RandomForestClassifier(n_estimators=300, max_features='auto')
randomForrest.fit(X_train, y_train)
print(randomForrest.predict(X_test))
print(randomForrest.score(X_test, y_test))

# Adaboost
decisionTree = DecisionTreeClassifier(max_depth=10)
adaBoostModel = AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate=1.5, algorithm='SAMME', base_estimator=decisionTree)
adaBoostModel.fit(X_train, y_train)
print(adaBoostModel.predict(X_test))
print(adaBoostModel.score(X_test, y_test))

# Logistic Regression
logisticRegressionModel = LogisticRegression(max_iter=500, solver='saga', penalty="l2", multi_class='multinomial', warm_start=True)
logisticRegressionModel.fit(X_train, y_train)
print(logisticRegressionModel.predict(X_test))
print(logisticRegressionModel.score(X_test, y_test))

