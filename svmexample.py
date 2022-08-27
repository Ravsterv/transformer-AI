from sklearn.model_selection import train_test_split
from transformer import FeatureDataset
from sklearn import svm
from sklearn import metrics
import numpy as np


transformer_data = FeatureDataset("NewHI-Modnew.csv")

x_train, X_test, y_train, y_test = train_test_split(transformer_data.data, transformer_data.target, test_size=0.3, random_state=25)

clf = svm.SVC(kernel="linear")

clf.fit(x_train.astype('int'), y_train.astype('int'))

y_pred = clf.predict(X_test)
#TODO: Works but accuracy is only 0.212
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))


